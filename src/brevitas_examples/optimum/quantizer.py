from functools import partial
import logging
from pathlib import Path
from types import MethodType

from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from optimum.exporters.onnx.__main__ import onnx_export
from optimum.quantization_base import OptimumQuantizer
import torch
from torch.fx import GraphModule as TorchGraphModule

from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas.fx import GraphModule
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer
from brevitas.utils.python_utils import recurse_getattr
from brevitas_examples.common.generative.quantize import quantize_model
from brevitas_examples.llm.llm_quant.calibrate import apply_calibration
from brevitas_examples.llm.llm_quant.data import get_c4
from brevitas_examples.llm.llm_quant.data import get_wikitext2
from brevitas_examples.llm.llm_quant.equalize import apply_act_equalization
from brevitas_examples.llm.llm_quant.equalize import apply_weight_equalization
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from brevitas_examples.llm.llm_quant.gptq import apply_gptq
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_mha_with_quantizable_layers
from brevitas_examples.llm.llm_quant.run_utils import get_block_name_with_pattern
from brevitas_examples.llm.llm_quant.run_utils import get_fx_graph
from brevitas_examples.optimum.utils import generate_dummy_inputs
from brevitas_examples.optimum.utils import get_forward_signature

from .config import BrevitasQuantizationConfig

LOGGER = logging.getLogger(__name__)


class BrevitasQuantizer(OptimumQuantizer):
    """
    Handles the Runtime quantization process for models shared on huggingface.co/models.
    """

    def __init__(self, model: torch.nn.Module, qconfig: BrevitasQuantizationConfig):

        super().__init__()
        self.model = model
        self.qconfig = qconfig
        try:
            self.config = self.model.config
        except OSError:
            LOGGER.warning(f"Could not load the config automatically.")
        self.validate_quant_config()
        self.group_of_parallel_layers = None

    def validate_quant_config(self):
        dtype = next(iter(self.model.parameters()))
        if dtype == torch.bfloat16 and self.qconfig.replace_mha_with_quantizable:
            raise RuntimeError("Scaled_dot_product does not support bfloat16 and cuda")
        if self.qconfig.input_quant_granularity == 'per_row' and not self.qconfig.replace_mha_with_quantizable:
            raise RuntimeError(
                "Per-row act quant requires setting replace_mha_with_quantizable to True")
        if self.qconfig.quantization_format == 'graph_mode':
            raise RuntimeError("FX quantization not yet compatible with accelerate")

    def find_groups_of_parallel_layers(self, names_of_groups_of_parallel_layers):
        names = [name for name, _ in self.model.named_modules()]
        group_of_parallel_layers = []
        set_found_layers = set()
        for group in names_of_groups_of_parallel_layers:
            first_name = group[0]
            for name in names:
                if name.endswith(first_name) and name not in set_found_layers:
                    all_names_present = True
                    prefix = name.removesuffix(first_name)
                    for name_in_group in group:
                        if not prefix + name_in_group in names:
                            all_names_present = False
                    if all_names_present:
                        found_names = [prefix + name_in_group for name_in_group in group]
                        group_of_parallel_layers.append(found_names)
                        set_found_layers.update(found_names)
        self.group_of_parallel_layers = group_of_parallel_layers

    def quantize(self, model, calib_dataloader, forward_call):
        dtype = next(iter(model.parameters())).dtype

        # Insert standard MHA layers when performing fx based weight/act equalization to avoid dealing
        # with all the variability in HF implementations
        if self.qconfig.replace_mha_with_quantizable:
            print("Replace HF MHA with quantizable variants...")
            model = replace_mha_with_quantizable_layers(model, dtype)
            print("Replacing done.")

        # Because accelerate is not compatible with FX, we keep two versions of the Model
        # one with FX-traced, the other one not.
        # Since weights are shared across the two, we can apply weight/activation equalization
        # by using one representation or the other based on needs.

        if self.qconfig.apply_weight_equalization:
            print("Apply weight equalization...")
            apply_weight_equalization(model)
            print("Weight equalization applied.")

        if self.qconfig.apply_act_equalization is not None:
            print("Apply Act Equalization (SmoothQuant)")
            apply_act_equalization(
                model, self.qconfig.apply_act_equalization, calib_dataloader, forward_call)
            print("Act equalization applied")

        # We do not quantize embedding and last fully connected layer
        model = quantize_model(
            model,
            dtype=dtype,
            weight_quant_format='int',
            weight_quant_type=self.qconfig.weight_quant_type,
            weight_bit_width=self.qconfig.weight_bit_width,
            weight_param_method=self.qconfig.weight_param_method,
            weight_scale_precision=self.qconfig.scale_precision,
            weight_quant_granularity=self.qconfig.weight_quant_granularity,
            weight_group_size=self.qconfig.weight_group_size,
            quantize_weight_zero_point=self.qconfig.quantize_zero_point,
            input_bit_width=self.qconfig.input_bit_width,
            input_quant_type=self.qconfig.input_quant_type,
            input_quant_format='int',
            input_param_method=self.qconfig.input_param_method,
            input_scale_precision=self.qconfig.scale_precision,
            input_scale_type=self.qconfig.input_scale_type,
            input_quant_granularity=self.qconfig.input_quant_granularity,
            input_group_size=self.qconfig.input_group_size,
            quantize_input_zero_point=self.qconfig.quantize_zero_point,
            seqlen=self.qconfig.seqlen)

        # Perform a single inference pass to generate the correct state_dict
        with torch.no_grad():
            forward_call(model, calib_dataloader[0])

        if self.qconfig.apply_gptq:
            print("Apply gptq")
            apply_gptq(
                model,
                calib_dataloader,
                forward_call,
                group_of_parallel_layers=self.group_of_parallel_layers)
            print("GPTQ applied")

        if self.qconfig.input_bit_width is not None and self.qconfig.input_scale_type == 'static':
            print("Apply act calibration...")
            apply_calibration(model, calib_dataloader, forward_call)
            print("Act calibration applied.")
        return model

    def get_calibration_dataloader(
        self,
        tokenizer,
        dataset_name='c4',
        num_samples: int = 100,
        seqlen: int = 2048,
        seed: int = 0,
    ):
        if dataset_name == 'c4':
            calib_dataloader = get_c4(
                nsamples=num_samples, seed=seed, tokenizer=tokenizer, seqlen=seqlen)
        elif dataset_name == 'wikitext2':
            calib_dataloader = get_wikitext2(
                nsamples=num_samples, seqlen=seqlen, seed=seed, tokenizer=tokenizer, type='')
        elif dataset_name == 'wikitext2-raw':
            calib_dataloader = get_wikitext2(
                nsamples=num_samples, seqlen=seqlen, seed=seed, tokenizer=tokenizer, type='raw')

        return calib_dataloader

    def export(self, model, export_path):
        export_class = StdQCDQONNXManager

        # When exporting large model, it is better to explicitly export the floating point weight
        # followed by quantize-dequantize, instead of integer weights + dequantize.
        # PyTorch ONNX export seems to run in some form of weight duplication with integer weights,
        # causing the export to fail because the total model is over 2GB.
        export_class.change_weight_handler(export_quantize_node_weight=True)
        # workaround for FX model
        extra_kwargs = {}
        if isinstance(self.model, (GraphModule, TorchGraphModule)):
            extra_kwargs['sequence_length'] = 1

        with torch.no_grad(), brevitas_proxy_export_mode(model, export_class):
            onnx_export(
                model,
                Path(export_path),
                task="text-generation-with-past",
                do_validation=False,
                **extra_kwargs)
