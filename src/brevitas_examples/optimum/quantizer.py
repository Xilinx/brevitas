from functools import partial
import logging
from pathlib import Path
from types import MethodType

from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from optimum.quantization_base import OptimumQuantizer
import torch

from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
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

    def validate_quant_config(self):
        dtype = next(iter(self.model.parameters()))
        if dtype == torch.bfloat16 and self.qconfig.replace_mha_with_quantizable:
            raise RuntimeError("Scaled_dot_product does not support bfloat16 and cuda")
        if self.qconfig.input_scale_type == 'dynamic' and self.qconfig.input_quant_type == 'asym':
            raise RuntimeError("Zero point not supported for dynamic quantization")
        if self.qconfig.input_quant_granularity == 'per_row' and not self.qconfig.replace_mha_with_quantizable:
            raise RuntimeError(
                "Per-row act quant requires setting replace_mha_with_quantizable to True")
        if self.qconfig.quantization_format == 'graph_mode':
            raise RuntimeError("FX quantization not yet compatible with accelerate")

    def quantize(self, model, calib_dataloader):
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
            apply_act_equalization(model, self.qconfig.apply_act_equalization, calib_dataloader)
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
            model(calib_dataloader[0])

        if self.qconfig.apply_gptq:
            print("Apply gptq")
            apply_gptq(model, calib_dataloader)
            print("GPTQ applied")

        if self.qconfig.input_bit_width is not None and self.qconfig.input_scale_type == 'static':
            print("Apply act calibration...")
            apply_calibration(model, calib_dataloader)
            print("Act calibration applied.")
        return model

    def get_calibration_dataloader(
        self,
        model,
        dataset_name='c4',
        num_samples: int = 100,
        seqlen: int = 2048,
        seed: int = 0,
    ):
        if dataset_name == 'c4':
            trainloader, valenc = get_c4(nsamples=num_samples, seed=seed, model=model, seqlen=seqlen)
        elif dataset_name == 'wikitext2':
            trainloader, valenc = get_wikitext2(nsamples=num_samples, seqlen=seqlen, seed=seed, model=model, type='')
        elif dataset_name == 'wikitext2-raw':
            trainloader, valenc = get_wikitext2(nsamples=num_samples, seqlen=seqlen, seed=seed, model=model, type='raw')

        return trainloader, valenc

    def export(self, model, export_path, format='onnx_qcdq'):
        # Currently we always export on CPU with a float32 container to avoid float16 CPU errors
        model = model.cpu().to(dtype=torch.float32)
        model_type = model.config.model_type
        if format == 'onnx_qcdq':
            export_class = StdQCDQONNXManager
            backend = "onnx"
            with torch.inference_mode(), brevitas_proxy_export_mode(model, export_class):
                # We would like to use the export_main optimum export since it takes care also of KV Cache
                # This API does not seem to do that
                onnx_path = Path(export_path)
                onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                    backend, model, task="text-generation")
                onnx_config = onnx_config_constructor(model.config)
                # Remove all arguments from the `generate_dummy_inputs` method which are not in the GraphModule's signature
                if isinstance(model, torch.fx.graph_module.GraphModule):
                    forward_signature_keys = get_forward_signature(model)
                    onnx_config._brv_generate_dummy_inputs_orig = onnx_config.generate_dummy_inputs
                    onnx_config.generate_dummy_inputs = MethodType(
                        partial(
                            generate_dummy_inputs, forward_signature_keys=forward_signature_keys),
                        onnx_config)
                onnx_inputs, onnx_outputs = export(model, onnx_config, onnx_path, onnx_config.DEFAULT_ONNX_OPSET)
        elif format == 'torchscript_qcdq':
            raise RuntimeError("TBD")
        elif format == 'onnx_packed_int':
            raise RuntimeError("TBD")
        else:
            raise RuntimeError("Format not supported")
