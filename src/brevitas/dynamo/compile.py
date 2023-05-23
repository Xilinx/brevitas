# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import os
import tempfile
from typing import Any, List
import warnings 
from collections import OrderedDict

import torch
from torch._dynamo.backends.onnxrt import default_provider
from torch._dynamo.backends.onnxrt import _np_dtype

from brevitas.export import export_onnx_qcdq
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import finalize_collect_stats
from brevitas.graph.gptq import gptq_mode
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model


_DYNAMO_PTQ_MODE = False

ONNXRT_PROVIDERS = {
    'onnxrt_cpu': ["CPUExecutionProvider"],
    'onnxrt_gpu': ["CUDAExecutionProvider", "CPUExecutionProvider"]
}

class brevitas_dynamo_ptq_mode:
    
    def __init__(self, dynamo_module) -> None:
        super().__init__()
        self.dynamo_module = dynamo_module
    
    def __enter__(self):
        global _DYNAMO_PTQ_MODE
        _DYNAMO_PTQ_MODE = True
        
    def __exit__(self, type, value, traceback):
        global _DYNAMO_PTQ_MODE
        _DYNAMO_PTQ_MODE = False


class ONNXRTBackend(object):
    
    def __init__(self, device_type) -> None:
        super().__init__()
        self.input_names = None
        self.output_names = None
        self.device_type = device_type
        
    def __call__(self, gm, example_inputs, *, filename, providers=None) -> Any:
        import onnxruntime  

        if not os.path.exists(filename):
            example_outputs = gm(*example_inputs)
            assert len(example_inputs) > 0, "Requires example inputs"
            self.output_spec = [
                (o.shape, o.dtype, o.layout, o.device, o.requires_grad) for o in example_outputs]
            self.input_names = [f"i{i}" for i in range(len(example_inputs))]
            self.output_names = [f"o{x}" for x in range(len(example_outputs))]
            export_onnx_qcdq(
                gm,
                *example_inputs,
                export_path=filename,
                input_names=self.input_names,
                output_names=self.output_names,
                opset_version=13)
            del example_inputs, example_outputs

        if providers is None:
            providers = [default_provider(self.device_type)]
        for provider in providers:
            assert provider in onnxruntime.get_available_providers()
        session = onnxruntime.InferenceSession(filename, providers=providers)
        
        def _call(*initial_args):
            binding = session.io_binding()
            active_inputs = {inp.name for inp in session.get_inputs()}
            args = [a.contiguous() for a in initial_args]
            for name, value in zip(self.input_names, args):
                if name not in active_inputs:
                    warnings.warn(
                        "input %s skipped as not found in onnx inference session", name)
                    continue
                dev = value.device
                binding.bind_input(
                    name,
                    dev.type,
                    dev.index or 0,
                    _np_dtype[value.dtype],
                    value.size(),
                    value.data_ptr(),
                )
            outputs = [
                torch.empty(
                    shape,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    requires_grad=requires_grad,
                )
                for shape, dtype, layout, device, requires_grad in self.output_spec
            ]

            for name, value in zip(self.output_names, outputs):
                dev = value.device
                binding.bind_output(
                    name,
                    dev.type,
                    dev.index or 0,
                    _np_dtype[value.dtype],
                    value.size(),
                    value.data_ptr(),
                )
            session.run_with_iobinding(binding)
            if self.device_type == "cpu":
                binding.copy_outputs_to_cpu()
            return outputs

        return _call


class QuantizeAndCompile(object):
    
    def __init__(
            self, 
            graph_model, 
            ptq_iters, 
            ptq_methods,
            example_inputs, 
            quantization_backend, 
            compiler_backend,
            equalization_iters,
            weight_bit_width,
            act_bit_width,
            bias_bit_width,
            scaling_per_output_channel,
            act_quant_percentile,
            act_quant_type,
            scale_factor_type,
            weight_narrow_range) -> None:
        assert ptq_iters > len(ptq_methods), "At least 1 batch per PTQ method is required."
        self.ptq_schedule = OrderedDict({ptq_method: ptq_iters // len(ptq_methods) for ptq_method in ptq_methods})
        self.ptq_schedule[next(reversed(self.ptq_schedule))] += (ptq_iters % len(ptq_methods))
        device = next(graph_model.parameters()).device
        if quantization_backend == 'generic' or quantization_backend == 'layerwise':
            graph_model = preprocess_for_quantize(
                graph_model, equalize_iters=equalization_iters, trace_model=False)
        elif quantization_backend == 'flexml':
            graph_model = preprocess_for_flexml_quantize(
                graph_model, *example_inputs, equalize_iters=equalization_iters, trace_model=False)
        else:
            raise ValueError(f"Quantization backend {quantization_backend} not supported.")
        graph_model = quantize_model(
            graph_model, 
            quantization_backend,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            bias_bit_width=bias_bit_width,
            scaling_per_output_channel=scaling_per_output_channel,
            act_quant_percentile=act_quant_percentile,
            act_quant_type=act_quant_type,
            scale_factor_type=scale_factor_type,
            weight_narrow_range=weight_narrow_range)
        graph_model.to(device)  
        if compiler_backend == 'onnxrt_cpu':
            self.compiler_backend_impl = ONNXRTBackend(device_type='cpu')
        elif compiler_backend == 'onnxrt_gpu':
            self.compiler_backend_impl = ONNXRTBackend(device_type='cuda')
        else:
            raise ValueError(f"Compiler backend {self.compiler_backend} not supported.")
        self.compiler_backend = compiler_backend
        self.graph_model = graph_model
        self.ptq_iters = ptq_iters
        self.example_inputs = example_inputs
        self.tmp_dir = tempfile.TemporaryDirectory() 
        self.current_iter = 0
        
    def act_calibration(self, *args, **kwargs):
        self.ptq_schedule['act_calibration'] -= 1
        with calibration_mode(self.graph_model, finalize_stats_on_exit=False):
            return self.graph_model(*args, **kwargs)
        
    def gptq(self, *args, **kwargs):
        self.ptq_schedule['gptq'] -= 1
        with gptq_mode(self.graph_model, act_order=False) as gptq:
            gptq_model = gptq.model
            for i in range(gptq.num_layers):
                gptq_model(*args, **kwargs)
                gptq.update()
        return self.graph_model(*args, **kwargs)
            
    def bias_correction(self, *args, **kwargs):
        self.ptq_schedule['bias_correction'] -= 1
        with bias_correction_mode(self.graph_model):
            return self.graph_model(*args, **kwargs) 
        
    def ptq(self, *args, **kwargs):
        self.current_iter += 1
        for ptq_method, iters in self.ptq_schedule.items():
            if iters == 0:
                if ptq_method == 'act_calibration':
                    finalize_collect_stats(self.graph_model)
                continue
            if ptq_method == 'act_calibration':
                return self.act_calibration(*args, **kwargs)
            elif ptq_method == 'bias_correction':
                return self.bias_correction(*args, **kwargs)
            elif ptq_method == 'gptq':
                return self.gptq(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported ptq method {ptq_method}.")
    
    def __call__(self, *args, **kwargs):
        global _DYNAMO_PTQ_MODE
        if _DYNAMO_PTQ_MODE:
            return self.graph_model(*args, **kwargs)
        if self.current_iter < self.ptq_iters:
            return self.ptq(*args, **kwargs)
        if 'onnxrt' in self.compiler_backend:
            return self.compiler_backend_impl(
                self.graph_model, 
                self.example_inputs, 
                providers=ONNXRT_PROVIDERS[self.compiler_backend],
                filename=os.path.join(self.tmp_dir.name, 'model.onnx'))(*args, **kwargs)
        else:
            raise ValueError(f"Compiler backend {self.compiler_backend} not supported.")


def brevitas_dynamo(
        ptq_iters, 
        ptq_methods,
        equalization_iters=20,
        weight_bit_width=8,
        act_bit_width=8,
        bias_bit_width='int32',
        scaling_per_output_channel=True,
        act_quant_percentile=99.999,
        act_quant_type='symmetric',
        scale_factor_type='float32',
        weight_narrow_range=False,
        quantization_backend='generic', 
        compiler_backend='onnxrt_cpu'):
    def compile(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        qac = QuantizeAndCompile(
            gm, 
            ptq_iters=ptq_iters, 
            ptq_methods=ptq_methods,
            example_inputs=example_inputs,
            equalization_iters=equalization_iters,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            bias_bit_width=bias_bit_width,
            scaling_per_output_channel=scaling_per_output_channel,
            act_quant_percentile=act_quant_percentile,
            act_quant_type=act_quant_type,
            scale_factor_type=scale_factor_type,
            weight_narrow_range=weight_narrow_range,
            quantization_backend=quantization_backend, 
            compiler_backend=compiler_backend)
        return qac
    return compile