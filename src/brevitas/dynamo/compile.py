# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import os
import tempfile
from typing import Any, List
import warnings 

import torch
from torch._dynamo.backends.common import device_from_inputs
from torch._dynamo.backends.onnxrt import default_provider
from torch._dynamo.backends.onnxrt import _np_dtype

from brevitas.export import export_onnx_qcdq
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize, quantize_flexml
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode


_DYNAMO_PTQ_MODE = False


class brevitas_dynamo_ptq_mode:
    
    def __init__(self) -> None:
        super().__init__()
    
    def __enter__(self):
        print("enter")
        global _DYNAMO_PTQ_MODE
        _DYNAMO_PTQ_MODE = True
        
    def __exit__(self, type, value, traceback):
        global _DYNAMO_PTQ_MODE
        _DYNAMO_PTQ_MODE = False


class ONNXRTBackend(object):
    
    def __init__(self) -> None:
        super().__init__()
        self.input_names = None
        self.output_names = None
    
    def __call__(self, gm, example_inputs, *, filename, provider=None) -> Any:
        import onnxruntime  

        device_type = device_from_inputs(example_inputs).type

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

        if provider is None:
            provider = default_provider(device_type)
        assert provider in onnxruntime.get_available_providers()
        session = onnxruntime.InferenceSession(filename, providers=[provider])
        
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
            if device_type == "cpu":
                binding.copy_outputs_to_cpu()
            return outputs

        return _call


class QuantizeAndCompile(object):
    
    def __init__(self, graph_model, ptq_iters, example_inputs, quantization_backend, compiler_backend) -> None:
        device = next(graph_model.parameters()).device
        if quantization_backend == 'generic':
            graph_model = preprocess_for_quantize(graph_model, trace_model=False)
            graph_model = quantize(graph_model)
        elif quantization_backend == 'flexml':
            graph_model = preprocess_for_flexml_quantize(graph_model, *example_inputs, trace_model=False)
            graph_model = quantize_flexml(graph_model)  
        else:
            raise ValueError(f"Quantization backend {quantization_backend} not supported.")
        graph_model.to(device)  
        if compiler_backend == 'onnxrt':
            self.compiler_backend_impl = ONNXRTBackend()
        else:
            raise ValueError(f"Compiler backend {self.compiler_backend} not supported.")
        self.compiler_backend = compiler_backend
        self.graph_model = graph_model
        self.ptq_iters = ptq_iters
        self.example_inputs = example_inputs
        self.tmp_dir = tempfile.TemporaryDirectory() 
        self.current_iter = 0
        
    def ptq(self, *args, **kwargs):
        self.current_iter += 1
        with calibration_mode(self.graph_model, finalize_stats_on_exit=False):
            self.graph_model(*args, **kwargs)
        with bias_correction_mode(self.graph_model):
            return self.graph_model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        global _DYNAMO_PTQ_MODE
        if _DYNAMO_PTQ_MODE:
            return self.graph_model(*args, **kwargs)
        if self.current_iter < self.ptq_iters:
            return self.ptq(*args, **kwargs)
        if self.compiler_backend == 'onnxrt':
            return self.compiler_backend_impl(
                self.graph_model, 
                self.example_inputs, 
                filename=os.path.join(self.tmp_dir.name, 'model.onnx'))(*args, **kwargs)
        else:
            raise ValueError(f"Compiler backend {self.compiler_backend} not supported.")


def brevitas_dynamo(ptq_iters=0, quantization_backend='generic', compiler_backend='onnxrt'):
    def ort(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        return QuantizeAndCompile(
            gm, 
            ptq_iters=ptq_iters, 
            example_inputs=example_inputs,
            quantization_backend=quantization_backend, 
            compiler_backend=compiler_backend)
    return ort