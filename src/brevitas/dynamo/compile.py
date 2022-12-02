import os
from typing import List

import torch
from torch._dynamo.optimizations.subgraph import SubGraph
from torch._dynamo.optimizations.backends import onnxrt_cpu

from brevitas.quant_tensor import QuantTensor
from brevitas.graph.target.flexml import preprocess_flexml, quantize_flexml
from brevitas.onnx import export_standard_qcdq_onnx


class SubSubGraph(SubGraph):

    @property
    def onnx_filename(self):
        name = "manual_export.onnx"
        filename = self.filename(name)
        if os.path.exists(filename):
            return filename
        try:
            export_standard_qcdq_onnx(
                self.model,
                *self.example_inputs,
                filename,
                input_names=self.input_names,
                output_names=self.output_names,
                do_constant_folding=True,
                opset_version=13)
        except IndexError:
            export_standard_qcdq_onnx(
                self.model,
                *self.example_inputs,
                filename,
                input_names=self.input_names,
                output_names=self.output_names,
                do_constant_folding=False,
                opset_version=13)
        return filename
    

class Wrapper(torch.nn.Module):
    
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        out = out[0]
        return out


class QuantizeAndOptimize(object):
    
    def __init__(self, graph_model, calibration_iterations, example_inputs) -> None:
        self.graph_model = preprocess_flexml(graph_model, trace_model=False, *example_inputs)
        self.graph_model = quantize_flexml(self.graph_model)
        self.graph_model = graph_model
        self.calibration_iterations = calibration_iterations
        self.example_inputs = example_inputs
        self.current_iter = 0
    
    def __call__(self, *args, **kwargs):
        if self.current_iter < self.calibration_iterations:
            self.current_iter += 1
            return self.graph_model(*args, **kwargs)
        elif self.current_iter == self.calibration_iterations:
            self.graph_model = Wrapper(self.graph_model)
            self.graph_model = SubSubGraph(self.graph_model, self.example_inputs, 'output_dir')
            self.current_iter += 1
        return onnxrt_cpu(self.graph_model)(*args, **kwargs)


def quantized_ort(iters):
    def ort(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        return(QuantizeAndOptimize(gm, calibration_iterations=iters, example_inputs=example_inputs))
    return ort