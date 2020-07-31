import copy

from torch.nn import Module

from ..base import onnx


def move_quant_attributes_into_annotations(model: Module):
    """Move quantization info in attributes into quantization_annotation"""
    if onnx is None:
        raise ModuleNotFoundError("Installation of ONNX is required.")

    model = copy.deepcopy(model)
    qaname = "finn_datatype"
    for n in model.graph.node:
        for a in n.attribute:
            mark_for_removal = False
            if a.name == "weight_qnt":
                # assume second input is weight, make sure it has an initializer
                w_tensor_name = n.input[1]
                assert w_tensor_name in [x.name for x in model.graph.initializer]
                tq = onnx.StringStringEntryProto(key=qaname, value=a.s)
                ta = onnx.TensorAnnotation(
                    tensor_name=w_tensor_name,
                    quant_parameter_tensor_names=[tq])
                model.graph.quantization_annotation.append(ta)
                mark_for_removal = True
            elif a.name == "activation_qnt":
                a_tensor_name = n.output[0]
                tq = onnx.StringStringEntryProto(key=qaname, value=a.s)
                ta = onnx.TensorAnnotation(
                    tensor_name=a_tensor_name,
                    quant_parameter_tensor_names=[tq])
                model.graph.quantization_annotation.append(ta)
                mark_for_removal = True
            if mark_for_removal:
                n.attribute.remove(a)
    return model


