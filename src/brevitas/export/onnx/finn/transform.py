import copy

from torch.nn import Module

from ..manager import onnx


def move_quant_attributes_into_annotations(model):
    """Move quantization info in attributes into quantization_annotation"""
    if onnx is None:
        raise ModuleNotFoundError("Installation of ONNX is required.")
    model = copy.deepcopy(model)
    qaname = "finn_datatype"

    def add_to_annotations(attribute, tensor_name):
        value = attribute.s
        if value != 'FLOAT32':
            tq = onnx.StringStringEntryProto(key=qaname, value=attribute.s)
            ta = onnx.TensorAnnotation(
                tensor_name=tensor_name,
                quant_parameter_tensor_names=[tq])
            model.graph.quantization_annotation.append(ta)

    for n in model.graph.node:
        for a in n.attribute:
            mark_for_removal = False
            if a.name == "weight_qnt":
                # assume second input is weight, make sure it has an initializer
                w_tensor_name = n.input[1]
                assert w_tensor_name in [x.name for x in model.graph.initializer]
                add_to_annotations(a, w_tensor_name)
                mark_for_removal = True
            if a.name == "bias_qnt":
                # assume second input is bias, make sure it has an initializer
                b_tensor_name = n.input[1]
                assert b_tensor_name in [x.name for x in model.graph.initializer]
                add_to_annotations(a, b_tensor_name)
                mark_for_removal = True
            elif a.name == "activation_qnt":
                a_tensor_name = n.output[0]
                add_to_annotations(a, a_tensor_name)
                mark_for_removal = True
            if mark_for_removal:
                n.attribute.remove(a)
    return model


def restore_domain(model):
    if onnx is None:
        raise ModuleNotFoundError("Installation of ONNX is required.")
    model = copy.deepcopy(model)
    for n in model.graph.node:
        if n.op_type in ['MatMul', 'Conv', 'Add', 'Div']:
            n.domain = ''
    return model

