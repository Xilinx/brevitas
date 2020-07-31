import copy

from torch.nn import Module

from ...base import onnx


_target_map = {
    'input': {'tensor_type': 'input', 'tensor_index': 0, 'is_init': False},  # inp is input 0
    'weight': {'tensor_type': 'input', 'tensor_index': 1, 'is_init': True}, # weight is input 1
    'bias': {'tensor_type': 'input', 'tensor_index': 2, 'is_init': True},  # bias is input 2
    'output': {'tensor_type': 'output', 'tensor_index': 0, 'is_init': False}}  # out is output 0


def _annotate_tensor(model, node, attr, name, key, tensor_type, tensor_index, is_init):
    if attr.name == name:
        w_tensor_name = getattr(node, tensor_type)[tensor_index]
        if is_init:
            assert w_tensor_name in [x.name for x in model.graph.initializer]
        # only str is supported as attribute of a tensor, so we cast the scale
        tq = onnx.StringStringEntryProto(key=key, value=str(attr.i))
        ta = onnx.TensorAnnotation(tensor_name=w_tensor_name, quant_parameter_tensor_names=[tq])
        model.graph.quantization_annotation.append(ta)


def move_quant_attributes_into_annotations(model: Module):
    """Move quantization info in attributes into quantization_annotation"""
    if onnx is None:
        raise ModuleNotFoundError("Installation of ONNX is required.")
    model = copy.deepcopy(model)
    for n in model.graph.node:
        for a in n.attribute:
            def annotate_tensor(target, name, key):
                _annotate_tensor(model, n, a, name, key, **target)
            for target in _target_map.keys():
                for key in ['scale', 'bit_width']:
                    annotate_tensor(_target_map[target], target + '_' + key, key)
    return model


