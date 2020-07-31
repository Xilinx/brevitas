import copy

from torch.nn import Module

from .base import onnx


def move_domain_attributes_into_domain(model: Module):
    """Move domain info in attributes into the actual node.domain field"""
    if onnx is None:
        raise ModuleNotFoundError("Installation of ONNX is required.")

    model = copy.deepcopy(model)
    for n in model.graph.node:
        for a in n.attribute:
            mark_for_removal = False
            if a.name == "domain":
                n.domain = a.s
                mark_for_removal = True
            if mark_for_removal:
                n.attribute.remove(a)
    return model