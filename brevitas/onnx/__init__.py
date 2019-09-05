import torch
import torch.onnx
import onnx
import onnx.optimizer as opt
import copy

def _prepare_for_finn_onnx_export(module, enable_export = True):
    """Traverse children of given module to prepare them for FINN-ONNX export.

    This sets the export_mode property of all child modules to enable_export
    where applicable (e.g. subclasses of QuantLayer), which changes their
    ONNX export behavior. No extra preparation is needed for non-Brevitas
    (or non-quantized) child modules, for which this function does nothing.
    """

    try:
        module.export_mode = enable_export
    except AttributeError:
        # module does not have the prepare_for_export function, skip
        pass
    for c in module.children():
        _prepare_for_finn_onnx_export(c, enable_export)

def _move_quant_attributes_into_annotations(model):
    """Move quantization info in attributes into quantization_annotation"""

    model = copy.deepcopy(model)
    for n in model.graph.node:
        for a in n.attribute:
            mark_for_removal = False
            if a.name == "weight_qnt":
                # assume first input is weight, make sure it has an initializer
                w_tensor_name = n.input[0]
                assert w_tensor_name in [x.name for x in model.graph.initializer]
                tq = onnx.StringStringEntryProto(key = a.name, value = a.s)
                ta = onnx.TensorAnnotation(
                    tensor_name = w_tensor_name,
                    quant_parameter_tensor_names = [tq]
                )
                model.graph.quantization_annotation.append(ta)
                mark_for_removal = True
            elif a.name == "activation_qnt":
                a_tensor_name = n.output[0]
                tq = onnx.StringStringEntryProto(key = a.name, value = a.s)
                ta = onnx.TensorAnnotation(
                    tensor_name = a_tensor_name,
                    quant_parameter_tensor_names = [tq]
                )
                model.graph.quantization_annotation.append(ta)
                mark_for_removal = True
            if mark_for_removal:
                n.attribute.remove(a)
    return model

def export_finn_onnx(module, input_shape, export_path):
    """Export given module with Brevitas layers to FINN-ONNX with some cleanup."""

    with torch.no_grad():
        # TODO maybe consider a deepcopy of the module first?
        module = module.eval()
        _prepare_for_finn_onnx_export(module, enable_export = True)
        torch.onnx.export(
            module, torch.empty(input_shape, dtype=torch.float), export_path
        )
        # restore the model to non-export mode to keep it clean
        _prepare_for_finn_onnx_export(module, enable_export = False)
        # do some cleanup on the exported ONNX model
        model = onnx.load(export_path)
        onnx_passes = [
            # use initializers instead of Constant nodes for fixed params
            "extract_constant_to_initializer",
            # remove unused graph inputs (e.g. zero_hw_sentinel) & initializers
            "eliminate_unused_initializer"
        ]
        model = opt.optimize(model, onnx_passes)
        model = _move_quant_attributes_into_annotations(model)
        onnx.save(model, export_path)
