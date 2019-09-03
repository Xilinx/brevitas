quantization_annotation = dict()

def prepare_for_onnx_export(pytorch_module, enable_export = True):
    """Traverse children of given module to prepare them for ONNX export.

    This sets the export_mode property of all child modules to enable_export
    where applicable (e.g. subclasses of QuantLayer), which changes their
    ONNX export behavior. No extra preparation is needed for non-Brevitas
    (or non-quantized) child modules, for which this function does nothing.
    """

    try:
        pytorch_module.export_mode = enable_export
    except AttributeError:
        # module does not have the prepare_for_export function, skip
        pass
    for c in pytorch_module.children():
        prepare_for_onnx_export(c, enable_export)
