def onnx_export_opset():
    try:
        import torch.onnx.symbolic_helper as cfg
        ATR_NAME = '_export_onnx_opset_version'
        opset = getattr(cfg, ATR_NAME)

    except:
        from torch.onnx._globals import GLOBALS as cfg
        ATR_NAME = 'export_onnx_opset_version'
        opset = getattr(cfg, ATR_NAME)
    
    return opset
