try:
    import torch.nn.quantized.functional as qF

# Skip for pytorch 1.1.0
except:
    qF = None