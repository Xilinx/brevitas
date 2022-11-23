def to_0dim_if_scalar(tensor):
    if len(tensor.shape) == 1 and tensor.shape[0] == 1:
        tensor = tensor.view(()) # 0-Dim tensor
    return tensor