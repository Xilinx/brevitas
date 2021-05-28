import torch


def module_class_name(m: torch.nn.Module):
    module = m.__class__.__module__
    if module is None or module == str.__class__.__module__:
        full_name = m.__class__.__name__
    else:
        full_name = module + '.' + m.__class__.__name__
    return full_name
