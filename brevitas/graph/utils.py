import torch


def module_class_name(m: torch.nn.Module):
    module = m.__class__.__module__
    if module is None or module == str.__class__.__module__:
        full_name = m.__class__.__name__
    else:
        full_name = module + '.' + m.__class__.__name__
    return full_name


def _inner_flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in _inner_flatten(i):
                yield j
        else:
            yield i


def flatten(container):
    if isinstance(container, list):
        return list(_inner_flatten(container))
    elif isinstance(container, tuple):
        return tuple(_inner_flatten(container))
    else:
        return container