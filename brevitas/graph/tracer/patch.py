from unittest import mock

import torch

original_torch_cat = torch.cat


def above_16_cat(tensors, dim):
    if isinstance(tensors, (tuple, list)):
        return original_torch_cat(tensors, dim)
    else:
        return original_torch_cat((tensors[0], tensors[1]), dim)

above_16_cat_patch = mock.patch('torch.cat', wraps=above_16_cat)

ABOVE_16_PATCHES = [above_16_cat_patch]