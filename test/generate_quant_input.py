import torch


def generate_quant_input(shape, bit, scale_factor = 0.2, narrow_band = True, signed = True):
    n_elements = 2**bit
    min_value = 0
    if narrow_band and signed:
        min_value = 1
    quant_input = torch.randint(min_value, 2**bit, shape)
    if signed:
        quant_input = quant_input - n_elements/2

    return quant_input.float(), quant_input.float() * scale_factor
