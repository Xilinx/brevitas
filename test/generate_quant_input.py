import torch


def generate_quant_input(shape, bit, scale_factor = 0.2, narrow_band = True, signed = True):
    n_elements = 2**bit
    min = 0
    if narrow_band and signed:
        min = 1
    quant_input = torch.randint(min, 2**bit, shape)
    if signed:
        quant_input = quant_input - n_elements/2

    return quant_input.float(), quant_input.float() * scale_factor


if __name__ == '__main__':
    A, B = generate_quant_input((10,20), 4, 0.2, True, True)
    print(A)
    print(B)