import torch

RTOL = 0
ATOL = 1e-23

FP_BIT_WIDTH = 32
MIN_INT_BIT_WIDTH = 2
MAX_INT_BIT_WIDTH = 8
INT_BIT_WIDTH_TO_TEST = range(MIN_INT_BIT_WIDTH, MAX_INT_BIT_WIDTH + 1)
BOOLS = [True, False]


def assert_allclose(generated, reference):
    assert torch.allclose(generated, reference, RTOL, ATOL)

