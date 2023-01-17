# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


def finn_datatype(bit_width_tensor, signed, supported_int_bit_width_range=(2, 33)):
    bit_width = int(bit_width_tensor.item())
    if bit_width == 1 and signed:
        return "BIPOLAR"
    elif bit_width == 1 and not signed:
        return 'BINARY'
    elif bit_width in range(*supported_int_bit_width_range):
        return f"INT{bit_width}" if signed else f"UINT{bit_width}"
    else:
        raise RuntimeError(f"Unsupported input bit width {bit_width} for export")
