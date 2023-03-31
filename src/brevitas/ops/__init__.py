# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Match naming of csrc so that e.g. I can import C++ from torch.ops.autograd_ste_ops
# and python from brevitas.ops.autograd_ste_ops

# import brevitas.ops.autograd_ste_ops as autograd_ste_ops
# #is not supported syntax on 3.6, resort to this
from brevitas.ops import autograd_ste_ops as autograd_ste_ops
