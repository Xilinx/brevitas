# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


def test_import_brevitas():
    import brevitas
    from brevitas.function import ops_ste


def test_import_inference_export():
    # The tests_brevitas_install_dev session has no `export` extra, so onnx is not installed there:
    # importing brevitas.export must not require it.
    from brevitas.export.inference import quant_inference_mode
