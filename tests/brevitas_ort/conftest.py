# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from .sampling import report_lines


def pytest_report_header(config):
    # Surface the sampling seed / counts so any run (or failure) is reproducible by
    # re-exporting BREVITAS_ORT_SAMPLE_SEED and BREVITAS_ORT_NUM_SAMPLES.
    return report_lines()
