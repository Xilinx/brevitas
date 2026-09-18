# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Lists valid GGUF export targets; kept separate so that
``--export-target`` can be validated in :mod:`brevitas_examples.llm.llm_args`
without eagerly importing the full GGUF export stack.
"""

import gguf

FTYPE_MAP: dict[str, gguf.LlamaFileType] = {
    "f32": gguf.LlamaFileType.ALL_F32,
    "f16": gguf.LlamaFileType.MOSTLY_F16,
    "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
    "q5_k": gguf.LlamaFileType.MOSTLY_Q5_K_S,
    "q5_k_s": gguf.LlamaFileType.MOSTLY_Q5_K_S,
    "q5_k_m": gguf.LlamaFileType.MOSTLY_Q5_K_M,
    "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    "q4_1": gguf.LlamaFileType.MOSTLY_Q4_1,
    "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_S,
    "q4_k_s": gguf.LlamaFileType.MOSTLY_Q4_K_S,
    "q4_k_m": gguf.LlamaFileType.MOSTLY_Q4_K_M,
    "q3_k": gguf.LlamaFileType.MOSTLY_Q3_K_S,
    "q3_k_s": gguf.LlamaFileType.MOSTLY_Q3_K_S,
    "q3_k_m": gguf.LlamaFileType.MOSTLY_Q3_K_M,
    "q3_k_l": gguf.LlamaFileType.MOSTLY_Q3_K_L,
    "q2_k": gguf.LlamaFileType.MOSTLY_Q2_K,
    "q2_k_s": gguf.LlamaFileType.MOSTLY_Q2_K_S,
    "auto": gguf.LlamaFileType.GUESSED,}

# Valid `--export-target` values for GGUF export, e.g. "gguf:q4_k_s".
GGUF_EXPORT_TARGETS = [f"gguf:{ftype}" for ftype in FTYPE_MAP]
