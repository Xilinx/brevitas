# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gguf

GGUF_QUANTIZER_FILE_TYPES: dict[str, gguf.LlamaFileType] = {
    "gguf_q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "gguf_q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
    "gguf_q5_k": gguf.LlamaFileType.MOSTLY_Q5_K_S,
    "gguf_q5_k_s": gguf.LlamaFileType.MOSTLY_Q5_K_S,
    "gguf_q5_k_m": gguf.LlamaFileType.MOSTLY_Q5_K_M,
    "gguf_q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    "gguf_q4_1": gguf.LlamaFileType.MOSTLY_Q4_1,
    "gguf_q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_S,
    "gguf_q4_k_s": gguf.LlamaFileType.MOSTLY_Q4_K_S,
    "gguf_q4_k_m": gguf.LlamaFileType.MOSTLY_Q4_K_M,
    "gguf_q3_k": gguf.LlamaFileType.MOSTLY_Q3_K_S,
    "gguf_q3_k_s": gguf.LlamaFileType.MOSTLY_Q3_K_S,
    "gguf_q3_k_m": gguf.LlamaFileType.MOSTLY_Q3_K_M,
    "gguf_q3_k_l": gguf.LlamaFileType.MOSTLY_Q3_K_L,
    "gguf_q2_k": gguf.LlamaFileType.MOSTLY_Q2_K,
    "gguf_q2_k_s": gguf.LlamaFileType.MOSTLY_Q2_K_S,}


def file_type_from_quantizer(quantizer_name: str | None) -> gguf.LlamaFileType:
    if quantizer_name is None:
        raise ValueError("GGUF export requires a GGUF custom quantizer.")
    try:
        return GGUF_QUANTIZER_FILE_TYPES[quantizer_name]
    except KeyError:
        supported = ", ".join(sorted(GGUF_QUANTIZER_FILE_TYPES))
        raise ValueError(
            f"GGUF export does not support custom quantizer {quantizer_name!r}. "
            f"Select one of: {supported}.") from None
