# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


class RecipeMixin:
    """Fail loud if the loaded model is not the one this recipe file is for."""

    expected_model_name: str = ""

    @classmethod
    def post_process_quant_model(cls, model):
        name = getattr(model.config, "name_or_path", "") or ""
        if cls.expected_model_name not in name:
            raise ValueError(
                f"Recipe plugin expects model name containing {cls.expected_model_name!r}, "
                f"got {name!r}. Check --model matches this recipe file.")
        return model
