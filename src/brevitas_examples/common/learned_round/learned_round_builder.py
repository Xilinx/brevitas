# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict

from brevitas_examples.common.learned_round.learned_round_method import AdaRound
from brevitas_examples.common.learned_round.learned_round_method import AutoRound
from brevitas_examples.common.learned_round.learned_round_method import LearnedRound
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundModelUtils
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundOptimizer
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import \
    LearnedRoundVisionUtils
from brevitas_examples.llm.llm_quant.learned_round_utils import LearnedRoundLLMUtils


def solve_learned_round_utils_cls(utils_type) -> LearnedRoundModelUtils:
    if utils_type == "imagenet_classification":
        return LearnedRoundVisionUtils
    elif utils_type == "llm":
        return LearnedRoundLLMUtils
    else:
        raise Exception(f"Learned round utilities for {utils_type} are not recognized.")


def solve_learned_round_method_cls(method_type) -> LearnedRound:
    if method_type == "ada_round":
        return AdaRound
    elif method_type == "auto_round":
        return AutoRound
    else:
        raise Exception(f"Learned round method {method_type} is not available.")


def instantiate_learned_round_optimizer(
        utils_type: str,
        method_type: str = "auto_round",
        iters: int = 200,
        method_params: Dict = {},
        optimizer_params: Dict = {},
        utils_params: Dict = {}) -> LearnedRoundOptimizer:
    # Instantiate learned round utilities
    learned_round_utils_cls = solve_learned_round_utils_cls(utils_type)
    learned_round_utils = learned_round_utils_cls(**utils_params)

    # Instantiate learned round method
    learned_round_method_cls = solve_learned_round_method_cls(method_type)
    learned_round_method = learned_round_method_cls(iters, **method_params)

    # Make sure that the iterations of the learned round method and optimizer match
    optimizer_params["iters"] = iters
    # Instantiate optimizer
    return LearnedRoundOptimizer(learned_round_method, learned_round_utils, **optimizer_params)
