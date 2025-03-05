# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
import functools
import logging
import os
from typing import Callable, List

import torch

from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks


# If the environment variable 'LOCAL_RANK' is not set, a single
# process is running, so os.environ.get('LOCAL_RANK', -1) returns
# -1.
def is_multi_process():
    return int(os.environ.get('LOCAL_RANK', -1)) != -1


def is_main_process():
    return int(os.environ.get('LOCAL_RANK', -1)) in [-1, 0]


def on_process(func: Callable, process_index: int):

    @functools.wraps(func)
    def _wrapper(model, *args, **kwargs):
        curr_process_index = int(os.environ.get('LOCAL_RANK', -1))
        # TODO: Change to logging.debug
        if curr_process_index == -1:
            logging.debug(f"Applying {func.__name__} on main process")
            return func(model, *args, **kwargs)
        elif process_index == curr_process_index:
            logging.debug(f"Applying {func.__name__} on process index {curr_process_index}")
            return func(model, *args, **kwargs)
        else:
            logging.debug(
                f"Skipping function {func.__name__} on process index {curr_process_index}")
            return model

    return _wrapper


on_main_process = functools.partial(on_process, process_index=0)


def validate_distributed_args(args: Namespace) -> None:
    assert args.optimize_rotations, "The entry-point should be run as a single-process if rotations are not being optimized."


class dist_offload_model:

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def __enter__(self):
        if is_main_process():
            self.model = offload_model(self.model)

    def __exit__(self, type, value, traceback):
        if is_main_process():
            remove_hooks(self.model)
