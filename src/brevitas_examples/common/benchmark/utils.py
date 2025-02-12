# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from argparse import ArgumentParser
from argparse import Namespace
from typing import Any, Dict, List, Tuple


class BenchmarkUtils(ABC):

    @staticmethod
    @abstractmethod
    def parse_log(job_log: str) -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def validate(args: Namespace, extra_args: List[str]) -> None:
        pass

    @staticmethod
    @abstractmethod
    def entrypoint_main(args: Namespace, extra_args: List[str]) -> Tuple[Dict, Any]:
        pass

    @property
    @abstractmethod
    def argument_parser() -> ArgumentParser:
        pass

    @property
    @abstractmethod
    def eval_metrics() -> List[str]:
        pass
