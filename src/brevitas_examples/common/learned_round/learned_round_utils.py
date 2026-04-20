# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import TypeVar

from accelerate.utils.operations import send_to_device
import torch
from torch import nn
from torch.utils.data import Dataset

from brevitas.utils.torch_utils import StopFwdException

T_inputs = TypeVar("_T_inputs")
T_outputs = TypeVar("_T_output")
T_model_inputs = TypeVar("_T_model_inputs")
T_cache = Tuple[T_inputs, T_outputs]


# Cache, as a subclass of torch.utils.data.Dataset, needs to implement __getitem__ and __len__
class Cache(Generic[T_inputs, T_outputs], Dataset[T_cache]):

    inputs: Sequence[T_inputs]
    outputs: Sequence[T_outputs]

    @abstractmethod
    def store_inputs(self, args: Tuple[torch.Tensor, ...], kwargs: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def store_output(self, output: Any) -> None:
        pass

    @abstractmethod
    def reset_cache(self) -> None:
        pass

    @abstractmethod
    def collate_fn(self, batch: Iterable[T_cache]) -> T_cache:
        pass

    def collate_fn_output_next(self, batch: Iterable[T_cache]) -> T_cache:
        raise NotImplementedError(
            f"{self.__class__.__name__} is not compatible with fast_update=True.")

    def collate_fn_input_next(self, batch: Iterable[T_cache]) -> T_cache:
        raise NotImplementedError(
            f"{self.__class__.__name__} is not compatible with fast_update=True.")


class ModelForwardFn(Protocol):

    def __call__(self, model: nn.Module, inputs: T_model_inputs) -> Any:
        ...


class BlockForwardFn(Protocol):

    def __call__(self, block: nn.Module, inputs: T_inputs) -> T_outputs:
        ...


class DataSaverHook:

    def __init__(
        self,
        cache: Cache,
        store_inputs: bool = True,
        store_output: bool = True,
        keep_gpu: bool = True,
    ) -> None:
        self.cache = cache
        self.store_inputs = store_inputs
        self.store_output = store_output
        self.keep_gpu = keep_gpu

    def __call__(self, module: nn.Module, args, kwargs, output) -> None:
        if self.store_inputs:
            if not self.keep_gpu:
                args = send_to_device(args, 'cpu')
                kwargs = send_to_device(kwargs, 'cpu')
            self.cache.store_inputs(args, kwargs)
        if self.store_output:
            if not self.keep_gpu:
                output = send_to_device(output, 'cpu')
            self.cache.store_output(output)

        raise StopFwdException
