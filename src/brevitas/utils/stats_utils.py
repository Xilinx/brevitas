# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from typing import Dict
from typing import Protocol


class StatFn(Protocol):

    def __call__(self, **payload) -> Dict[str, Any]:
        ...


class BaseStatsCollector(ABC):

    def __init__(self) -> None:
        self._stats_fn: Dict[str, StatFn] = {}

    def on(self, key: str, fn: StatFn) -> None:
        self._stats_fn[key] = fn

    def log(self, event: str, **payload) -> None:
        if (fn := self._stats_fn.get(event)):
            self._log(fn, **payload)

    @abstractmethod
    def _log(self, fn: StatFn, **payload) -> None:
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(self._stats_fn.keys())})'

    @property
    def is_active(self) -> bool:
        return True


class NullCollector(BaseStatsCollector):

    def on(self, key: str, fn: StatFn) -> None:
        pass

    def _log(self, fn: StatFn, **payload) -> None:
        pass

    # NullCollector is always inactive, so stats are not collected
    @property
    def is_active(self) -> bool:
        return False


def recursive_update(d: Dict, u: Dict) -> Dict:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class DictStatsCollector(BaseStatsCollector):

    def __init__(self) -> None:
        super().__init__()
        self.stats: Dict[str, Any] = {}

    def _log(self, fn: StatFn, **payload) -> None:
        recursive_update(self.stats, fn(**payload))


StatsCollectorCtx: ContextVar[BaseStatsCollector] = ContextVar(
    "StatsCollector", default=NullCollector())


@contextmanager
def collect_stats(collector: BaseStatsCollector):
    token = StatsCollectorCtx.set(collector)
    try:
        yield
    finally:
        StatsCollectorCtx.reset(token)
