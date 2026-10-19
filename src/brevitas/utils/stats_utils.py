# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from contextlib import nullcontext
from contextvars import ContextVar
from typing import Any
from typing import ContextManager
from typing import Dict
from typing import List
from typing import Protocol


class StatFn(Protocol):

    def __call__(self, **payload) -> Dict[str, Any]:
        ...


class BaseStatsCollector(ABC):
    """Event-driven collector of statistics produced during PTQ.

    Producers call :meth:`log` with an event name and an opaque payload;
    whether anything is computed at all is decided by whoever registered a
    handler for that event via :meth:`on`. This keeps the instrumentation
    in the algorithms free of any knowledge of which metrics are collected.
    """

    def __init__(self) -> None:
        self._stats_fn: Dict[str, StatFn] = {}

    def on(self, key: str, fn: StatFn) -> None:
        self._stats_fn[key] = fn

    def log(self, event: str, **payload) -> None:
        if (fn := self._stats_fn.get(event)):
            self._log(fn, **payload)

    @abstractmethod
    def save_to_yaml(self, path: str) -> None:
        pass

    @abstractmethod
    def _log(self, fn: StatFn, **payload) -> None:
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(self._stats_fn.keys())})'

    @property
    def is_active(self) -> bool:
        return True


class NullCollector(BaseStatsCollector):
    """Collector that discards everything.

    This is the default context value, so the instrumentation in the PTQ
    algorithms is a no-op unless a real collector is installed with
    :func:`collect_stats`.
    """

    def on(self, key: str, fn: StatFn) -> None:
        pass

    def _log(self, fn: StatFn, **payload) -> None:
        pass

    def save_to_yaml(self, path: str) -> None:
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

    def save_to_yaml(self, path: str) -> None:
        # Imported lazily: pyyaml is not a runtime dependency of brevitas.
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.stats, f, sort_keys=False)


StatsCollectorCtx: ContextVar[BaseStatsCollector] = ContextVar(
    "StatsCollector", default=NullCollector())

StatsPrefixCtx: ContextVar[str] = ContextVar("StatsPrefix", default="")


@contextmanager
def collect_stats(collector: BaseStatsCollector):
    token = StatsCollectorCtx.set(collector)
    try:
        yield collector
    finally:
        StatsCollectorCtx.reset(token)


@contextmanager
def stats_prefix(prefix: str):
    """Namespace the names reported by the statistics collected inside.

    Used when the same layer names recur across transformer blocks, so that
    each block contributes unique keys (e.g. ``model.layers.0.self_attn.q_proj``).
    """
    token = StatsPrefixCtx.set(prefix)
    try:
        yield prefix
    finally:
        StatsPrefixCtx.reset(token)


def prefixed_stats_name(name: str) -> str:
    prefix = StatsPrefixCtx.get()
    return f"{prefix}.{name}" if prefix else name


def is_stats_collector_active():
    return StatsCollectorCtx.get().is_active


def collect_stats_ctx(key: str, stats: List[str]) -> ContextManager:
    if key in stats:
        return collect_stats(DictStatsCollector())
    return nullcontext(NullCollector())
