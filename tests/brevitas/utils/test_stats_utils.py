# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.utils.stats_utils import BaseStatsCollector
from brevitas.utils.stats_utils import collect_stats
from brevitas.utils.stats_utils import collect_stats_ctx
from brevitas.utils.stats_utils import DictStatsCollector
from brevitas.utils.stats_utils import is_stats_collector_active
from brevitas.utils.stats_utils import NullCollector
from brevitas.utils.stats_utils import prefixed_stats_name
from brevitas.utils.stats_utils import stats_prefix
from brevitas.utils.stats_utils import StatsCollectorCtx


class _SpyCollector(BaseStatsCollector):
    """Collector for testing BaseStatsCollector semantics."""

    def __init__(self):
        super().__init__()
        self.calls = []  # List of (fn, payload)

    def _log(self, fn, **payload) -> None:
        self.calls.append((fn, payload))

    def save_to_yaml(self, path: str) -> None:
        pass


def test_collector_dispatches():
    c = _SpyCollector()

    def fn(**payload):
        return {"seen": payload}

    c.on("evt", fn)
    c.log("evt", a=1, b=2)

    assert len(c.calls) == 1
    called_fn, called_payload = c.calls[0]
    assert called_fn is fn
    assert called_payload == {"a": 1, "b": 2}

    # If the key is missing, log() should be a no-op
    c.log("missing", x=1)
    assert len(c.calls) == 1


def test_repr_keys():
    c = _SpyCollector()

    c.on("a", lambda **p: {})
    c.on("b", lambda **p: {})
    r = repr(c)

    assert r.startswith("_SpyCollector(")
    assert "a" in r and "b" in r


def test_nullcollector_is_inactive():
    n = NullCollector()

    called = {"count": 0}

    def fn(**payload):
        called["count"] += 1
        return {"x": 1}

    # NullCollector should not register handlers or call them
    n.on("evt", fn)
    n.log("evt", a=1)

    assert len(n._stats_fn) == 0
    assert n.is_active is False
    assert called["count"] == 0


def test_dict_collector_merges_stats():
    c = DictStatsCollector()

    c.on("e1", lambda **p: {"outer": {"a": 1}, "flat": p["v"]})
    c.on("e2", lambda **p: {"outer": {"b": 2}, "flat": 123})

    c.log("e1", v=5)
    assert c.stats == {"outer": {"a": 1}, "flat": 5}

    # "e2" deep-merges "outer" and overwrites "flat"
    c.log("e2")
    assert c.stats == {"outer": {"a": 1, "b": 2}, "flat": 123}


def test_collect_stats_sets_context():
    default = StatsCollectorCtx.get()
    assert isinstance(default, NullCollector)
    assert default.is_active is False

    c = DictStatsCollector()
    with collect_stats(c):
        assert StatsCollectorCtx.get() is c
        assert StatsCollectorCtx.get().is_active is True

    # Back to default after exit
    after = StatsCollectorCtx.get()
    assert isinstance(after, NullCollector)
    assert after.is_active is False


def test_collect_stats_ctx_only_activates_for_requested_key():
    with collect_stats_ctx("gptq", ["gptq", "qronos"]) as collector:
        assert isinstance(collector, DictStatsCollector)
        assert is_stats_collector_active()

    with collect_stats_ctx("gpfq", ["gptq", "qronos"]) as collector:
        assert isinstance(collector, NullCollector)
        assert not is_stats_collector_active()


def test_stats_prefix_namespaces_names():
    assert prefixed_stats_name("q_proj") == "q_proj"

    with stats_prefix("layers.0"):
        assert prefixed_stats_name("q_proj") == "layers.0.q_proj"
        # Prefixes do not nest: the innermost one wins.
        with stats_prefix("layers.1"):
            assert prefixed_stats_name("q_proj") == "layers.1.q_proj"
        assert prefixed_stats_name("q_proj") == "layers.0.q_proj"

    assert prefixed_stats_name("q_proj") == "q_proj"
