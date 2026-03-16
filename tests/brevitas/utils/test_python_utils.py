# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from enum import auto

import pytest

from brevitas.utils.python_utils import AutoName
from brevitas.utils.python_utils import Registry


class TestEnum(AutoName):
    FIRST = auto()
    SECOND = auto()


def test_eq_upper_str():
    assert TestEnum.FIRST == 'FIRST'
    assert TestEnum.SECOND == 'SECOND'


def test_neq_upper_str():
    assert TestEnum.FIRST != 'SECOND'
    assert TestEnum.SECOND != 'FIRST'


def test_eq_lower_str():
    assert TestEnum.FIRST == 'first'
    assert TestEnum.SECOND == 'second'


def test_neq_lower_str():
    assert TestEnum.FIRST != 'second'
    assert TestEnum.SECOND != 'first'


def test_eq_enum():
    assert TestEnum.FIRST == TestEnum.FIRST
    assert TestEnum.SECOND == TestEnum.SECOND


def test_neq_enum():
    assert TestEnum.FIRST != TestEnum.SECOND


class TestRegistry:

    def test_register_single_name(self):
        r = Registry()

        @r.register("k")
        class Dummy:
            pass

        dummy = r.get("k")

        assert dummy is Dummy
        assert len(r.get_registered_keys()) == 1
        assert next(iter(r.get_registered_keys())) == "k"

    def test_register_multiple_names(self):
        r = Registry()

        @r.register(["k1", "k2"])
        class Dummy:
            pass

        dummy1 = r.get("k1")
        dummy2 = r.get("k2")

        assert dummy1 is Dummy
        assert dummy2 is Dummy
        assert len(r.get_registered_keys()) == 2
        assert set(r.get_registered_keys()) == {"k1", "k2"}

    def test_register_single_name_static(self):
        r = Registry()

        @Registry.register(r, "k")
        class Dummy:
            pass

        dummy = r.get("k")

        assert dummy is Dummy
        assert len(r.get_registered_keys()) == 1
        assert next(iter(r.get_registered_keys())) == "k"

    def test_register_duplicate_raises_warning(self):
        r = Registry("TestRegistry")

        r.register("dup")("k")
        # Patch warnings.warn and check that it is called with the expected message
        with pytest.warns(UserWarning) as record:
            r.register("dup")("k")
        msg = str(record[0].message)
        assert "'dup' is already registered in TestRegistry. Overwriting the existing value." == msg

    def test_get_missing_empty_raises_valueerror(self):
        r = Registry("TestRegistry")

        with pytest.raises(ValueError) as excinfo:
            r.get("missing")

        msg = str(excinfo.value)
        assert msg == "'missing' not found in TestRegistry. The available values are: <empty>"

    def test_get_missing_raises_valueerror(self):
        r = Registry("TestRegistry")

        r.register("k1")("v1")
        r.register("k2")("v2")
        with pytest.raises(ValueError) as excinfo:
            r.get("missing")

        msg = str(excinfo.value)
        assert msg == "'missing' not found in TestRegistry. The available values are: k1, k2"
