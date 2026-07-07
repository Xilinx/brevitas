# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from enum import auto

import pytest

from brevitas.utils.python_utils import AutoName
from brevitas.utils.python_utils import convert_str_dict
from brevitas.utils.python_utils import FLOAT_RE
from brevitas.utils.python_utils import INT_RE
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
        assert msg == "'missing' not found in TestRegistry. The registered keys are: <empty>"

    def test_get_missing_raises_valueerror(self):
        r = Registry("TestRegistry")

        r.register("k1")("v1")
        r.register("k2")("v2")
        with pytest.raises(ValueError) as excinfo:
            r.get("missing")

        msg = str(excinfo.value)
        assert msg == "'missing' not found in TestRegistry. The registered keys are: k1, k2"


class TestConvertDictStringVals:

    @pytest.mark.parametrize(
        "value, expected_type, expected_value",
        [
            # bools
            ("true", bool, True),
            ("false", bool, False),
            ("TRUE", bool, True),
            ("False", bool, False),
            ("tRuE", bool, True),
            ("fAlSe", bool, False),
            # integers
            ("0", int, 0),
            ("123", int, 123),
            ("001", int, 1),
            ("0000", int, 0),
            ("+1", int, 1),
            ("-1", int, -1),
            ("-0", int, 0),
            # floats
            ("3.0", float, 3.0),
            ("0.0", float, 0.0),
            ("00.10", float, 0.10),
            (".25", float, 0.25),
            ("5.", float, 5.0),
            ("0.", float, 0.0),
            ("-3.5", float, -3.5),
            ("+3.5", float, 3.5),
            ("-.5", float, -0.5),
            ("+.5", float, 0.5),
            ("-5.", float, -5.0),
            ("+5.", float, 5.0),
            # scientific notation
            ("1e3", float, 1000.0),
            ("1E3", float, 1000.0),
            ("1e+3", float, 1000.0),
            ("1e-3", float, 0.001),
            ("-2.5e2", float, -250.0),
            ("3.0E-2", float, 0.03),
            (".5e2", float, 50.0),
            ("5.e2", float, 500.0),
            ("+5.e-1", float, 0.5),],
    )
    def test_parametrized_scalar_conversions(self, value, expected_type, expected_value):
        d = {"k": value}
        out = convert_str_dict(d)

        assert isinstance(out["k"], expected_type)
        assert out["k"] == expected_value

    def test_nested_dict_is_converted_recursively(self):
        d = {
            "outer": {
                "t": "true",
                "i": "123",
                "f": "4.5",
                "inner": {
                    "x": "false", "y": "0"},}}
        out = convert_str_dict(d)

        assert out["outer"]["t"] is True
        assert out["outer"]["i"] == 123
        assert out["outer"]["f"] == 4.5
        assert out["outer"]["inner"]["x"] is False
        assert out["outer"]["inner"]["y"] == 0
