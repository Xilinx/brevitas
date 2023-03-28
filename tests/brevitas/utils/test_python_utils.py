# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from enum import auto

from brevitas.utils.python_utils import AutoName


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
