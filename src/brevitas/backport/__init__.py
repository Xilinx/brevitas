"""
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
   NEC Laboratories America and IDIAP Research Institute nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Forked from PyTorch 2.0.1
"""

import builtins
import math


class SymInt:
    """
    Like an int (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """

    def __init__(self, node):
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __bool__(self):
        return self.node.bool_()

    def __int__(self):
        return self.node.int_()

    # Magic methods installed by torch.fx.experimental.symbolic_shapes

    def __eq__(self, other: object) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __lt__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __gt__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __le__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __ge__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __sym_max__(self, other):
        raise AssertionError("type stub not overridden")

    def __sym_min__(self, other):
        raise AssertionError("type stub not overridden")

    def __sym_float__(self):
        raise AssertionError("type stub not overridden")

    def __repr__(self):
        return str(self.node)


class SymFloat:
    """
    Like an float (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """

    def __init__(self, node):
        from torch.fx.experimental.symbolic_shapes import SymNode
        assert isinstance(node, SymNode)
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __bool__(self):
        return self.node.bool_()

    # Magic methods installed by torch.fx.experimental.symbolic_shapes

    def __eq__(self, other: object) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __lt__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __gt__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __le__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __ge__(self, other) -> builtins.bool:
        raise AssertionError("type stub not overridden")

    def __sym_max__(self, other):
        raise AssertionError("type stub not overridden")

    def __sym_min__(self, other):
        raise AssertionError("type stub not overridden")

    def __sym_int__(self):
        raise AssertionError("type stub not overridden")

    def __repr__(self):
        return self.node.str()


class SymBool:
    """
    Like an bool (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.

    Unlike regular bools, regular boolean operators will force extra guards instead
    of symbolically evaluate.  Use the bitwise operators instead to handle this.
    """

    def __init__(self, node):
        from torch.fx.experimental.symbolic_shapes import SymNode
        assert isinstance(node, SymNode)
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __bool__(self):
        return self.node.bool_()

    # Magic methods installed by torch.fx.experimental.symbolic_shapes
    def __and__(self, other) -> "SymBool":
        raise AssertionError("type stub not overridden")

    def __or__(self, other) -> "SymBool":
        raise AssertionError("type stub not overridden")

    # We very carefully define __sym_not__, and not a number of other
    # plausible alternatives:
    #
    #   - We do not override __not__ because this is not a real magic
    #     method; you cannot override the meaning of the not builtin in
    #     Python.  We use the name 'sym_not' to clarify that in user code you
    #     cannot use the builtin not or operator.not_ or operator.__not__ and
    #     hit this magic method; you must use our custom sym_not operator.
    #
    #   - We do not override the __invert__ method because SymBool is
    #     meant to be usable in situations where bool is expected.  However,
    #     bitwise negation ~a does the wrong thing with booleans (because
    #     bool is a subclass of int, so ~1 = -2 which is not falseish.)
    #     This would be a giant footgun, so we get around it by defining
    #     our own operator.  Note that bitwise and/or do the right thing,
    #     so we reuse the conventional operators there for readability.
    #
    def __sym_not__(self) -> "SymBool":
        raise AssertionError("type stub not overridden")

    def __repr__(self):
        return self.node.str()


def sym_not(a):
    r""" SymInt-aware utility for logical negation.

    Args:
        a (SymBool or bool): Object to negate
    """
    if hasattr(a, '__sym_not__'):
        return a.__sym_not__()
    return not a


def sym_float(a):
    r""" SymInt-aware utility for float casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """
    if isinstance(a, SymFloat):
        return a
    elif hasattr(a, '__sym_float__'):
        return a.__sym_float__()
    return py_float(a)  # type: ignore[operator]


def sym_int(a):
    r""" SymInt-aware utility for int casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """
    if isinstance(a, SymInt):
        return a
    elif isinstance(a, SymFloat):
        return math.floor(a) if a >= 0 else math.ceil(a)  # type: ignore[arg-type]
    return py_int(a)  # type: ignore[operator]


def sym_max(a, b):
    """ SymInt-aware utility for max()."""
    if isinstance(a, (SymInt, SymFloat)):
        return a.__sym_max__(b)
    elif isinstance(b, (SymInt, SymFloat)):
        # NB: If you actually care about preserving output type exactly
        # if you do something like max(0, 0.0), it is NOT sound to treat
        # min/max as commutative
        return b.__sym_max__(a)
    return builtins.max(a, b)  # type: ignore[operator]


def sym_min(a, b):
    """ SymInt-aware utility for max()."""
    if isinstance(a, (SymInt, SymFloat)):
        return a.__sym_min__(b)
    elif isinstance(b, (SymInt, SymFloat)):
        return b.__sym_min__(a)
    return builtins.min(a, b)  # type: ignore[operator]


# Populate magic methods on SymInt and SymFloat
import brevitas.backport.fx.experimental.symbolic_shapes
import brevitas.backport.fx
