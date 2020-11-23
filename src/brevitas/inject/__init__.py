from _dependencies.injector import Injector
from _dependencies.injector import _InjectorType, __init__, let, injector_doc
from _dependencies.checks.circles import _check_circles
from _dependencies.checks.injector import _check_attrs_redefinition
from _dependencies.checks.injector import _check_dunder_name
from _dependencies.checks.injector import _check_inheritance
from _dependencies.checks.loops import _check_loops
from _dependencies.spec import _make_dependency_spec


class _ExtendedInjectorType(_InjectorType):

    """
    Copyright 2016-2020 Artem Malyshev

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the
       distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    def __new__(cls, class_name, bases, namespace):

        if not bases:
            namespace["__dependencies__"] = {}
            namespace["__wrapped__"] = None  # Doctest module compatibility.
            namespace["_subs_tree"] = None  # Typing module compatibility.
            return type.__new__(cls, class_name, bases, namespace)

        _check_inheritance(bases, (Injector, BaseInjector))
        ns = {}
        for attr in ("__module__", "__doc__", "__weakref__", "__qualname__"):
            try:
                ns[attr] = namespace.pop(attr)
            except KeyError:
                pass
        for name in namespace:
            _check_dunder_name(name)
            _check_attrs_redefinition(name)
        dependencies = {}
        for base in reversed(bases):
            dependencies.update(base.__dependencies__)
        for name, dep in namespace.items():
            dependencies[name] = _make_dependency_spec(name, dep)
        _check_loops(class_name, dependencies)
        _check_circles(dependencies)
        ns["__dependencies__"] = dependencies
        return type.__new__(cls, class_name, bases, ns)

    # Lack of __len__ magic on Injector breaks the debugger ability to inspect it
    def __len__(self):
        return None


BaseInjector = _ExtendedInjectorType(
    "Injector",
    (),
    {"__init__": __init__, "__doc__": injector_doc, "let": classmethod(let)})