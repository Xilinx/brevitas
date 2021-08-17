import inspect

from _dependencies.injector import Injector
from _dependencies.injector import _InjectorType, __init__, let, injector_doc
from _dependencies.checks.circles import _check_circles
from _dependencies.checks.injector import _check_attrs_redefinition
from _dependencies.checks.injector import _check_dunder_name
from _dependencies.checks.injector import _check_inheritance
from _dependencies.checks.loops import _check_loops
from _dependencies.spec import _make_init_spec, _make_this_spec, _make_dependency_spec
from _dependencies.this import This
from _dependencies.exceptions import DependencyError
from _dependencies.attributes import _Replace
from _dependencies.replace import _deep_replace_dependency
from dependencies import value, this  # noqa


def _replace_dependency(injector, current_attr, spec):
    replaced_dependency = injector.__dependencies__[current_attr]
    injector.__dependencies__[current_attr] = spec
    _check_loops(injector.__name__, injector.__dependencies__)
    _check_circles(injector.__dependencies__)
    return replaced_dependency


class _ExtendedInjectorType(_InjectorType):
    """
    Extended _InjectorType based on dependencies 2.0.1.
    - Fixes issues with interacting debugging.
    - Allows to inject an object instantiated from a class returned from a @value function.
    - Allows to return this.something from a @value function, to inject stuff conditionally.
    - Add __signature__ and __text_signature__.

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
            namespace["__signature__"] = None  # Sphinx compatibility.
            namespace["__func__"] = None  # JIT tracing compatibility.
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
        return 0

    def __getattr__(cls, attrname):
        __tracebackhide__ = True

        cache, cached = {"__self__": cls}, {"__self__"}
        current_attr, attrs_stack = attrname, [attrname]
        have_default = False
        replaced_dependencies = {}

        while attrname not in cache:

            spec = cls.__dependencies__.get(current_attr)

            if spec is None:
                if have_default:
                    cached.add(current_attr)
                    current_attr = attrs_stack.pop()
                    have_default = False
                    continue
                if len(attrs_stack) > 1:
                    message = "{!r} can not resolve attribute {!r} while building {!r}".format(  # noqa: E501
                        cls.__name__, current_attr, attrs_stack.pop()
                    )
                else:
                    message = "{!r} can not resolve attribute {!r}".format(
                        cls.__name__, current_attr
                    )
                raise DependencyError(message)

            marker, attribute, args, have_defaults = spec

            if set(args).issubset(cached):
                kwargs = {k: cache[k] for k in args if k in cache}

                try:
                    dependency = attribute(**kwargs)
                    if ('nested' not in marker
                            and inspect.isclass(dependency)
                            and not current_attr.endswith("_class")):
                        spec = _make_init_spec(dependency)
                        replaced_dependency = _replace_dependency(cls, current_attr, spec)
                        replaced_dependencies[current_attr] = replaced_dependency
                        continue
                    elif isinstance(dependency, This):
                        spec = _make_this_spec(dependency)
                        replaced_dependency = _replace_dependency(cls, current_attr, spec)
                        replaced_dependencies[current_attr] = replaced_dependency
                        continue
                    else:
                        cache[current_attr] = dependency
                except _Replace as replace:
                    _deep_replace_dependency(cls, current_attr, replace)
                    _check_loops(cls.__name__, cls.__dependencies__)
                    _check_circles(cls.__dependencies__)
                    continue

                cached.add(current_attr)
                current_attr = attrs_stack.pop()
                have_default = False
                continue

            for n, arg in enumerate(args, 1):
                if arg not in cached:
                    attrs_stack.append(current_attr)
                    current_attr = arg
                    have_default = False if n < have_defaults else True
                    break

        # Restore @value dependencies that returned a class from the result class
        # to their defining function
        for attr, dep in replaced_dependencies.items():
            cls.__dependencies__[attr] = dep

        return cache[attrname]



ExtendedInjector = _ExtendedInjectorType(
    "Injector",
    (),
    {"__init__": __init__, "__doc__": injector_doc, "let": classmethod(let)})
BaseInjector = ExtendedInjector # retrocompatibility wrt naming