from abc import ABCMeta
from typing import List, Optional
from inspect import isclass

import torch
from torch import Tensor

from brevitas.inject import ExtendedInjector, Injector
from brevitas.nn.mixin.base import QuantProxyMixin
from brevitas.nn.mixin.utils import filter_kwargs
from brevitas.nn.mixin import WeightQuantType
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyProtocol


@torch.jit.script
def reverse(lst: List[Tensor]) -> List[Tensor]:
    out = torch.jit.annotate(List[Tensor], [])
    end = len(lst)
    index = len(lst) - 1
    for i in range(end):
        out += [lst[index]]
        index = index - 1
    return out


class QuantIOMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            io_quant,
            **kwargs):
        kwargs_prefix = 'io_quant_'
        if io_quant is None:
            quant_injector = ExtendedInjector.let(tensor_quant=None)
            quant_injector = quant_injector.let(**filter_kwargs(kwargs_prefix, kwargs))
        elif isclass(io_quant) and issubclass(io_quant, (Injector, ExtendedInjector)):
            quant_injector = io_quant
            quant_injector = quant_injector.let(**filter_kwargs(kwargs_prefix, kwargs))
        else:
            raise RuntimeError("The IO quantizer passed is not supported.")
        self.io_quant = quant_injector.tensor_quant


class QuantWeightRMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight_quant: Optional[WeightQuantType],
            **kwargs):
        QuantProxyMixin.__init__(
            self,
            quant=weight_quant,
            proxy_from_injector_impl=WeightQuantProxyFromInjector,
            proxy_protocol=WeightQuantProxyProtocol,
            proxy_kwargs={'tracked_parameter_names': ['weight_ri', 'weight_rh']},
            kwargs_prefix='weight_',
            proxy_prefix='weight_r_',
            **kwargs)


class QuantWeightCMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight_quant: Optional[WeightQuantType],
            **kwargs):
        QuantProxyMixin.__init__(
            self,
            quant=weight_quant,
            proxy_from_injector_impl=WeightQuantProxyFromInjector,
            proxy_protocol=WeightQuantProxyProtocol,
            proxy_kwargs={'tracked_parameter_names': ['weight_ci', 'weight_ch']},
            kwargs_prefix='weight_',
            proxy_prefix='weight_c_',
            **kwargs)


class QuantWeightNMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight_quant: Optional[WeightQuantType],
            **kwargs):
        QuantProxyMixin.__init__(
            self,
            quant=weight_quant,
            proxy_from_injector_impl=WeightQuantProxyFromInjector,
            proxy_protocol=WeightQuantProxyProtocol,
            proxy_kwargs={'tracked_parameter_names': ['weight_ni', 'weight_nh']},
            kwargs_prefix='weight_',
            proxy_prefix='weight_n_',
            **kwargs)