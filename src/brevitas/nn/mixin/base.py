# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from inspect import isclass
import math
from typing import Optional, Tuple, Union
from warnings import warn

import packaging.version
import torch
from torch import nn
from torch import Tensor
import torch.jit
from torch.nn.utils.rnn import PackedSequence

from brevitas import config
from brevitas import is_dynamo_compiling
from brevitas import torch_version
from brevitas.common import ExportMixin
from brevitas.inject import ExtendedInjector
from brevitas.inject import Injector
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import FloatQuantTensor
from brevitas.quant_tensor import IntQuantTensor
from brevitas.quant_tensor import QuantTensor
from brevitas.quant_tensor.groupwise_float_quant_tensor import GroupwiseFloatQuantTensor
from brevitas.quant_tensor.groupwise_int_quant_tensor import GroupwiseIntQuantTensor

from .utils import filter_kwargs


class QuantProxyMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            quant,
            proxy_protocol,
            none_quant_injector,
            proxy_prefix: str,
            kwargs_prefix: str,
            **kwargs):
        proxy_name = proxy_prefix + 'quant'
        if quant is None:
            quant_injector = none_quant_injector.let(**filter_kwargs(kwargs_prefix, kwargs))
            quant = quant_injector.proxy_class(self, quant_injector)
        elif isclass(quant) and issubclass(quant, (Injector, ExtendedInjector)):
            quant_injector = quant
            quant_injector = quant_injector.let(**filter_kwargs(kwargs_prefix, kwargs))
            quant = quant_injector.proxy_class(self, quant_injector)
        else:
            if not isinstance(quant, proxy_protocol):
                raise RuntimeError(
                    "The quantizer passed does not adhere to the quantization protocol.")
            quant.add_tracked_module(self)
            if filter_kwargs(kwargs_prefix, kwargs):
                warn('Keyword arguments are being passed but they not being used.')
        setattr(self, proxy_name, quant)


class QuantLayerMixin(ExportMixin):
    __metaclass__ = ABCMeta

    def __init__(self, return_quant_tensor: bool):
        ExportMixin.__init__(self)
        self.accept_quant_tensor = True
        self.return_quant_tensor = return_quant_tensor

    @property
    @abstractmethod
    def channelwise_separable(self) -> bool:
        pass

    def get_quant_tensor_class(self, inp: Union[Tensor, QuantTensor]):
        quant_tensor_classes = [
            IntQuantTensor, FloatQuantTensor, GroupwiseIntQuantTensor, GroupwiseFloatQuantTensor]
        for qt_class in quant_tensor_classes:
            if len(inp) == len(qt_class._fields):
                return qt_class
        return None

    def unpack_input(self, inp: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        # Hack to recognize a QuantTensor that has decayed to a tuple
        # when used as input to tracing (e.g. during ONNX export)
        if (torch._C._get_tracing_state() is not None and isinstance(inp, tuple) and
                all([isinstance(t, Tensor) for t in inp])):
            qt_class = self.get_quant_tensor_class(inp)
            if qt_class is not None:
                inp = qt_class(*inp)
        if not torch._C._get_tracing_state() and not is_dynamo_compiling():
            if isinstance(inp, QuantTensor):
                inp = inp.set(value=inp.value.rename(None))
            else:
                inp = inp.rename(None)
        return inp

    def pack_output(self, quant_output: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        if self.return_quant_tensor:
            assert isinstance(quant_output, QuantTensor), 'QuantLayer is not correctly configured, check if warnings were raised'
            return quant_output
        else:
            return _unpack_quant_tensor(quant_output)


class QuantRecurrentLayerMixin(ExportMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            cell: nn.Module,
            io_quant: nn.Module,
            input_size: int,
            hidden_size: int,
            reverse_input: bool,
            quantize_output_only: bool,
            shared_input_hidden_weights: bool,
            return_quant_tensor: bool):
        ExportMixin.__init__(self)
        self.cell = cell
        self.io_quant = io_quant
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reverse_input = reverse_input
        self.quantize_output_only = quantize_output_only
        self.shared_input_hidden_weights = shared_input_hidden_weights
        self.return_quant_tensor = return_quant_tensor
        self.accept_quant_tensor = True
        self.fast_mode = True
        self._fast_cell = None

    @property
    @abstractmethod
    def weights_to_share(self):
        pass

    @property
    @abstractmethod
    def quantizers_to_share(self):
        pass

    @property
    @abstractmethod
    def fast_cell(self):
        pass

    @property
    def requires_export_handler(self):
        return True

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        output_dict = super(QuantRecurrentLayerMixin, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in list(output_dict.keys()):
            if '_fast_cell' in key:
                del output_dict[key]
        return output_dict

    @staticmethod
    def gate_params_fwd(gate, quant_input):
        quant_weight_ih = gate.input_weight()
        quant_weight_hh = gate.hidden_weight()
        quant_bias = gate.bias_quant(gate.bias, quant_input, quant_weight_ih)
        return quant_weight_ih, quant_weight_hh, quant_bias

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if 'gate' in name:
                nn.init.uniform_(weight, -stdv, stdv)

    def maybe_quantize_input(self, inp):
        if isinstance(inp, PackedSequence):
            raise RuntimeError("PackedSequence input currently not supported.")
        quant_input = inp
        if not self.quantize_output_only:
            quant_input = self.io_quant(quant_input)
        return quant_input

    def maybe_quantize_state(self, inp, state, quant):
        if state is None:
            batch_size = inp.size(0) if self.cell.batch_first else inp.size(1)
            quant_state = torch.zeros(
                int(batch_size), self.hidden_size, dtype=inp.dtype, device=inp.device)
        else:
            quant_state = quant(state)
        return quant_state

    def pack_quant_outputs(self, quant_outputs):
        # In export mode, quant_outputs has the shape of the output concatenated value
        # Even though we check that return_quant_tensor can be enabled only with io_quant != None,
        # inner layers in a deep network overrides it, so we check again.
        if self.export_mode:
            if self.return_quant_tensor and self.io_quant.is_quant_enabled:
                return IntQuantTensor(
                    quant_outputs,
                    self.io_quant.scale(),
                    self.io_quant.zero_point(),
                    self.io_quant.bit_width(),
                    self.io_quant.is_signed,
                    self.training)
            else:
                return quant_outputs
        seq_dim = 1 if self.cell.batch_first else 0
        if self.return_quant_tensor and self.io_quant.is_quant_enabled:
            outputs = [
                IntQuantTensor(
                    torch.unsqueeze(quant_output[0], dim=seq_dim),
                    quant_output[1],
                    quant_output[2],
                    quant_output[3],
                    self.io_quant.is_signed,
                    self.training) for quant_output in quant_outputs]
        else:
            outputs = [torch.unsqueeze(o[0], dim=seq_dim) for o in quant_outputs]
        if self.reverse_input:
            return torch.cat(list(reversed(outputs)), dim=seq_dim)
        else:
            return torch.cat(outputs, dim=seq_dim)

    def pack_quant_state(self, quant_state, quant):
        # Even though we check that return_quant_tensor can be enabled only with quant != None,
        # inner layers in a deep network overrides it, so we check again.
        if self.export_mode:
            if self.return_quant_tensor and quant.is_quant_enabled:
                quant_state = IntQuantTensor(
                    torch.unsqueeze(quant_state, dim=0),
                    quant.scale(),
                    quant.zero_point(),
                    quant.bit_width(),
                    quant.is_signed,
                    self.training)
            else:
                quant_state = torch.unsqueeze(quant_state, dim=0)
        else:
            if self.return_quant_tensor and quant.is_quant_enabled:
                quant_state = IntQuantTensor(
                    torch.unsqueeze(quant_state[0], dim=0),
                    quant_state[1],
                    quant_state[2],
                    quant_state[3],
                    quant.is_signed,
                    self.training)
            else:
                quant_state = torch.unsqueeze(quant_state[0], dim=0)
        return quant_state

    def _wrap_act_proxy(self, quant_name):

        class _Wrapper(nn.Module):

            def __init__(self, module_to_wrap=None):
                super(_Wrapper, self).__init__()
                if module_to_wrap is None:
                    module_to_wrap = nn.Identity()
                self.module_to_wrap = module_to_wrap

            def forward(
                self, x: torch.Tensor
            ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
                x = self.module_to_wrap(x)
                return (x, None, None, None)

        proxy = getattr(self.cell, quant_name)
        if proxy.fused_activation_quant_proxy is None:
            proxy = _Wrapper(proxy.fused_activation_quant_proxy)
        else:
            proxy = proxy.fused_activation_quant_proxy
        return proxy
