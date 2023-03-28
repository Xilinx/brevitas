# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from inspect import isclass
import math
from typing import Optional, Tuple, Union
from warnings import warn

from torch import nn
from torch import Tensor
import torch.jit
from torch.nn.utils.rnn import PackedSequence

from brevitas import config
from brevitas.common import ExportMixin
from brevitas.inject import ExtendedInjector
from brevitas.inject import Injector
from brevitas.nn.utils import compute_channel_view_shape
from brevitas.quant_tensor import QuantTensor

from .utils import filter_kwargs


class _CachedIO:

    def __init__(self, quant_tensor: QuantTensor, metadata_only: bool):
        self.shape = quant_tensor.value.shape
        if metadata_only:
            self.quant_tensor = quant_tensor.set(value=None)
        else:
            self.quant_tensor = quant_tensor

    @property
    def scale(self):
        return self.quant_tensor.scale

    @property
    def zero_point(self):
        return self.quant_tensor.zero_point

    @property
    def bit_width(self):
        return self.quant_tensor.bit_width

    @property
    def signed(self):
        return self.quant_tensor.signed


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

    def __init__(
            self,
            return_quant_tensor: bool,
            cache_inference_quant_inp: bool = False,
            cache_inference_quant_out: bool = False,
            cache_quant_io_metadata_only: bool = True):
        ExportMixin.__init__(self)
        self.accept_quant_tensor = True
        self.return_quant_tensor = return_quant_tensor
        self.cache_inference_quant_inp = cache_inference_quant_inp
        self.cache_inference_quant_out = cache_inference_quant_out
        self.cache_quant_io_metadata_only = cache_quant_io_metadata_only
        self._cached_inp = None
        self._cached_out = None

    @property
    @abstractmethod
    def channelwise_separable(self) -> bool:
        pass

    @property
    def is_quant_input_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self._cached_inp is not None:
            return self._cached_inp.signed
        else:
            return None

    def _set_global_is_quant_layer(self, value):
        config._IS_INSIDE_QUANT_LAYER = value

    def quant_input_scale(self):
        if self._cached_inp is not None:
            return self._cached_inp.scale
        else:
            return None

    def quant_input_zero_point(self):
        if self._cached_inp is not None:
            return self._cached_inp.zero_point
        else:
            return None

    def quant_input_bit_width(self):
        if self._cached_inp is not None:
            return self._cached_inp.bit_width
        else:
            return None

    @property
    def is_quant_output_signed(self) -> Optional[bool]:  # tri-valued logic output
        if self._cached_out is not None:
            return self._cached_out.signed
        else:
            return None

    def quant_output_scale(self):
        if self._cached_out is not None:
            return self._cached_out.scale
        else:
            return None

    def quant_output_zero_point(self):
        if self._cached_out is not None:
            return self._cached_out.zero_point
        else:
            return None

    def quant_output_bit_width(self):
        if self._cached_out is not None:
            return self._cached_out.bit_width
        else:
            return None

    def unpack_input(self, inp: Union[Tensor, QuantTensor]):
        self._set_global_is_quant_layer(True)
        # Hack to recognize a QuantTensor that has decayed to a tuple
        # when used as input to tracing (e.g. during ONNX export)
        if (torch._C._get_tracing_state() is not None and isinstance(inp, tuple) and
                len(inp) == len(QuantTensor._fields) and all([isinstance(t, Tensor) for t in inp])):
            inp = QuantTensor(*inp)
        if isinstance(inp, QuantTensor):
            # don't cache values during export pass
            if not self.training and not self._export_mode and self.cache_inference_quant_inp:
                cached_inp = _CachedIO(inp.detach(), self.cache_quant_io_metadata_only)
                self._cached_inp = cached_inp
            return inp
        else:
            inp = QuantTensor(inp, training=self.training)
            if not self.training and self.cache_inference_quant_inp:
                cached_inp = _CachedIO(inp.detach(), self.cache_quant_io_metadata_only)
                self._cached_inp = cached_inp
            return inp

    def pack_output(self, quant_output: QuantTensor):
        if not self.training and self.cache_inference_quant_out:
            self._cached_out = _CachedIO(quant_output.detach(), self.cache_quant_io_metadata_only)
        self._set_global_is_quant_layer(False)
        if self.return_quant_tensor:
            return quant_output
        else:
            return quant_output.value


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
        acc_scale = None
        acc_bit_width = None
        quant_weight_ih = gate.input_weight()
        quant_weight_hh = gate.hidden_weight()
        if quant_input.bit_width is not None:
            acc_bit_width = None  # TODO
        if quant_input.scale is not None and quant_weight_ih.scale is not None:
            acc_scale_shape = compute_channel_view_shape(quant_input.value, channel_dim=1)
            acc_scale = quant_weight_ih.scale.view(acc_scale_shape)
            acc_scale = acc_scale * quant_input.scale.view(acc_scale_shape)
        quant_bias = gate.bias_quant(gate.bias, acc_scale, acc_bit_width)
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
        elif not isinstance(inp, QuantTensor):
            quant_input = QuantTensor(quant_input)
        return quant_input

    def maybe_quantize_state(self, inp, state, quant):
        if state is None:
            batch_size = inp.size(0) if self.cell.batch_first else inp.size(1)
            quant_state = torch.zeros(
                int(batch_size), self.hidden_size, dtype=inp.dtype, device=inp.device)
            quant_state = QuantTensor(quant_state)
        else:
            quant_state = quant(state)
        return quant_state

    def pack_quant_outputs(self, quant_outputs):
        # In export mode, quant_outputs has the shape of the output concatenated value
        if self.export_mode:
            if self.return_quant_tensor:
                return QuantTensor(
                    quant_outputs,
                    self.io_quant.scale(),
                    self.io_quant.zero_point(),
                    self.io_quant.bit_width(),
                    self.io_quant.is_signed,
                    self.training)
            else:
                return quant_outputs
        seq_dim = 1 if self.cell.batch_first else 0
        if self.return_quant_tensor:
            outputs = [
                QuantTensor(
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
        if self.export_mode:
            if self.return_quant_tensor:
                quant_state = QuantTensor(
                    torch.unsqueeze(quant_state, dim=0),
                    quant.scale(),
                    quant.zero_point(),
                    quant.bit_width(),
                    quant.is_signed,
                    self.training)
            else:
                quant_state = torch.unsqueeze(quant_state, dim=0)
        else:
            if self.return_quant_tensor:
                quant_state = QuantTensor(
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
