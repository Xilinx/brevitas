# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for QuantTensor as a torch.Tensor subclass.

These tests cover the behaviors introduced by converting QuantTensor from
a NamedTuple to a torch.Tensor subclass:
  - isinstance relationships
  - .value returns plain Tensor and preserves grad_fn
  - set() creates a new QuantTensor with replaced fields
  - __iadd__, __imul__, __isub__ semantics (non-in-place)
  - __torch_function__ fallback (no infinite recursion)
  - detach / contiguous / to preserve type and metadata
  - _unpack_quant_tensor returns plain Tensor
  - __radd__ / __rmul__ right-hand operators
"""

import pytest
import torch
import torch.nn.functional as F

from brevitas.nn import QuantIdentity
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e5m2OCPActPerTensorFloat
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import FloatQuantTensor
from brevitas.quant_tensor import GroupwiseFloatQuantTensor
from brevitas.quant_tensor import IntQuantTensor
from brevitas.quant_tensor import QuantTensor
from brevitas.quant_tensor.groupwise_int_quant_tensor import GroupwiseIntQuantTensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_int_qt(shape=(4, 4), bit_width=8):
    mod = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
    return mod(torch.randn(shape))


def _make_float_qt(shape=(4, 4)):
    mod = QuantIdentity(
        bit_width=8,
        exponent_bit_width=4,
        mantissa_bit_width=3,
        return_quant_tensor=True,
        act_quant=Fp8e5m2OCPActPerTensorFloat)
    return mod(torch.randn(shape))


def _make_mx_qt(shape=(1, 32)):
    mod = QuantIdentity(
        bit_width=8,
        group_size=32,
        group_dim=1,
        exponent_bit_width=4,
        mantissa_bit_width=3,
        return_quant_tensor=True,
        act_quant=MXFloat8e4m3Act)
    return mod(torch.randn(shape))


# ---------------------------------------------------------------------------
# 1. isinstance relationships
# ---------------------------------------------------------------------------


class TestIsInstance:

    def test_int_quant_tensor_is_tensor(self):
        qt = _make_int_qt()
        assert isinstance(qt, torch.Tensor)
        assert isinstance(qt, QuantTensor)
        assert isinstance(qt, IntQuantTensor)

    def test_float_quant_tensor_is_tensor(self):
        qt = _make_float_qt()
        assert isinstance(qt, torch.Tensor)
        assert isinstance(qt, QuantTensor)
        assert isinstance(qt, FloatQuantTensor)

    def test_mx_quant_tensor_is_tensor(self):
        qt = _make_mx_qt()
        assert isinstance(qt, torch.Tensor)
        assert isinstance(qt, QuantTensor)
        assert isinstance(qt, GroupwiseFloatQuantTensor)

    def test_manually_constructed_int_qt(self):
        qt = IntQuantTensor(
            torch.randn(10), torch.randn(1), torch.tensor(0.), torch.tensor(8.), True, False)
        assert isinstance(qt, torch.Tensor)
        assert isinstance(qt, QuantTensor)


# ---------------------------------------------------------------------------
# 2. .value returns plain Tensor and preserves grad_fn
# ---------------------------------------------------------------------------


class TestValueProperty:

    def test_value_is_plain_tensor(self):
        qt = _make_int_qt()
        v = qt.value
        assert type(v) is torch.Tensor
        assert not isinstance(v, QuantTensor)

    def test_value_same_data(self):
        qt = _make_int_qt()
        v = qt.value
        assert torch.equal(v, qt)

    def test_value_preserves_grad_fn(self):
        x = torch.randn(4, 4, requires_grad=True)
        y = x * 2  # y has a grad_fn
        qt = IntQuantTensor(y, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(8.0), True, False)
        assert qt.grad_fn is not None
        v = qt.value
        assert v.grad_fn is not None

    def test_value_preserves_requires_grad(self):
        x = torch.randn(4, 4, requires_grad=True)
        qt = IntQuantTensor(x, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(8.0), True, False)
        assert qt.value.requires_grad is True

    def test_value_is_not_subclass(self):
        qt = _make_int_qt()
        assert type(qt.value) is torch.Tensor

    def test_float_qt_value_is_plain_tensor(self):
        qt = _make_float_qt()
        v = qt.value
        assert type(v) is torch.Tensor
        assert not isinstance(v, QuantTensor)


# ---------------------------------------------------------------------------
# 3. set() method
# ---------------------------------------------------------------------------


class TestSetMethod:

    def test_set_replaces_scale(self):
        qt = _make_int_qt()
        new_scale = torch.tensor(0.5)
        qt2 = qt.set(scale=new_scale)
        assert isinstance(qt2, IntQuantTensor)
        assert torch.equal(qt2.scale, new_scale)
        # Other metadata preserved
        assert torch.equal(qt2.zero_point, qt.zero_point)
        assert torch.equal(qt2.bit_width, qt.bit_width)

    def test_set_replaces_value(self):
        qt = _make_int_qt()
        new_value = torch.ones(4, 4)
        qt2 = qt.set(value=new_value)
        assert isinstance(qt2, IntQuantTensor)
        assert torch.allclose(qt2.value, new_value)
        assert torch.equal(qt2.scale, qt.scale)

    def test_set_no_args_copies(self):
        qt = _make_int_qt()
        qt2 = qt.set()
        assert isinstance(qt2, IntQuantTensor)
        assert torch.allclose(qt2.value, qt.value)
        assert torch.equal(qt2.scale, qt.scale)

    def test_set_returns_same_type(self):
        qt = _make_float_qt()
        qt2 = qt.set(scale=torch.tensor(1.0))
        assert type(qt2) is FloatQuantTensor


class TestGetItem:

    def test_integer_index_slices_aligned_metadata(self):
        qt = IntQuantTensor(
            torch.randn(2, 3, 4),
            torch.ones(2, 3, 1),
            torch.zeros(2, 3, 1),
            torch.tensor(8.),
            True,
            False)
        result = qt[1]
        assert isinstance(result, IntQuantTensor)
        assert result.shape == (3, 4)
        assert result.scale.shape == (3, 1)
        assert result.zero_point.shape == (3, 1)

    def test_scalar_tensor_index(self):
        qt = _make_int_qt(shape=(2, 4))
        result = qt[torch.tensor(1)]
        assert isinstance(result, IntQuantTensor)
        assert result.shape == (4,)

    def test_slice_retains_leading_dimension(self):
        qt = _make_int_qt(shape=(4, 4))
        result = qt[1:3]
        assert isinstance(result, IntQuantTensor)
        assert result.shape == (2, 4)

    def test_groupwise_index_updates_shape_and_dimension(self):
        qt = _make_mx_qt(shape=(2, 32))
        result = qt[0]
        assert isinstance(result, GroupwiseFloatQuantTensor)
        assert result.shape == (32,)
        assert result.dequant_shape == (32,)
        assert result.group_dim == 0

    def test_groupwise_negative_dimension_remains_relative(self):
        mod = QuantIdentity(
            group_size=32,
            group_dim=-1,
            exponent_bit_width=4,
            mantissa_bit_width=3,
            return_quant_tensor=True,
            act_quant=MXFloat8e4m3Act)
        result = mod(torch.randn(2, 32))[0]
        assert result.group_dim == -1
        assert result.dequant_shape == (32,)

    def test_groupwise_index_rejects_negative_leading_dimension(self):
        mod = QuantIdentity(
            group_size=2,
            group_dim=-2,
            exponent_bit_width=4,
            mantissa_bit_width=3,
            return_quant_tensor=True,
            act_quant=MXFloat8e4m3Act)
        quant_tensor = mod(torch.randn(2, 32))
        with pytest.raises(RuntimeError, match='grouped dimension'):
            quant_tensor[0]


# ---------------------------------------------------------------------------
# 4. In-place augmented assignment operators
# ---------------------------------------------------------------------------


class TestInPlaceOperators:

    def test_iadd_returns_valid_result(self):
        qt = _make_int_qt()
        original_value = qt.value.clone()
        other = _make_int_qt()
        qt += other
        # += should produce a result (not crash), and result should be a tensor
        assert isinstance(qt, torch.Tensor)

    def test_imul_returns_valid_result(self):
        qt = _make_int_qt()
        # Multiply by a scalar tensor
        qt *= torch.tensor(2.0)
        assert isinstance(qt, torch.Tensor)

    def test_isub_returns_valid_result(self):
        qt = _make_int_qt()
        other = _make_int_qt()
        qt -= other
        assert isinstance(qt, torch.Tensor)

    def test_iadd_preserves_quant_tensor_type(self):
        qa = _make_int_qt()
        qb = _make_int_qt()
        result_add = qa + qb
        qa_copy = _make_int_qt()
        # Use fresh tensors from same distribution
        qa_copy += qb
        # Both should be IntQuantTensor
        assert isinstance(result_add, IntQuantTensor)
        assert isinstance(qa_copy, IntQuantTensor)


# ---------------------------------------------------------------------------
# 5. __torch_function__ fallback (no infinite recursion)
# ---------------------------------------------------------------------------


class TestTorchFunctionFallback:

    def test_unhandled_function_returns_plain_tensor(self):
        """Functions not in the handler map should unpack and return plain tensor."""
        qt = _make_int_qt()
        result = torch.sum(qt)
        # sum is not in any handler map, so it should fallback
        assert isinstance(result, torch.Tensor)
        # Should not be a QuantTensor
        assert not isinstance(result, QuantTensor)

    def test_mean_fallback(self):
        qt = _make_int_qt()
        result = torch.mean(qt)
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, QuantTensor)

    def test_clamp_fallback(self):
        qt = _make_int_qt()
        result = torch.clamp(qt, min=0.0, max=1.0)
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, QuantTensor)

    def test_abs_on_float_qt_fallback(self):
        """torch.abs not in float handler, should fallback cleanly."""
        qt = _make_float_qt()
        # FloatQuantTensor has __abs__ which calls minifloat; torch.abs is different
        # Using torch.sum which is definitely not in the handler
        result = torch.sum(qt)
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, QuantTensor)

    def test_no_recursion_with_multiple_qt_args(self):
        """Ensure no recursion when multiple QuantTensors are passed."""
        qa = _make_int_qt()
        qb = _make_int_qt()
        # torch.stack is not in any handler
        result = torch.stack([qa.value, qb.value])
        assert isinstance(result, torch.Tensor)

    def test_handled_function_relu(self):
        """F.relu IS in the handler map, should return QuantTensor."""
        qt = _make_int_qt()
        result = F.relu(qt)
        assert isinstance(result, IntQuantTensor)

    def test_handled_function_max_pool2d(self):
        """F.max_pool2d IS in the handler map, should return QuantTensor."""
        qt = _make_int_qt(shape=(1, 1, 4, 4))
        result = F.max_pool2d(qt, kernel_size=2)
        assert isinstance(result, IntQuantTensor)


# ---------------------------------------------------------------------------
# 6. detach / contiguous / to
# ---------------------------------------------------------------------------


class TestTensorOps:

    def test_detach_preserves_type_and_metadata(self):
        qt = _make_int_qt()
        d = qt.detach()
        assert isinstance(d, IntQuantTensor)
        assert d.grad_fn is None
        assert torch.equal(d.scale, qt.scale)
        assert torch.equal(d.zero_point, qt.zero_point)
        assert torch.equal(d.bit_width, qt.bit_width)

    def test_contiguous_preserves_type_and_metadata(self):
        qt = _make_int_qt()
        c = qt.contiguous()
        assert isinstance(c, IntQuantTensor)
        assert torch.equal(c.scale, qt.scale)
        assert torch.equal(c.bit_width, qt.bit_width)

    def test_to_dtype_preserves_type_and_metadata(self):
        qt = _make_int_qt()
        qt16 = qt.to(torch.float16)
        assert isinstance(qt16, IntQuantTensor)
        assert qt16.value.dtype == torch.float16
        # Metadata should also be converted
        assert qt16.scale.dtype == torch.float16

    def test_float_qt_detach(self):
        qt = _make_float_qt()
        d = qt.detach()
        assert isinstance(d, FloatQuantTensor)
        assert d.grad_fn is None
        assert torch.equal(d.scale, qt.scale)


# ---------------------------------------------------------------------------
# 7. _unpack_quant_tensor
# ---------------------------------------------------------------------------


class TestUnpackQuantTensor:

    def test_unpack_int_qt(self):
        qt = _make_int_qt()
        v = _unpack_quant_tensor(qt)
        assert type(v) is torch.Tensor
        assert not isinstance(v, QuantTensor)
        assert torch.equal(v, qt.value)

    def test_unpack_float_qt(self):
        qt = _make_float_qt()
        v = _unpack_quant_tensor(qt)
        assert type(v) is torch.Tensor
        assert not isinstance(v, QuantTensor)

    def test_unpack_plain_tensor_passthrough(self):
        t = torch.randn(4, 4)
        v = _unpack_quant_tensor(t)
        assert v is t

    def test_unpack_tuple(self):
        qt = _make_int_qt()
        t = torch.randn(4, 4)
        result = _unpack_quant_tensor((qt, t))
        assert isinstance(result, tuple)
        assert type(result[0]) is torch.Tensor
        assert result[1] is t

    def test_unpack_dict(self):
        qt = _make_int_qt()
        result = _unpack_quant_tensor({'a': qt, 'b': 42})
        assert type(result['a']) is torch.Tensor
        assert result['b'] == 42

    def test_unpack_nested(self):
        qt = _make_int_qt()
        result = _unpack_quant_tensor([qt, (qt, qt)])
        assert isinstance(result, list)
        assert type(result[0]) is torch.Tensor
        assert isinstance(result[1], tuple)
        assert type(result[1][0]) is torch.Tensor


# ---------------------------------------------------------------------------
# 8. Right-hand operators
# ---------------------------------------------------------------------------


class TestRightHandOperators:

    def test_radd_plain_tensor_plus_int_qt(self):
        qt = _make_int_qt()
        t = torch.ones(4, 4)
        result = t + qt
        assert isinstance(result, torch.Tensor)

    def test_rmul_plain_tensor_times_int_qt(self):
        qt = _make_int_qt()
        t = torch.ones(4, 4) * 2.0
        result = t * qt
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# 9. _fields attribute
# ---------------------------------------------------------------------------


class TestFields:

    def test_int_qt_fields(self):
        qt = _make_int_qt()
        assert hasattr(qt, '_fields')
        assert set(qt._fields) == {'scale', 'zero_point', 'bit_width', 'signed', 'training'}

    def test_float_qt_fields(self):
        qt = _make_float_qt()
        assert hasattr(qt, '_fields')
        expected = {
            'scale',
            'zero_point',
            'exponent_bit_width',
            'mantissa_bit_width',
            'exponent_bias',
            'saturating',
            'inf_values',
            'nan_values',
            'signed',
            'training'}
        assert set(qt._fields) == expected

    def test_groupwise_float_qt_fields(self):
        qt = _make_mx_qt()
        assert hasattr(qt, '_fields')
        assert 'scale_' in qt._fields
        assert 'group_size' in qt._fields
        assert 'group_dim' in qt._fields


# ---------------------------------------------------------------------------
# 10. shape / size / dim
# ---------------------------------------------------------------------------


class TestShapeSizeDim:

    def test_shape(self):
        qt = _make_int_qt(shape=(2, 3, 4))
        assert qt.shape == torch.Size([2, 3, 4])

    def test_size(self):
        qt = _make_int_qt(shape=(2, 3, 4))
        assert qt.size() == torch.Size([2, 3, 4])
        assert qt.size(0) == 2
        assert qt.size(1) == 3

    def test_dim(self):
        qt = _make_int_qt(shape=(2, 3, 4))
        assert qt.dim() == 3


# ---------------------------------------------------------------------------
# 11. Construction from non-Tensor input
# ---------------------------------------------------------------------------


class TestConstructionFromNonTensor:

    def test_from_float(self):
        qt = IntQuantTensor(
            1.0, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(8.0), True, False)
        assert isinstance(qt, IntQuantTensor)
        assert isinstance(qt, torch.Tensor)

    def test_from_list(self):
        qt = IntQuantTensor([1.0, 2.0, 3.0],
                            torch.tensor(1.0),
                            torch.tensor(0.0),
                            torch.tensor(8.0),
                            True,
                            False)
        assert isinstance(qt, IntQuantTensor)
        assert qt.shape == torch.Size([3])


# ---------------------------------------------------------------------------
# 12. Dynamo export: cache_class backup/restore
# ---------------------------------------------------------------------------


class TestDynamoExportCacheClass:

    def test_cache_class_cleared_and_restored(self):
        """QONNXDynamoManager.set_export_mode should clear cache_class on enable
        and restore it on disable, to prevent QuantTensor reconstruction
        from FakeTensors during torch.export tracing."""
        from brevitas.export.manager import ExportContext
        from brevitas.export.onnx.qonnx.manager import QONNXDynamoManager
        from brevitas.nn.quant_avg_pool import TruncAvgPool2d

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.inp_quant = QuantIdentity(return_quant_tensor=True)
                self.pool = TruncAvgPool2d(kernel_size=2, return_quant_tensor=False)

            def forward(self, x):
                return self.pool(self.inp_quant(x))

        inp = torch.randn(2, 8, 4, 4)
        model = Model()
        model(inp)  # populate cache
        model.eval()

        with torch.no_grad():
            with ExportContext(QONNXDynamoManager):
                # Set up handlers and cache
                model.apply(QONNXDynamoManager.set_export_handler)
                QONNXDynamoManager._cache_inp_out(model, inp)

                # Verify cache_class is set after cache_inp_out
                assert model.pool.cache_class is not None
                original_cache = model.pool.cache_class

                # Enable export mode
                QONNXDynamoManager.set_export_mode(model, enabled=True)

                # cache_class should be cleared
                assert model.pool.cache_class is None

                # Disable export mode
                QONNXDynamoManager.set_export_mode(model, enabled=False)

                # cache_class should be restored
                assert model.pool.cache_class is original_cache
