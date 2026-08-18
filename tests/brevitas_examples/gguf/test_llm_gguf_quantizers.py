# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from gguf import GGML_QUANT_SIZES
import gguf.quants as gguf_quants
import numpy as np
import pytest
import pytest_cases
import torch

from brevitas.core.zero_point import StatsFromParameterZeroPoint
import brevitas.nn as qnn
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ2_KWeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ3_KWeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ4_0WeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ4_1WeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ4_KWeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ5_KWeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ6_KWeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ8_0WeightQuant
import brevitas_examples.llm.gguf_export.custom_quantizers  # noqa: F401
from brevitas_examples.llm.gguf_export.export import _enable_gguf_export_caching
from brevitas_examples.llm.gguf_export.export import _restore_gguf_export_caching
from brevitas_examples.llm.gguf_export.quant import ggml_quant

from .common import *

# StatsFromParameterZeroPoint accepts scale_shift_zero_point_impl.
# ParameterFromStatsFromParameterZeroPoint does not accept that argument until PR #1585.
_weight_quant_kwargs = {'zero_point_impl': StatsFromParameterZeroPoint}


def _packed_row_size(in_features, qtype):
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    return in_features // block_size * type_size


def _custom_export(weight: np.ndarray, weight_quant, qtype) -> np.ndarray:
    """Quantize ``weight`` with a Brevitas custom quantizer and pack it to a
    GGUF block as ``convert.ModelBase.quantize`` does.

    Returns the packed uint8 block. Shape is one packed row per weight row.
    """
    out_features, in_features = weight.shape
    layer = qnn.QuantLinear(in_features, out_features, bias=False, weight_quant=weight_quant)
    layer.weight.data = torch.from_numpy(weight.copy())
    # Initialize the scale/zero-point first. Caching before this step raises RuntimeError.
    layer.quant_weight()
    prior = _enable_gguf_export_caching(layer)
    try:
        # A second call populates the metadata-only cache.
        quant_weight = layer.quant_weight()
        wq = layer.weight_quant
        # Pass ndarray codes so ggml_quant does not squeeze a singleton out-features dim.
        block = ggml_quant(
            quant_weight.int().detach().cpu().numpy(),
            qtype,
            scale=wq.scale,
            zero_point=wq.zero_point,
            scale_of_scale=wq.scale_of_scale,
            scale_of_zero_point=wq.scale_of_zero_point)
    finally:
        _restore_gguf_export_caching(prior)
    return block.reshape(out_features, _packed_row_size(in_features, qtype))


def _select_weight_quant(weight_quant, module, name):
    """Return the per-module weight quantizer.

    Uniform recipes store the injector class on ``weight_quant``. Mixed
    recipes store a lambda that selects first/last vs body.
    """
    if isinstance(weight_quant, type):
        return weight_quant
    return weight_quant(module, name)


@pytest.mark.llm
class _CustomQuantTests:
    """Shared checks for each GGUF quantizer in ``custom_quantizers.py``.

    The packed GGUF block must have the on-disk layout. Decoding it must
    match Brevitas reconstruction, up to fp16 storage of super-block scales.
    """
    weight_quant = None
    qtype = None

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_block_layout(self, x):
        block = _custom_export(x, self.weight_quant, self.qtype)
        assert block.dtype == np.uint8
        assert block.shape == (x.shape[0], _packed_row_size(x.shape[1], self.qtype))

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_export_is_consistent_with_calibration(self, x):
        """Decoding the exported block reproduces Brevitas reconstruction, up to
        the fp16 rounding of the super-block d / dmin factors stored on disk."""
        layer = qnn.QuantLinear(x.shape[1], x.shape[0], bias=False, weight_quant=self.weight_quant)
        layer.weight.data = torch.from_numpy(x.copy())
        recon_brevitas = layer.quant_weight().value.detach().numpy()
        recon_gguf = gguf_quants.dequantize(
            _custom_export(x, self.weight_quant, self.qtype), self.qtype).reshape(x.shape)
        amax = np.abs(x).max()
        # fp16 super-scale storage introduces at most a small relative error.
        atol = 1e-2 * amax if amax > 0 else 1e-6
        np.testing.assert_allclose(recon_gguf, recon_brevitas, rtol=0, atol=atol)


class TestQ4_0Custom(_CustomQuantTests):
    weight_quant = GGUFQ4_0WeightQuant
    qtype = Q4_0


class TestQ4_1Custom(_CustomQuantTests):
    weight_quant = GGUFQ4_1WeightQuant
    qtype = Q4_1


class TestQ8_0Custom(_CustomQuantTests):
    weight_quant = GGUFQ8_0WeightQuant
    qtype = Q8_0


class TestQ2KCustom(_CustomQuantTests):
    weight_quant = GGUFQ2_KWeightQuant.let(**_weight_quant_kwargs)
    qtype = Q2_K


class TestQ3KCustom(_CustomQuantTests):
    weight_quant = GGUFQ3_KWeightQuant
    qtype = Q3_K


class TestQ4KCustom(_CustomQuantTests):
    weight_quant = GGUFQ4_KWeightQuant.let(**_weight_quant_kwargs)
    qtype = Q4_K


class TestQ5KCustom(_CustomQuantTests):
    weight_quant = GGUFQ5_KWeightQuant.let(**_weight_quant_kwargs)
    qtype = Q5_K


class TestQ6KCustom(_CustomQuantTests):
    weight_quant = GGUFQ6_KWeightQuant
    qtype = Q6_K


# ``custom_quantizers.py`` registers a recipe for each GGUF format. Low-bit
# recipes keep the first/last layer at Q6_K. Q6_K and Q8_0 are uniform.
_MOSTLY_UNIFORM_RECIPES = (
    ("gguf_q2_k", GGUFQ2_KWeightQuant, GGUFQ6_KWeightQuant),
    ("gguf_q3_k", GGUFQ3_KWeightQuant, GGUFQ6_KWeightQuant),
    ("gguf_q4_0", GGUFQ4_0WeightQuant, GGUFQ6_KWeightQuant),
    ("gguf_q4_1", GGUFQ4_1WeightQuant, GGUFQ6_KWeightQuant),
    ("gguf_q4_k", GGUFQ4_KWeightQuant, GGUFQ6_KWeightQuant),
    ("gguf_q5_k", GGUFQ5_KWeightQuant, GGUFQ6_KWeightQuant),
    ("gguf_q6_k", GGUFQ6_KWeightQuant, GGUFQ6_KWeightQuant),
    ("gguf_q8_0", GGUFQ8_0WeightQuant, GGUFQ8_0WeightQuant),
)


@pytest.mark.llm
@pytest.mark.parametrize(
    "key, body_quant, first_last_quant",
    _MOSTLY_UNIFORM_RECIPES,
    ids=[row[0] for row in _MOSTLY_UNIFORM_RECIPES])
def test_gguf_quant_registered(key, body_quant, first_last_quant):
    """Each gguf_* custom quantizer is registered with the llama.cpp first/last
    layer mapping. Uniform recipes store the injector class directly."""
    assert key in QUANTIZERS_REGISTRY.get_registered_keys()
    weight_quant = QUANTIZERS_REGISTRY.get(key).weight_quant
    embedding = torch.nn.Embedding(8, 8)
    linear = torch.nn.Linear(8, 8)
    assert _select_weight_quant(weight_quant, embedding, "model.embed_tokens") is first_last_quant
    assert _select_weight_quant(weight_quant, linear, "model.lm_head") is first_last_quant
    assert _select_weight_quant(weight_quant, linear, "model.layers.0.mlp.gate_proj") is body_quant
