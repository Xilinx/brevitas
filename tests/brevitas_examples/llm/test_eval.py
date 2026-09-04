# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch import nn

from brevitas_examples.llm.llm_quant.eval import compute_float_evaluation_metrics
from brevitas_examples.llm.llm_quant.eval import compute_quantized_evaluation_metrics
from brevitas_examples.llm.llm_quant.eval import ReferenceProbabilityCache


class FixedLogitsModel(nn.Module):

    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32), requires_grad=False)
        self.forward_count = 0

    def forward(self, input_ids):
        self.forward_count += 1
        batch_size, sequence_length = input_ids.shape
        logits = self.logits.view(1, 1, -1).expand(batch_size, sequence_length, -1)
        return {"logits": logits}


def test_expected_acceptance_rate_uses_original_top_k_without_renormalization():
    reference_model = FixedLogitsModel([2.0, 1.0, 0.0])
    quantized_model = FixedLogitsModel([1.5, 0.5, 1.0])
    data = [{"input_ids": torch.tensor([[0, 1, 2, 0]])}]

    reference_metrics = compute_float_evaluation_metrics(
        model=reference_model, data=data, context_length=2, tokenizer=None, top_k=2)
    reference_cache = reference_metrics.reference_probabilities
    quantized_metrics = compute_quantized_evaluation_metrics(
        model=quantized_model,
        data=data,
        context_length=2,
        tokenizer=None,
        reference_probabilities=reference_cache,
        normalize=False)

    reference_p = torch.softmax(reference_model.logits, dim=-1)
    quantized_q = torch.softmax(quantized_model.logits, dim=-1)
    reference_top_k = reference_p.topk(2).indices
    expected_ear = torch.minimum(reference_p[reference_top_k],
                                 quantized_q[reference_top_k]).sum().item()
    expected_kld = (
        reference_p[reference_top_k] *
        (reference_p[reference_top_k].log() - quantized_q[reference_top_k].log())).sum().item()

    assert quantized_metrics.ear == pytest.approx(expected_ear)
    assert quantized_metrics.kld == pytest.approx(expected_kld)
    assert reference_cache.top_k == 2
    assert reference_cache.num_positions == 2
    assert reference_cache.chunks[0].token_ids.dtype == torch.int32
    assert reference_cache.chunks[0].token_ids.device.type == "cpu"
    assert reference_cache.chunks[0].probabilities.dtype == torch.float32
    assert reference_cache.chunks[0].probabilities.device.type == "cpu"


def test_identical_models_return_reference_top_k_mass():
    model = FixedLogitsModel([2.0, 1.0, 0.0])
    data = [{"input_ids": torch.tensor([[0, 1, 2, 0]])}]
    reference_cache = compute_float_evaluation_metrics(
        model=model, data=data, context_length=2, tokenizer=None, top_k=2)
    reference_cache = reference_cache.reference_probabilities

    quantized_metrics = compute_quantized_evaluation_metrics(
        model=model,
        data=data,
        context_length=2,
        tokenizer=None,
        reference_probabilities=reference_cache,
        normalize=False)
    expected = torch.softmax(model.logits, dim=-1).topk(2).values.sum().item()

    assert quantized_metrics.ear == pytest.approx(expected)
    assert quantized_metrics.ear < 1.0


def test_identical_models_normalized_ear_reaches_full_mass():
    model = FixedLogitsModel([2.0, 1.0, 0.0])
    data = [{"input_ids": torch.tensor([[0, 1, 2, 0]])}]
    reference_metrics = compute_float_evaluation_metrics(
        model=model, data=data, context_length=2, tokenizer=None, top_k=2)
    reference_cache = reference_metrics.reference_probabilities

    quantized_metrics = compute_quantized_evaluation_metrics(
        model=model,
        data=data,
        context_length=2,
        tokenizer=None,
        reference_probabilities=reference_cache,
        normalize=True)

    # Perfect top-K alignment reaches a normalized EAR of 1.0.
    # The same reference-mass normalization applies to KLD.
    assert quantized_metrics.ear == pytest.approx(1.0)


def test_normalized_ear_scales_by_reference_top_k_mass():
    reference_model = FixedLogitsModel([2.0, 1.0, 0.0])
    quantized_model = FixedLogitsModel([1.5, 0.5, 1.0])
    data = [{"input_ids": torch.tensor([[0, 1, 2, 0]])}]

    reference_metrics = compute_float_evaluation_metrics(
        model=reference_model, data=data, context_length=2, tokenizer=None, top_k=2)
    reference_cache = reference_metrics.reference_probabilities
    unnormalized_metrics = compute_quantized_evaluation_metrics(
        model=quantized_model,
        data=data,
        context_length=2,
        tokenizer=None,
        reference_probabilities=reference_cache,
        normalize=False)
    normalized_metrics = compute_quantized_evaluation_metrics(
        model=quantized_model,
        data=data,
        context_length=2,
        tokenizer=None,
        reference_probabilities=reference_cache,
        normalize=True)

    reference_p = torch.softmax(reference_model.logits, dim=-1)
    reference_top_k = reference_p.topk(2).indices
    reference_mass = reference_p[reference_top_k].sum().item()

    assert normalized_metrics.ear == pytest.approx(unnormalized_metrics.ear / reference_mass)
    assert normalized_metrics.kld == pytest.approx(unnormalized_metrics.kld / reference_mass)


def test_combined_evaluation_uses_one_forward_per_chunk():
    model = FixedLogitsModel([2.0, 1.0, 0.0])
    data = [{"input_ids": torch.tensor([[0, 1, 2, 0]])}]

    metrics = compute_float_evaluation_metrics(
        model=model, data=data, context_length=2, tokenizer=None, top_k=2)

    assert model.forward_count == 1
    assert metrics.ppl is not None
    assert metrics.reference_probabilities is not None


def test_expected_acceptance_rate_rejects_cache_with_fewer_chunks():
    model = FixedLogitsModel([2.0, 1.0, 0.0])
    data = [{"input_ids": torch.tensor([[0, 1, 2, 0]])}]
    empty_cache = ReferenceProbabilityCache(chunks=[], top_k=2, num_positions=0)

    with pytest.raises(ValueError, match="more chunks"):
        compute_quantized_evaluation_metrics(
            model=model,
            data=data,
            context_length=2,
            tokenizer=None,
            reference_probabilities=empty_cache)
