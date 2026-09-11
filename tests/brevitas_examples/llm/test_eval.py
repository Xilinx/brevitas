# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch import nn

from brevitas_examples.llm.llm_quant.eval import compute_float_evaluation_metrics
from brevitas_examples.llm.llm_quant.eval import compute_quantized_evaluation_metrics
from brevitas_examples.llm.llm_quant.eval import ProbabilityCache


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


@pytest.fixture(params=[1, 2, 3])
def top_k(request):
    return request.param


class TestEvaluation:

    reference_logits = [2.0, 1.0, 0.0]
    quantized_logits = [1.5, 0.5, 1.0]
    input_ids = torch.tensor([[0, 1, 2, 0]])

    @property
    def data(self):
        return [{"input_ids": self.input_ids}]

    @pytest.mark.parametrize("normalize", [True, False])
    def test_expected_acceptance_rate_with_and_without_normalization(
            self, top_k: int, normalize: bool):
        reference_model = FixedLogitsModel(self.reference_logits)
        quantized_model = FixedLogitsModel(self.quantized_logits)

        reference_metrics = compute_float_evaluation_metrics(
            model=reference_model, data=self.data, context_length=2, tokenizer=None, top_k=top_k)
        reference_cache = reference_metrics.probabilities
        quantized_metrics = compute_quantized_evaluation_metrics(
            model=quantized_model,
            data=self.data,
            context_length=2,
            tokenizer=None,
            reference_probabilities=reference_cache,
            normalize=normalize)

        reference_p = torch.softmax(reference_model.logits, dim=-1)
        quantized_q = torch.softmax(quantized_model.logits, dim=-1)
        reference_top_k = reference_p.topk(top_k).indices
        expected_ear = torch.minimum(reference_p[reference_top_k],
                                     quantized_q[reference_top_k]).sum().item()
        expected_kld = (
            reference_p[reference_top_k] *
            (reference_p[reference_top_k].log() - quantized_q[reference_top_k].log())).sum().item()
        if normalize:
            reference_mass = reference_p[reference_top_k].sum().item()
            expected_ear /= reference_mass
            expected_kld /= reference_mass

        assert quantized_metrics.ear == pytest.approx(expected_ear)
        assert quantized_metrics.kld == pytest.approx(expected_kld)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_identical_models_with_and_without_normalization(self, top_k, normalize):
        model = FixedLogitsModel(self.reference_logits)
        reference_metrics = compute_float_evaluation_metrics(
            model=model, data=self.data, context_length=2, tokenizer=None, top_k=top_k)
        reference_cache = reference_metrics.probabilities

        quantized_metrics = compute_quantized_evaluation_metrics(
            model=model,
            data=self.data,
            context_length=2,
            tokenizer=None,
            reference_probabilities=reference_cache,
            normalize=normalize)

        if top_k < len(self.reference_logits) and not normalize:
            expected = torch.softmax(model.logits, dim=-1).topk(top_k).values.sum().item()
            assert quantized_metrics.ear == pytest.approx(expected)
            assert quantized_metrics.ear < 1.0
            return

        assert quantized_metrics.ear == pytest.approx(1.0)

    def test_float_evaluation_computes_metrics_and_caches_probabilities(self, top_k):
        model = FixedLogitsModel(self.reference_logits)

        metrics = compute_float_evaluation_metrics(
            model=model, data=self.data, context_length=2, tokenizer=None, top_k=top_k)

        assert metrics.ppl is not None
        assert metrics.probabilities is not None
        assert metrics.probabilities.chunks[0].token_ids.dtype == torch.int32
        assert metrics.probabilities.chunks[0].token_ids.device.type == "cpu"
        assert metrics.probabilities.chunks[0].probabilities.dtype == torch.float32
        assert metrics.probabilities.chunks[0].probabilities.device.type == "cpu"
        # Verify that one model call computes all metrics for the data chunk.
        assert model.forward_count == 1

    def test_expected_acceptance_rate_rejects_cache_with_fewer_chunks(self):
        model = FixedLogitsModel(self.reference_logits)
        empty_cache = ProbabilityCache(chunks=[])

        with pytest.raises(AssertionError, match="same length"):
            compute_quantized_evaluation_metrics(
                model=model,
                data=self.data,
                context_length=2,
                tokenizer=None,
                reference_probabilities=empty_cache)
