# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Optional

import pytest
import torch
from torch import nn

from brevitas_examples.llm.llm_quant.eval import compute_float_evaluation_metrics
from brevitas_examples.llm.llm_quant.eval import compute_quantized_evaluation_metrics


def standard_error(values: torch.Tensor) -> Optional[float]:
    """Reference implementation of the per-token standard error estimator.

    Moments are accumulated in double, matching ``MetricBase``.
    """
    values = values.to(dtype=torch.float64).flatten()
    count = values.numel()
    if count < 2:
        return None
    mean = (values.sum() / count).item()
    variance = (values.square().sum() / count).item() - mean ** 2
    if variance <= 0:
        return None
    return math.sqrt(variance / (count - 1))


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
        assert metrics.probabilities[0].token_ids.dtype == torch.int32
        assert metrics.probabilities[0].token_ids.device.type == "cpu"
        assert metrics.probabilities[0].probabilities.dtype == torch.float32
        assert metrics.probabilities[0].probabilities.device.type == "cpu"
        # Verify that one model call computes all metrics for the data chunk.
        assert model.forward_count == 1

    def test_expected_acceptance_rate_rejects_cache_with_fewer_chunks(self):
        model = FixedLogitsModel(self.reference_logits)

        with pytest.raises(AssertionError, match="same length"):
            compute_quantized_evaluation_metrics(
                model=model,
                data=self.data,
                context_length=2,
                tokenizer=None,
                reference_probabilities=[])


class PositionLogitsModel(nn.Module):
    """Emit a different logit vector at every position."""

    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32), requires_grad=False)

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape
        assert sequence_length <= self.logits.shape[0]
        return {"logits": self.logits[:sequence_length].unsqueeze(0).expand(batch_size, -1, -1)}


class TestStandardError:
    """The standard errors are derived from per-token moments accumulated in double.

    Moments are accumulated per token, so a single chunk holding more than one
    token is already enough to define a standard error.
    """

    reference_logits = [[2.0, 1.0, 0.0], [0.5, 2.5, 1.0], [1.0, 0.0, 3.0], [2.0, 2.0, 0.5]]
    quantized_logits = [[1.5, 0.5, 1.0], [0.0, 2.0, 1.5], [1.5, 0.5, 2.5], [1.0, 2.5, 0.0]]
    input_ids = torch.tensor([[0, 1, 2, 0]])
    context_length = 2

    @property
    def data(self):
        return [{"input_ids": self.input_ids}]

    def _expected_nlls(self, logits) -> torch.Tensor:
        # _get_shifted_logits_and_labels yields logits[:, context_length - 1:-1]
        # against input_ids[:, context_length:].
        shifted_logits = torch.tensor(logits)[self.context_length - 1:-1]
        labels = self.input_ids[0, self.context_length:]
        return nn.functional.cross_entropy(shifted_logits, labels, reduction="none")

    def test_perplexity_standard_error(self):
        model = PositionLogitsModel(self.reference_logits)
        metrics = compute_float_evaluation_metrics(
            model=model,
            data=self.data,
            context_length=self.context_length,
            tokenizer=None,
            top_k=3)

        nlls = self._expected_nlls(self.reference_logits)
        expected_ppl = math.exp(nlls.to(dtype=torch.float64).mean().item())
        # The delta method reports ppl +/- sigma * ppl.
        expected_std = expected_ppl * standard_error(nlls)

        assert metrics.ppl == pytest.approx(expected_ppl)
        assert metrics.ppl_std == pytest.approx(expected_std)

    def test_perplexity_is_token_weighted_but_unchanged_for_equal_chunks(self):
        # All chunks carry the same number of tokens, so the token-weighted mean
        # agrees with the mean of the per-chunk means.
        model = PositionLogitsModel(self.reference_logits)
        two_chunks = [{"input_ids": self.input_ids}, {"input_ids": self.input_ids}]
        metrics = compute_float_evaluation_metrics(
            model=model,
            data=two_chunks,
            context_length=self.context_length,
            tokenizer=None,
            top_k=3)

        nlls = self._expected_nlls(self.reference_logits).to(dtype=torch.float64)
        chunk_means = torch.stack([nlls.mean(), nlls.mean()])
        assert metrics.ppl == pytest.approx(math.exp(chunk_means.mean().item()))
        # Four tokens across two chunks, rather than two chunk means.
        expected_std = metrics.ppl * standard_error(torch.cat([nlls, nlls]))
        assert metrics.ppl_std == pytest.approx(expected_std)

    def test_ear_and_kld_standard_error(self):
        reference_model = PositionLogitsModel(self.reference_logits)
        quantized_model = PositionLogitsModel(self.quantized_logits)
        top_k = 3

        reference_metrics = compute_float_evaluation_metrics(
            model=reference_model,
            data=self.data,
            context_length=self.context_length,
            tokenizer=None,
            top_k=top_k)
        quantized_metrics = compute_quantized_evaluation_metrics(
            model=quantized_model,
            data=self.data,
            context_length=self.context_length,
            tokenizer=None,
            reference_probabilities=reference_metrics.probabilities,
            normalize=False)

        # Reuse the cached reference probabilities rather than recomputing them, so
        # that the assertions below isolate the standard error estimator.
        reference_chunk = reference_metrics.probabilities[0]
        reference_p = reference_chunk.probabilities.to(dtype=torch.float32)
        token_ids = reference_chunk.token_ids.to(dtype=torch.int64)
        quantized_logits = torch.tensor(self.quantized_logits)[self.context_length -
                                                               1:-1].unsqueeze(0)
        quantized_q = torch.softmax(quantized_logits, dim=-1).gather(-1, token_ids)

        per_token_ear = torch.minimum(quantized_q, reference_p).sum(dim=-1)
        per_token_kld = (reference_p * (reference_p.log() - quantized_q.log())).sum(dim=-1)

        assert quantized_metrics.ear == pytest.approx(per_token_ear.mean().item())
        assert quantized_metrics.ear_std == pytest.approx(standard_error(per_token_ear))
        assert quantized_metrics.kld == pytest.approx(per_token_kld.mean().item())
        assert quantized_metrics.kld_std == pytest.approx(standard_error(per_token_kld))

    def test_standard_error_is_none_for_a_single_token(self):
        model = PositionLogitsModel(self.reference_logits)
        # Three tokens with context_length=2 leave a single token to score.
        single_token = [{"input_ids": self.input_ids[:, :3]}]
        metrics = compute_float_evaluation_metrics(
            model=model,
            data=single_token,
            context_length=self.context_length,
            tokenizer=None,
            top_k=3)

        assert metrics.ppl is not None
        assert metrics.ppl_std is None

    def test_standard_error_is_none_for_zero_variance(self):
        # Position-independent logits give every token the same EAR and KLD, so
        # their variance is zero and no standard error is defined.
        reference_model = FixedLogitsModel([2.0, 1.0, 0.0])
        quantized_model = FixedLogitsModel([1.5, 0.5, 1.0])
        reference_metrics = compute_float_evaluation_metrics(
            model=reference_model,
            data=self.data,
            context_length=self.context_length,
            tokenizer=None,
            top_k=3)
        quantized_metrics = compute_quantized_evaluation_metrics(
            model=quantized_model,
            data=self.data,
            context_length=self.context_length,
            tokenizer=None,
            reference_probabilities=reference_metrics.probabilities)

        assert quantized_metrics.ear_std is None
        assert quantized_metrics.kld_std is None
        # The two scored tokens have different targets, so the NLLs do differ.
        assert quantized_metrics.ppl_std is not None
