# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Adapted from https://github.com/huggingface/optimum-amd, released under the following LICENSE:

MIT License

Copyright (c) 2023 Hugging Face

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import random
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from brevitas_examples.llm.llm_quant.data_utils import recursive_to_device


def create_validation_dataloader(data, seqlen, device):
    nsamples = data['input_ids'].numel() // seqlen
    val_dataloader = []
    for i in tqdm(range(nsamples)):
        batch = data['input_ids'][:, (i * seqlen):((i + 1) * seqlen)].to(device)
        attention_mask = torch.ones_like(batch)
        val_dataloader.append({'input_ids': batch, 'attention_mask': attention_mask})
    return val_dataloader


@dataclass(frozen=True)
class TopKReferenceChunk:
    """Store original-model probabilities for one evaluation chunk."""

    token_ids: torch.Tensor
    probabilities: torch.Tensor


@dataclass(frozen=True)
class ReferenceProbabilityCache:
    """Store compact original-model probabilities for EAR and KLD evaluation."""

    chunks: List[TopKReferenceChunk]
    top_k: int
    num_positions: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """Store metrics and reference data from one evaluation pass."""

    ppl: Optional[float] = None
    ear: Optional[float] = None
    kld: Optional[float] = None
    reference_probabilities: Optional[ReferenceProbabilityCache] = None


def _set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def _move_subsample_to_model(model: torch.nn.Module, subsample: Dict[str, Any]) -> Dict[str, Any]:
    use_accelerate = hasattr(model, "hf_device_map")
    if not use_accelerate or not hasattr(model, "_hf_hook"):
        device = next(model.parameters()).device
    else:
        device = model._hf_hook.execution_device
    for name, value in subsample.items():
        subsample[name] = recursive_to_device(value, device)
    return subsample


def _get_logits(
        model: torch.nn.Module,
        data: Iterable[Dict],
        context_length: int,
        tokenizer: Any,
        seed: int = 0,
        dtype: torch.dtype = torch.float32) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield scored logits and labels from the model."""

    _set_eval_seed(seed)
    model = model.eval()
    for sample in tqdm(data, desc="Computing logits..."):
        sample_length = sample["input_ids"].shape[1]
        for start_index in range(0, sample_length, context_length * 2):
            end_index = min(start_index + sample_length, sample_length - 1)
            subsample = {
                key: value[:, start_index:end_index + 1] for (key, value) in sample.items()}

            # FX models require the traced cache input.
            if "past_key_values" in sample and isinstance(model, torch.fx.GraphModule):
                subsample["past_key_values"] = sample["past_key_values"]

            subsample = _move_subsample_to_model(model, subsample)
            logits = model(**subsample)["logits"]
            yield logits[:, context_length - 1:-1].to(dtype), \
                subsample["input_ids"][:, context_length:]


class MetricBase(ABC):

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    @abstractmethod
    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric state with one evaluation chunk."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> float:
        """Return the final metric value."""
        raise NotImplementedError


class Perplexity(MetricBase):

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        self.nlls = []

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        self.nlls.append(
            nn.functional.cross_entropy(output.reshape(-1, output.shape[-1]), target.reshape(-1)))

    def finalize(self) -> float:
        nlls = torch.stack(self.nlls).to(dtype=self.dtype)
        return torch.exp(nlls.mean()).item()


class EAR(MetricBase):
    """
    Expected Acceptance Rate (EAR) as proposed in https://arxiv.org/pdf/2605.02404
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        self.ear_sum = 0.0
        self.num_positions = 0

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        output = output.to(dtype=self.dtype)
        target = target.to(dtype=self.dtype)
        self.ear_sum += torch.minimum(output, target).sum().item()
        self.num_positions += target.numel() // target.shape[-1]

    def finalize(self) -> float:
        return self.ear_sum / self.num_positions


class KLD(MetricBase):

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        self.kld_sum = 0.0
        self.num_positions = 0

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        output = output.to(dtype=self.dtype)
        target = target.to(dtype=self.dtype)
        self.kld_sum += (target * (target.log() - output.log())).sum().item()
        self.num_positions += target.numel() // target.shape[-1]

    def finalize(self) -> float:
        return self.kld_sum / self.num_positions


@torch.no_grad()
def compute_float_evaluation_metrics(
        model: torch.nn.Module,
        data: Iterable[Dict],
        context_length: int,
        tokenizer: Any,
        top_k: int = 10,
        seed: int = 0,
        dtype: torch.dtype = torch.float32) -> EvaluationMetrics:
    """Compute float PPL and cache top-K probabilities."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    ppl = Perplexity(dtype=dtype)
    reference_chunks = []
    num_positions = 0
    for logits, labels in _get_logits(
            model, data, context_length, tokenizer, seed=seed, dtype=dtype):
        if top_k > logits.shape[-1]:
            raise ValueError(f"top_k ({top_k}) exceeds the vocabulary size ({logits.shape[-1]}).")
        ppl.update(logits, labels)
        top_logits, top_ids = logits.topk(top_k, dim=-1)
        top_probabilities = torch.exp(top_logits - logits.logsumexp(dim=-1, keepdim=True))
        reference_chunks.append(
            TopKReferenceChunk(
                token_ids=top_ids.to(device="cpu", dtype=torch.int32),
                probabilities=top_probabilities.to(device="cpu", dtype=torch.float32)))
        num_positions += top_ids.numel() // top_k

    return EvaluationMetrics(
        ppl=ppl.finalize(),
        reference_probabilities=ReferenceProbabilityCache(
            chunks=reference_chunks, top_k=top_k, num_positions=num_positions))


@torch.no_grad()
def compute_quantized_evaluation_metrics(
        model: torch.nn.Module,
        data: Iterable[Dict],
        context_length: int,
        tokenizer: Any,
        reference_probabilities: ReferenceProbabilityCache,
        normalize: bool = True,
        seed: int = 0,
        dtype: torch.dtype = torch.float32) -> EvaluationMetrics:
    """Compute quantized PPL, EAR, and top-K KLD.

    By default, normalize both distributions by the reference top-K probability mass.
    A quantized model that matches the reference top-K probabilities then gets
    an EAR of 1.0. Set normalize to False to use unnormalized EAR and KLD.
    """

    ppl = Perplexity(dtype=dtype)
    ear = EAR(dtype=dtype)
    kld = KLD(dtype=dtype)
    num_positions = 0
    chunk_index = 0
    for logits, labels in _get_logits(
            model, data, context_length, tokenizer, seed=seed, dtype=dtype):
        ppl.update(logits, labels)
        if chunk_index >= len(reference_probabilities.chunks):
            raise ValueError(
                "The evaluation data has more chunks than the reference probability cache.")
        reference_chunk = reference_probabilities.chunks[chunk_index]
        expected_shape = (*logits.shape[:-1], reference_probabilities.top_k)
        if reference_chunk.token_ids.shape != expected_shape or \
                reference_chunk.probabilities.shape != expected_shape:
            raise ValueError("The reference probability cache does not match the evaluation data.")

        token_ids = reference_chunk.token_ids.to(device=logits.device, dtype=torch.int64)
        reference_p = reference_chunk.probabilities.to(device=logits.device)
        quantized_log_q = torch.log_softmax(
            logits.to(dtype=torch.float32), dim=-1).gather(-1, token_ids)
        quantized_q = quantized_log_q.exp()
        # Use full-softmax probabilities on the reference model's top-K support.
        # Normalize EAR and KLD by the reference top-K mass when requested.
        if normalize:
            reference_mass = reference_p.sum(dim=-1, keepdim=True)
            reference_p = reference_p / reference_mass
            quantized_q = quantized_q / reference_mass
        ear.update(quantized_q, reference_p)
        kld.update(quantized_q, reference_p)
        num_positions += reference_p.numel() // reference_probabilities.top_k
        chunk_index += 1

    if chunk_index != len(reference_probabilities.chunks):
        raise ValueError(
            "The evaluation data has fewer chunks than the reference probability cache.")
    if num_positions != reference_probabilities.num_positions:
        raise ValueError(
            "The reference probability cache has a different number of token positions.")

    return EvaluationMetrics(ppl=ppl.finalize(), ear=ear.finalize(), kld=kld.finalize())
