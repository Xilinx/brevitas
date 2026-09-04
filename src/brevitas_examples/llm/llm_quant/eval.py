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
            lm_logits = model(**subsample)["logits"]
            yield lm_logits[:, context_length - 1:-1].to(dtype), \
                subsample["input_ids"][:, context_length:]


def compute_perplexity(
        logits: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        dtype: torch.dtype = torch.float32) -> float:
    """Compute perplexity from scored logits and labels."""

    cross_entropy_loss = nn.CrossEntropyLoss()
    nlls = []
    for scored_logits, labels in logits:
        scored_logits = scored_logits.to(dtype)
        nlls.append(
            cross_entropy_loss(
                scored_logits.reshape(-1, scored_logits.shape[-1]), labels.reshape(-1)))
    return torch.exp(torch.stack(nlls).mean()).item()


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

    nlls = []
    reference_chunks = []
    num_positions = 0
    for scored_logits, labels in _get_logits(
            model, data, context_length, tokenizer, seed=seed, dtype=dtype):
        if top_k > scored_logits.shape[-1]:
            raise ValueError(
                f"top_k ({top_k}) exceeds the vocabulary size ({scored_logits.shape[-1]}).")
        nlls.append(
            nn.functional.cross_entropy(
                scored_logits.reshape(-1, scored_logits.shape[-1]), labels.reshape(-1)))
        top_logits, top_ids = scored_logits.topk(top_k, dim=-1)
        top_probabilities = torch.exp(top_logits - scored_logits.logsumexp(dim=-1, keepdim=True))
        reference_chunks.append(
            TopKReferenceChunk(
                token_ids=top_ids.to(device="cpu", dtype=torch.int32),
                probabilities=top_probabilities.to(device="cpu", dtype=torch.float32)))
        num_positions += top_ids.numel() // top_k

    return EvaluationMetrics(
        ppl=torch.exp(torch.stack(nlls).mean()).item(),
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

    nlls = []
    ear_sum = 0.0
    kld_sum = 0.0
    num_positions = 0
    chunk_index = 0
    for scored_logits, labels in _get_logits(
            model, data, context_length, tokenizer, seed=seed, dtype=dtype):
        if chunk_index >= len(reference_probabilities.chunks):
            raise ValueError(
                "The evaluation data has more chunks than the reference probability cache.")
        reference_chunk = reference_probabilities.chunks[chunk_index]
        expected_shape = (*scored_logits.shape[:-1], reference_probabilities.top_k)
        if reference_chunk.token_ids.shape != expected_shape or \
                reference_chunk.probabilities.shape != expected_shape:
            raise ValueError("The reference probability cache does not match the evaluation data.")

        nlls.append(
            nn.functional.cross_entropy(
                scored_logits.reshape(-1, scored_logits.shape[-1]), labels.reshape(-1)))
        token_ids = reference_chunk.token_ids.to(device=scored_logits.device, dtype=torch.int64)
        reference_p = reference_chunk.probabilities.to(device=scored_logits.device)
        quantized_log_q = torch.log_softmax(scored_logits, dim=-1).gather(-1, token_ids)
        quantized_q = quantized_log_q.exp()
        # Use full-softmax probabilities on the reference model's top-K support.
        # Normalize EAR and KLD by the reference top-K mass when requested.
        if normalize:
            reference_mass = reference_p.sum(dim=-1, keepdim=True)
            reference_p = reference_p / reference_mass
            quantized_q = quantized_q / reference_mass
            quantized_log_q = quantized_log_q - reference_mass.log()
        ear_sum += torch.minimum(reference_p, quantized_q).double().sum().item()
        kld_sum += (reference_p * (reference_p.log() - quantized_log_q)).double().sum().item()
        num_positions += reference_p.numel() // reference_probabilities.top_k
        chunk_index += 1

    if chunk_index != len(reference_probabilities.chunks):
        raise ValueError(
            "The evaluation data has fewer chunks than the reference probability cache.")
    if num_positions != reference_probabilities.num_positions:
        raise ValueError(
            "The reference probability cache has a different number of token positions.")

    return EvaluationMetrics(
        ppl=torch.exp(torch.stack(nlls).mean()).item(),
        ear=ear_sum / num_positions,
        kld=kld_sum / num_positions)
