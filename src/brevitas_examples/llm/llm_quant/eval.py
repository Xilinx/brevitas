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
from typing import List
from typing import Optional

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

    perplexity: Optional[float] = None
    expected_acceptance_rate: Optional[float] = None
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


def _iter_eval_subsamples(
        model: torch.nn.Module, data: Iterable[Dict], context_length: int, description: str):
    for sample in tqdm(data, desc=description):
        sample_length = sample["input_ids"].shape[1]
        for start_index in range(0, sample_length, context_length * 2):
            end_index = min(start_index + sample_length, sample_length - 1)
            subsample = {
                key: value[:, start_index:end_index + 1] for (key, value) in sample.items()}

            # FX models require the traced cache input.
            if "past_key_values" in sample and isinstance(model, torch.fx.GraphModule):
                subsample["past_key_values"] = sample["past_key_values"]

            yield _move_subsample_to_model(model, subsample)


@torch.no_grad()
def compute_evaluation_metrics(
        model: torch.nn.Module,
        data: Iterable[Dict],
        context_length: int,
        tokenizer: Any,
        compute_perplexity: bool = True,
        build_reference_probabilities: bool = False,
        reference_probabilities: Optional[ReferenceProbabilityCache] = None,
        top_k: int = 10,
        seed: int = 0,
        dtype: torch.dtype = torch.float32) -> EvaluationMetrics:
    """Compute requested metrics from one model forward pass per evaluation chunk."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if build_reference_probabilities and reference_probabilities is not None:
        raise ValueError("Cannot build and consume reference probabilities in the same pass.")
    if not compute_perplexity and not build_reference_probabilities and \
            reference_probabilities is None:
        raise ValueError("Select at least one evaluation metric.")

    _set_eval_seed(seed)

    model = model.eval()
    cross_entropy_loss = nn.CrossEntropyLoss()
    nlls = []
    reference_chunks = []
    reference_position_count = 0
    overlap_sum = 0.0
    kld_sum = 0.0
    overlap_position_count = 0
    reference_chunk_index = 0
    description = "Computing evaluation metrics..."
    for subsample in _iter_eval_subsamples(model, data, context_length, description):
        lm_logits = model(**subsample)["logits"]
        reference_labels = subsample["input_ids"][:, context_length:]
        shift_logits = lm_logits[:, context_length - 1:-1]
        shift_logits = shift_logits.to(dtype)

        if compute_perplexity:
            loss = cross_entropy_loss(
                shift_logits.reshape(-1, shift_logits.shape[-1]), reference_labels.reshape(-1))
            nlls.append(loss)

        if build_reference_probabilities:
            if top_k > shift_logits.shape[-1]:
                raise ValueError(
                    f"top_k ({top_k}) exceeds the vocabulary size "
                    f"({shift_logits.shape[-1]}).")
            top_logits, top_ids = shift_logits.topk(top_k, dim=-1)
            log_normalizer = shift_logits.logsumexp(dim=-1, keepdim=True)
            top_probabilities = torch.exp(top_logits - log_normalizer)
            reference_chunks.append(
                TopKReferenceChunk(
                    token_ids=top_ids.to(device="cpu", dtype=torch.int32),
                    probabilities=top_probabilities.to(device="cpu", dtype=torch.float32)))
            reference_position_count += top_ids.numel() // top_k

        if reference_probabilities is not None:
            if reference_chunk_index >= len(reference_probabilities.chunks):
                raise ValueError(
                    "The evaluation data has more chunks than the reference probability cache.")
            reference_chunk = reference_probabilities.chunks[reference_chunk_index]
            expected_shape = (*shift_logits.shape[:-1], reference_probabilities.top_k)
            if reference_chunk.token_ids.shape != expected_shape or \
                    reference_chunk.probabilities.shape != expected_shape:
                raise ValueError(
                    "The reference probability cache does not match the evaluation data.")

            token_ids = reference_chunk.token_ids.to(device=shift_logits.device, dtype=torch.int64)
            reference_p = reference_chunk.probabilities.to(device=shift_logits.device)
            quantized_log_q = torch.log_softmax(
                shift_logits, dim=-1).gather(
                    dim=-1, index=token_ids)
            quantized_q = quantized_log_q.exp()
            # We use unnormalized full-softmax probabilities on p's top-K support,
            # following https://arxiv.org/abs/2605.02404; see Section 3.2.
            overlap_sum += torch.minimum(reference_p, quantized_q).double().sum().item()
            kld_sum += (reference_p * (reference_p.log() - quantized_log_q)).double().sum().item()
            overlap_position_count += reference_p.numel() // reference_probabilities.top_k
            reference_chunk_index += 1

    if reference_probabilities is not None:
        if reference_chunk_index != len(reference_probabilities.chunks):
            raise ValueError(
                "The evaluation data has fewer chunks than the reference probability cache.")
        if overlap_position_count != reference_probabilities.num_positions:
            raise ValueError(
                "The reference probability cache has a different number of token positions.")

    perplexity = torch.exp(torch.stack(nlls).mean()).item() if compute_perplexity else None
    expected_acceptance_rate = (
        overlap_sum / overlap_position_count if reference_probabilities is not None else None)
    kld = kld_sum / overlap_position_count if reference_probabilities is not None else None
    reference_cache = None
    if build_reference_probabilities:
        reference_cache = ReferenceProbabilityCache(
            chunks=reference_chunks, top_k=top_k, num_positions=reference_position_count)

    return EvaluationMetrics(
        perplexity=perplexity,
        expected_acceptance_rate=expected_acceptance_rate,
        kld=kld,
        reference_probabilities=reference_cache)


@torch.no_grad()
def compute_perplexity(
        model: torch.nn.Module,
        data: List[Dict],
        context_length: int,
        tokenizer: Any,
        seed: int = 0,
        dtype: torch.dtype = torch.float32):
    """Compute perplexity over the scored token positions."""

    return compute_evaluation_metrics(
        model=model,
        data=data,
        context_length=context_length,
        tokenizer=tokenizer,
        seed=seed,
        dtype=dtype).perplexity


@torch.no_grad()
def compute_reference_probabilities(
        model: torch.nn.Module,
        data: List[Dict],
        context_length: int,
        tokenizer: Any,
        top_k: int = 10,
        seed: int = 0,
        dtype: torch.dtype = torch.float32) -> ReferenceProbabilityCache:
    """Cache original-model top-K probabilities for EAR evaluation."""

    metrics = compute_evaluation_metrics(
        model=model,
        data=data,
        context_length=context_length,
        tokenizer=tokenizer,
        compute_perplexity=False,
        build_reference_probabilities=True,
        top_k=top_k,
        seed=seed,
        dtype=dtype)
    return metrics.reference_probabilities


@torch.no_grad()
def compute_expected_acceptance_rate(
        model: torch.nn.Module,
        data: List[Dict],
        context_length: int,
        tokenizer: Any,
        reference_probabilities: ReferenceProbabilityCache,
        seed: int = 0,
        dtype: torch.dtype = torch.float32) -> float:
    """Compute EAR against cached original-model probabilities."""

    metrics = compute_evaluation_metrics(
        model=model,
        data=data,
        context_length=context_length,
        tokenizer=tokenizer,
        compute_perplexity=False,
        reference_probabilities=reference_probabilities,
        seed=seed,
        dtype=dtype)
    return metrics.expected_acceptance_rate
