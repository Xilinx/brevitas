"""
Adapted from https://github.com/IST-DASLab/gptq, released under the following LICENSE:

Copyright 2023 IST-DASLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def create_validation_dataloader(data, seqlen, device):
    nsamples = data['input_ids'].numel() // seqlen
    val_dataloader = []
    for i in tqdm(range(nsamples)):
        batch = data['input_ids'][:, (i * seqlen):((i + 1) * seqlen)].to(device)
        attention_mask = torch.ones_like(batch)
        val_dataloader.append({'input_ids': batch, 'attention_mask': attention_mask})
    return val_dataloader


@torch.no_grad()
def recursive_to_device(tensor_or_iterable: Union[Iterable, torch.Tensor], device) -> None:
    if isinstance(tensor_or_iterable, torch.Tensor):
        return tensor_or_iterable.to(device)
    elif isinstance(tensor_or_iterable,
                    tuple):  # Special handling of tuples, since they are immutable
        tmp_list = []
        for i in tensor_or_iterable:
            tmp_list.append(recursive_to_device(i, device))
        return tuple(tmp_list)
    elif isinstance(tensor_or_iterable, Iterable):
        for i in tensor_or_iterable:
            tensor_or_iterable[i] = recursive_to_device(i, device)
        return tensor_or_iterable
    else:
        raise ValueError(f"Cannot move {type(tensor_or_iterable)} to {device}")


@torch.no_grad()
def compute_perplexity(
        model: torch.nn.Module,
        data: List[Dict],
        context_length: int,
        tokenizer: Any,
        seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    model = model.eval()

    cross_entropy_loss = nn.CrossEntropyLoss()

    nlls = []
    for sample in tqdm(data, desc="Computing perplexity..."):
        sample_length = sample["input_ids"].shape[1]
        for start_index in range(0, sample_length, context_length * 2):
            end_index = min(start_index + sample_length, sample_length - 1)

            subsample = {
                "input_ids": sample["input_ids"][:, start_index:end_index + 1],
                "attention_mask": sample["attention_mask"][:, start_index:end_index + 1],}

            # In case we are using torch.fx, we can not have optional inputs, and we have traced the model with past_key_values inputs, thus we need them here as well.
            if "past_key_values" in sample and isinstance(model, torch.fx.GraphModule):
                subsample["past_key_values"] = sample["past_key_values"]

            # Add BOS token.
            if tokenizer.bos_token_id is not None:
                subsample["input_ids"][:, 0] = tokenizer.bos_token_id

            use_accelerate = hasattr(model, "hf_device_map")
            if not use_accelerate or (use_accelerate and not hasattr(model, "_hf_hook")):
                device = next(model.parameters()).device
                for name, val in subsample.items():
                    subsample[name] = recursive_to_device(val, device)
            else:
                # In accelerate by default `io_same_device=True`, and here we want the of the model output on device.
                device = model._hf_hook.execution_device
                for name, val in subsample.items():
                    subsample[name] = recursive_to_device(val, device)

            lm_logits = model(**subsample)["logits"]

            reference_labels = subsample["input_ids"][:, context_length:]

            shift_logits = lm_logits[:, context_length - 1:-1]

            # Fuse batch and sequence length dimensions.
            reference_labels = reference_labels.view(reference_labels.shape[-1])
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])

            loss = cross_entropy_loss(shift_logits, reference_labels)

            nlls.append(loss)

    ppl = torch.exp(torch.stack(nlls).mean())

    return ppl
