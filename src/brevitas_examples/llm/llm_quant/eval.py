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

import random
from typing import Any, Dict, List

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
