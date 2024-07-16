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
from typing import Any, Optional, Union

import numpy as np
from optimum.amd.brevitas.data_utils import DatasetToDevice
from optimum.amd.brevitas.data_utils import get_c4
from optimum.amd.brevitas.data_utils import get_wikitext2
from optimum.utils.normalized_config import NormalizedConfigManager
import torch
from transformers import AutoConfig


def get_dataset_for_model(
    model_name_or_path: str,
    dataset_name: str,
    tokenizer: Any,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    split: str = "train",
    fuse_sequences: bool = True,
    require_fx: bool = False,
    device: Optional[Union[str, torch.device]] = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    get_dataset_map = {
        "wikitext2": get_wikitext2,
        "c4": get_c4,}
    if split not in ["train", "validation"]:
        raise ValueError(f"The split need to be 'train' or 'validation' but found {split}")
    if dataset_name not in get_dataset_map:
        raise ValueError(
            f"Expected a value in {list(get_dataset_map.keys())} but found {dataset_name}")
    get_dataset_fn = get_dataset_map[dataset_name]

    data = get_dataset_fn(
        tokenizer=tokenizer,
        nsamples=nsamples,
        seqlen=seqlen,
        split=split,
        fuse_sequences=fuse_sequences,
        seed=seed)

    # In case the dataset is loaded to be used with an fx.GraphModule, we need to add empty past_key_values inputs in the dataset.
    if require_fx:
        config = AutoConfig.from_pretrained(model_name_or_path)

        normalized_config_class = NormalizedConfigManager.get_normalized_config_class(
            config.model_type)
        normalized_config = normalized_config_class(config)

        num_heads = normalized_config.num_attention_heads
        if hasattr(normalized_config, "num_key_value_heads"):
            num_kv_heads = normalized_config.num_key_value_heads
        else:
            num_kv_heads = num_heads
        head_dim = normalized_config.hidden_size // num_heads
        num_layers = normalized_config.num_layers

        for sample in data:
            sample["past_key_values"] = tuple((
                torch.zeros(
                    1,
                    num_kv_heads,
                    0,
                    head_dim,
                    device=sample["input_ids"].device,
                    dtype=sample["input_ids"].dtype),
                torch.zeros(
                    1,
                    num_kv_heads,
                    0,
                    head_dim,
                    device=sample["input_ids"].device,
                    dtype=sample["input_ids"].dtype),
            ) for _ in range(num_layers))

    data = DatasetToDevice(data, device=device)

    return data
