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
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
import warnings

import numpy as np
from optimum.utils.normalized_config import NormalizedConfigManager
import torch
from transformers import AutoConfig

from brevitas_examples.llm.llm_quant.data import get_clm_dataset
from brevitas_examples.llm.llm_quant.data import get_wikitext2
from brevitas_examples.llm.llm_quant.data import load_raw_dataset


class DatasetToDevice(torch.utils.data.Dataset):

    def __init__(self, data: List, device: Optional[Union[str, torch.device]]):
        super().__init__()
        self.data = data
        self.device = device

    def __getitem__(self, idx):
        if self.device is not None:
            return {
                name: recursive_to_device(val, self.device) for name, val in self.data[idx].items()}
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


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


def get_dataset_for_model(
    model_name_or_path: str,
    dataset_name: str,
    tokenizer: Any,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    split: str = "train",
    bos_preprocessing: Optional[str] = None,
    add_eos_token: bool = False,
    fuse_documents: bool = True,
    require_fx: bool = False,
    device: Optional[Union[str, torch.device]] = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    test_splits = ["validation", "test"]

    if split not in ["train", *test_splits]:
        raise ValueError(f"The split need to be 'train' or 'validation' but found {split}")

    raw_dataset = load_raw_dataset(dataset_name=dataset_name, split=split, seed=seed)
    if dataset_name == "wikitext2" or (dataset_name == "pile" and split in test_splits):
        # Document level BOS preprocessing is not supported for Wikitext2 as each row does not belong to
        # a single document
        if bos_preprocessing == "document":
            bos_preprocessing = "sequence"
            warnings.warn(
                "Wikitext2 does not support document-level BOS. Default to sequence-level.")
        # Wikitext2 preprocessing matches the preprocessing in https://github.com/IST-DASLab/gptq/blob/main/datautils.py
        data = get_wikitext2(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            seqlen=seqlen,
            nsamples=nsamples,
            split=split,
            add_bos_token=(bos_preprocessing == "sequence" and tokenizer.bos_token_id is not None),
            seed=seed)
    else:
        data = get_clm_dataset(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            nsamples=nsamples,
            seqlen=seqlen,
            bos_preprocessing=bos_preprocessing,
            add_eos_token=add_eos_token,
            fuse_documents=fuse_documents)

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
