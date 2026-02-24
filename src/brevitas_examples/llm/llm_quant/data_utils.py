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
from typing import Callable
from typing import Dict
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
    # Pile and fineweb does not have a test section
    testless_datasets = ['pile', 'fineweb']

    if split not in ["train", *test_splits]:
        raise ValueError(f"The split need to be 'train' or 'validation' but found {split}")

    raw_dataset = load_raw_dataset(dataset_name=dataset_name, split=split, seed=seed)
    if dataset_name == "wikitext2" or (dataset_name in testless_datasets and split in test_splits):
        # Document level BOS preprocessing is not supported for Wikitext2 as each row does not belong to
        # a single document
        if bos_preprocessing == "document":
            bos_preprocessing = "sequence"
            warnings.warn(
                "Wikitext2 does not support document-level BOS. Default to sequence-level.")
        # Wikitext2 preprocessing matches the preprocessing in https://github.com/IST-DASLab/gptq/blob/main/datautils.py
        dataset = get_wikitext2(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            seqlen=seqlen,
            nsamples=nsamples,
            split=split,
            add_bos_token=(bos_preprocessing == "sequence" and tokenizer.bos_token_id is not None),
            seed=seed)
    else:
        dataset = get_clm_dataset(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            nsamples=nsamples,
            seqlen=seqlen,
            bos_preprocessing=bos_preprocessing,
            add_eos_token=add_eos_token,
            fuse_documents=fuse_documents)

    return dataset


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    kwargs = {}
    for curr_dict in batch:
        for key, value in curr_dict.items():
            if isinstance(value, torch.Tensor):
                if key not in kwargs:
                    kwargs[key] = []
                kwargs[key].append(value)
            else:
                if key not in kwargs:
                    kwargs[key] = value
    for key, value in kwargs.items():
        if isinstance(value, list) and len(value) > 0:
            kwargs[key] = torch.cat(kwargs[key], dim=0)
    return kwargs
