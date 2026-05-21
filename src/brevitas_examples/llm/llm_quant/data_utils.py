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

from functools import partial
import random
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
import warnings

from datasets import Dataset
import numpy as np
from optimum.utils.normalized_config import NormalizedConfigManager
import torch
from transformers import AutoConfig

from brevitas_examples.llm.llm_quant.data import get_clm_dataset
from brevitas_examples.llm.llm_quant.data import get_wikitext2
from brevitas_examples.llm.llm_quant.data import load_raw_dataset


def collate_batch(examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, torch.Tensor]:
    # Make sure we load only what's necessary, ie we only load a `input_ids` column.
    if not all(list(example.keys()) == ["input_ids"] for example in examples):
        raise ValueError("Missing input_ids keys")

    input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
    input_ids = torch.tensor(input_ids)
    result: Dict[str, torch.Tensor] = {}
    # Process inputs: last token is the label
    result["input_ids"] = input_ids
    result["labels"] = input_ids
    result["attention_mask"] = torch.ones_like(input_ids)

    return result


def collate_batch_fx(examples: List[Dict[str, List[np.ndarray]]], num_kv_heads: int,
                     head_dim: int) -> Dict[str, torch.Tensor]:
    result = collate_batch(examples=examples)
    # In case the dataset is loaded to be used with an fx.GraphModule, we need to add empty past_key_values inputs in the dataset.
    dtype = result["input_ids"].dtype
    result["past_key_values"] = tuple((
        torch.zeros((1, num_kv_heads, 0, head_dim), dtype=dtype),
        torch.zeros((1, num_kv_heads, 0, head_dim), dtype=dtype),
    ) for _ in range(num_kv_heads))

    return result


def llm_collate(
    model_name_or_path: str,
    require_fx: bool = False
) -> Callable[[List[Dict[str, List[np.ndarray]]]], Dict[str, torch.Tensor]]:
    num_kv_heads = None
    head_dim = None
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
        return partial(collate_batch_fx, num_kv_heads=num_kv_heads, head_dim=head_dim)

    return partial(collate_batch)


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
        dataset_name: str,
        tokenizer: Any,
        nsamples: int = 128,
        seqlen: int = 2048,
        seed: int = 0,
        split: str = "train",
        bos_preprocessing: Optional[str] = None,
        add_eos_token: bool = False,
        fuse_documents: bool = True) -> Dataset:
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
            seed=seed,
            bos_preprocessing=bos_preprocessing,
            add_eos_token=add_eos_token,
            fuse_documents=fuse_documents)

    return dataset
