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
from typing import Dict
from typing import List
from typing import Optional
import warnings

from datasets import Dataset
from datasets import Features
from datasets import load_dataset
from datasets import Sequence
from datasets import Value
import numpy as np
import torch
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def group_texts(
        examples: Dict[str, List[np.ndarray]],
        fuse_documents: bool,
        sequence_length: int,
        bos_token_id: Optional[int],
        add_bos_token: bool) -> Dict[str, List[np.ndarray]]:
    # Concatenate all texts.
    if fuse_documents:
        examples = {k: [np.concatenate(v)] for k, v in examples.items()}
    add_bos_token = add_bos_token and bos_token_id is not None
    sequence_length = sequence_length - 1 if add_bos_token and bos_token_id is not None else sequence_length
    # Split by chunks of sequence_length.
    # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    result = {
        k: [
            np.concatenate(
                (np.array([bos_token_id]),
                 seq[i:i + sequence_length])) if add_bos_token else seq[i:i + sequence_length]
            for seq in t
            for i in range(0, len(seq) - sequence_length + 1, sequence_length)] for k,
        t in examples.items()}
    return result


def tokenize_and_group_texts(
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        filter_empty_sequences: bool = True,
        bos_preprocessing: Optional[str] = None,
        add_eos_token: bool = False,
        fuse_documents: bool = False) -> Dict[str, List[np.ndarray]]:
    # Filter empty sequences
    if filter_empty_sequences:
        texts = [text for text in texts if len(text) > 0]
    tokenized_batch = tokenizer.batch_encode_plus(
        texts,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=bos_preprocessing == "document")
    tokenized_batch = {
        k: [
            np.array(
                tokenized_texts + [tokenizer.eos_token_id] if (
                    add_eos_token and tokenizer.eos_token_id is not None and
                    tokenized_texts[-1] != tokenizer.eos_token_id) else tokenized_texts)
            for tokenized_texts in v] for k,
        v in tokenized_batch.items()}
    return group_texts(
        examples=tokenized_batch,
        fuse_documents=fuse_documents,
        sequence_length=sequence_length,
        bos_token_id=tokenizer.bos_token_id,
        add_bos_token=bos_preprocessing == "sequence",
    )


def _clm_dataset_to_list(row: np.ndarray,) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor(row["input_ids"], dtype=torch.int64).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def get_clm_dataset(
    raw_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    nsamples: int,
    seqlen: int,
    filter_empty_sequences: bool = True,
    bos_preprocessing: Optional[str] = None,
    add_eos_token: bool = False,
    fuse_documents: bool = False,
    dataset_processing_num_proc_per_process: int = 1,
    text_column_name: str = "text",
):
    """
    Methods group_texts, tokenize_and_group_texts and get_clm_dataset are adapted from
    https://github.com/huggingface/nanotron/blob/main/src/nanotron/data/processing.py,
    released under the following LICENSE:

    Copyright 2022 The HuggingFace Team. All rights reserved.

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
    # Preprocess dataset
    dataset = raw_dataset.map(
        partial(
            tokenize_and_group_texts,
            tokenizer=tokenizer,
            sequence_length=seqlen,
            filter_empty_sequences=filter_empty_sequences,
            bos_preprocessing=bos_preprocessing,
            add_eos_token=add_eos_token,
            fuse_documents=fuse_documents),
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=seqlen)}),
        batched=True,
        num_proc=dataset_processing_num_proc_per_process,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {seqlen}",
    )
    # Retrieve a random subset of sequences
    random_indices = [i for i in range(len(dataset))]
    random.shuffle(random_indices)
    random_indices = random_indices[:nsamples]
    # Retrive random slice of dataset
    dataset = dataset.select(random_indices)
    # Now return the slice in a format that can be converted to a DatasetToDevice
    return list(map(_clm_dataset_to_list, dataset))


def get_wikitext2(
        raw_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        seqlen: int,
        nsamples: int,
        split: str = 'train',
        add_bos_token: bool = False,
        seed: int = 42) -> List[Dict[str, torch.Tensor]]:
    random.seed(seed)
    # Add BOS token to each sequence if add_bos_token is True and the tokenizer supports this token
    if add_bos_token and tokenizer.bos_token_id is not None:
        seqlen = seqlen - 1
        sequence_process_fn = lambda inp: torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=inp.dtype, device=inp.device), inp],
                                                    dim=1)
    else:
        # Identity, the BOS token is not added
        sequence_process_fn = lambda inp: inp

    data = tokenizer("\n\n".join(raw_dataset['text']), return_tensors='pt')
    dataloader = []
    if split == 'train':
        for _ in tqdm(range(nsamples)):
            i = random.randint(0, data.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = sequence_process_fn(data.input_ids[:, i:j])
            attention_mask = torch.ones_like(inp)
            dataloader.append({'input_ids': inp, 'attention_mask': attention_mask})
    elif split in ['test', 'validation']:
        nsamples = data['input_ids'].numel() // seqlen
        for i in tqdm(range(nsamples)):
            batch = sequence_process_fn(data['input_ids'][:, (i * seqlen):((i + 1) * seqlen)])
            attention_mask = torch.ones_like(batch)
            dataloader.append({'input_ids': batch, 'attention_mask': attention_mask})
    return dataloader


def load_raw_dataset(dataset_name: str, split: str, seed: int = 42) -> Dataset:
    if dataset_name == "wikitext2":
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    elif dataset_name == "c4":
        if split == "train":
            data = load_dataset(
                "allenai/c4",
                split="train",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"})
        elif split == "validation":
            data = load_dataset(
                "allenai/c4",
                split="validation",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            )
        data = data.shuffle(seed=seed).select(range(10000))  # c4 is too big.
    elif dataset_name == "pile":
        if split == "train":
            data = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            data = data.shuffle(seed=seed).select(range(10000))  # c4 is too big.
        elif split == "validation":
            warnings.warn(
                f"There is no available validation split for pile. Defaulting to wikitext2.")
            data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    else:
        raise ValueError(f"Dataset {dataset_name} is not available")
    return data
