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
from typing import Any, Dict, List

from datasets import Dataset
from datasets import Features
from datasets import load_dataset
from datasets import Sequence
from datasets import Value
import numpy as np
import torch
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def get_c4(
        tokenizer: Any,
        seqlen: int,
        nsamples: int,
        split: str = "train",
        fuse_sequences: bool = True,
        seed: int = 42):
    random.seed(seed)

    if split == "train":
        data = load_dataset(
            "allenai/c4", split="train", data_files={"train": "en/c4-train.00000-of-01024.json.gz"})
    elif split == "validation":
        data = load_dataset(
            "allenai/c4",
            split="validation",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        )

    if fuse_sequences:
        data = data.shuffle(seed=seed)[:10000]  # c4 is too big.
        full_text = "\n\n".join(data["text"])
        tokenized_data = tokenizer(full_text, return_tensors="pt")

        dataset = []
        for _ in range(nsamples):
            i = random.randint(0, tokenized_data.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = tokenized_data.input_ids[:, i:j]
            attention_mask = torch.ones((1, seqlen), dtype=torch.int64)
            dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    else:
        dataset = []
        with tqdm(total=nsamples) as pbar:
            while len(dataset) < nsamples:
                data_index = random.randint(0, len(data) - 1)

                enc = tokenizer(data[data_index]["text"], return_tensors="pt")

                if enc["input_ids"].shape[1] < seqlen:
                    continue

                start_idx = random.randint(0, enc["input_ids"].shape[1] - seqlen)
                end_idx = start_idx + seqlen - 1
                attention_mask = torch.ones((1, seqlen), dtype=torch.int64)
                input_ids = enc["input_ids"][:, start_idx:end_idx + 1]

                # Add BOS token.
                if tokenizer.eos_token_id is not None:
                    input_ids[:, 0] = tokenizer.eos_token_id

                dataset.append({"input_ids": input_ids, "attention_mask": attention_mask})
                pbar.update(1)

    return dataset


def group_texts(examples: Dict[str, List[np.ndarray]],
                sequence_length: int) -> Dict[str, List[np.ndarray]]:
    # Concatenate all texts.
    concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
    total_length = len(concatenated_examples[next(iter(examples.keys()))])
    # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= sequence_length:
        total_length = ((total_length) // sequence_length) * sequence_length
    # Split by chunks of sequence_length.
    result = {
        k: [
            t[i:i + sequence_length]
            for i in range(0, total_length - (sequence_length), sequence_length)] for k,
        t in concatenated_examples.items()}
    return result


def _tokenize_and_group_texts(
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        filter_empty_sequences: bool = True) -> Dict[str, List[np.ndarray]]:
    # Filter empty sequences
    if filter_empty_sequences:
        texts = [text for text in texts if len(texts) > 0]
    tokenized_batch = tokenizer.batch_encode_plus(
        texts, return_attention_mask=False, return_token_type_ids=False)
    tokenized_batch = {
        k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
    return group_texts(tokenized_batch, sequence_length)


def _load_dataset(dataset_name: str, seed: int = 42) -> Dataset:
    if dataset_name == "wikitext2":
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    elif dataset_name == "c4":
        data = load_dataset(
            "allenai/c4", split="train", data_files={"train": "en/c4-train.00000-of-01024.json.gz"})
        data = data.shuffle(seed=seed).select(range(10000))  # c4 is too big.
    elif dataset_name == "pile":
        data = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise ValueError(f"Dataset {dataset_name} is not available.")
    return data


def _clm_dataset_to_list(row: np.ndarray,) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor(row["input_ids"], dtype=torch.int64).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def get_dataset_clm(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    nsamples: int,
    seqlen: int,
    split: str,
    fuse_sequences: bool,
    seed: int = 42,
    filter_empty_sequences: bool = True,
    dataset_processing_num_proc_per_process: int = 1,
    dataset_overwrite_cache: bool = False,
    text_column_name: str = "text",
):
    random.seed(seed)
    # Load given dataset
    raw_dataset = _load_dataset(dataset_name, seed)
    # Preprocess dataset
    dataset = raw_dataset.map(
        partial(
            _tokenize_and_group_texts,
            tokenizer=tokenizer,
            sequence_length=seqlen,
            filter_empty_sequences=filter_empty_sequences),
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=seqlen)}),
        batched=True,
        num_proc=dataset_processing_num_proc_per_process,
        load_from_cache_file=False,
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
        tokenizer: Any,
        seqlen: int,
        nsamples: int,
        split: str = 'train',
        fuse_sequences: bool = True,
        seed: int = 42):
    random.seed(seed)

    if split == 'train':
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        trainloader = []
        for _ in tqdm(range(nsamples)):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            attention_mask = torch.ones_like(inp)
            trainloader.append({'input_ids': inp, 'attention_mask': attention_mask})
        return trainloader
    elif split == 'validation':
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        data = tokenizer("\n\n".join(data['text']), return_tensors='pt')
        nsamples = data['input_ids'].numel() // seqlen
        testloader = []
        for i in tqdm(range(nsamples)):
            batch = data['input_ids'][:, (i * seqlen):((i + 1) * seqlen)]
            attention_mask = torch.ones_like(batch)
            testloader.append({'input_ids': batch, 'attention_mask': attention_mask})
        return testloader
    else:
        raise ValueError(f"{split} is invalid")
