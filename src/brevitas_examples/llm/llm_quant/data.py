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
from typing import Any

from datasets import load_dataset
import torch
from tqdm import tqdm


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
