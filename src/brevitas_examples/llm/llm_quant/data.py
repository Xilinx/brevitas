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
