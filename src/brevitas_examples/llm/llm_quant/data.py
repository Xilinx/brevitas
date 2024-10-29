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


def get_c4(nsamples, seed, seqlen, tokenizer, split='train', nvalsamples=0):
    if split == 'train':
        data = load_dataset(
            'allenai/c4',
            'allenai--c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train',
            use_auth_token=False)

        random.seed(seed)
        dataloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(data) - 1)
                trainenc = tokenizer(data[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            dataloader.append(inp)
        return dataloader
    elif split == 'validation':
        data = load_dataset(
            'allenai/c4',
            'allenai--c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation',
            use_auth_token=False)

        random.seed(0)  # hardcoded for validation reproducibility
        valenc = []
        for _ in range(nvalsamples):
            while True:
                i = random.randint(0, len(data) - 1)
                tmp = tokenizer(data[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])

        valenc = torch.hstack(valenc)
        return valenc


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
