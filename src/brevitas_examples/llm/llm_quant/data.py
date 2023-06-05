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

from datasets import load_dataset
import torch
from transformers import AutoTokenizer


def get_c4(nsamples, seed, seqlen, model, nvalsamples=256):
    traindata = load_dataset(
        'allenai/c4',
        'allenai--c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train',
        use_auth_token=False)
    valdata = load_dataset(
        'allenai/c4',
        'allenai--c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation',
        use_auth_token=False)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    random.seed(0)  # hardcoded for validation reproducibility
    valenc = []
    for _ in range(nvalsamples):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])

    valenc = torch.hstack(valenc)
    return trainloader, valenc
