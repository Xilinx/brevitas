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

import torch
from torch import nn
from tqdm import tqdm


def create_validation_dataloader(data, seqlen):
    nsamples = data['input_ids'].numel() // seqlen
    val_dataloader = []
    for i in tqdm(range(nsamples)):
        batch = data['input_ids'][:, (i * seqlen):((i + 1) * seqlen)].cuda()
        attention_mask = torch.ones_like(batch)
        val_dataloader.append({'input_ids': batch, 'attention_mask': attention_mask})
    return val_dataloader


@torch.no_grad()
def model_eval(model, valenc, seqlen):

    nsamples = len(valenc)

    with torch.no_grad():
        nlls = []
        for inps in valenc:
            lm_logits = model(**inps)['logits']
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inps['input_ids'][:, 1:].cuda()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl
