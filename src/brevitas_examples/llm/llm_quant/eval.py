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

from brevitas_examples.llm.llm_quant.run_utils import apply_layer_inference_fn
from brevitas_examples.llm.llm_quant.run_utils import get_model_impl
from brevitas_examples.llm.llm_quant.run_utils import InputCatcherException


def eval_inference_fn(curr_layer, inps, outs, cached_values):
    curr_layer.cuda()
    for j in range(len(inps)):
        outs[j] = curr_layer(inps[j].unsqueeze(0).cuda(), **cached_values)[0]
    curr_layer.cpu()


@torch.no_grad()
def model_eval(model, valenc, seqlen):

    nsamples = valenc.numel() // seqlen

    def eval_input_capture_fn(model, data):
        for i in range(nsamples):
            batch = data[:, (i * seqlen):((i + 1) * seqlen)].cuda()
            try:
                model(batch)
            except InputCatcherException:
                pass

    inps = apply_layer_inference_fn(
        model,
        valenc,
        nsamples,
        input_capture_fn=eval_input_capture_fn,
        inference_fn=eval_inference_fn,
        seqlen=seqlen)

    model_impl = get_model_impl(model)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if hasattr(model_impl, 'norm') and model_impl.norm is not None:
        model_impl.norm = model_impl.norm.cuda()
    if hasattr(model_impl, 'final_layer_norm') and model_impl.final_layer_norm is not None:
        model_impl.final_layer_norm = model_impl.final_layer_norm.cuda()
    if hasattr(model_impl, 'project_out') and model_impl.project_out is not None:
        model_impl.project_out = model_impl.project_out.cuda()
    if hasattr(model, 'lm_head'):
        model.lm_head = model.lm_head.cuda()

    valenc = valenc.cuda()
    nlls = []
    for i in tqdm(range(nsamples)):
        hidden_states = inps[i].unsqueeze(0)
        if hasattr(model_impl, 'norm') and model_impl.norm is not None:
            hidden_states = model_impl.norm(hidden_states)
        if hasattr(model_impl, 'final_layer_norm') and model_impl.final_layer_norm is not None:
            hidden_states = model_impl.final_layer_norm(hidden_states)
        if hasattr(model_impl, 'project_out') and model_impl.project_out is not None:
            hidden_states = model_impl.project_out(hidden_states)
        lm_logits = hidden_states
        if hasattr(model, 'lm_head'):
            lm_logits = model.lm_head(lm_logits)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = valenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    model.config.use_cache = use_cache
    return ppl
