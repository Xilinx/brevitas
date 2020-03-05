# Copyright (c) 2019 NVIDIA Corporation
import torch
import torch.nn as nn

class GreedyCTCDecoder(nn.Module):
    """
    Greedy decoder that computes the argmax over a softmax distribution
    """
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, log_probs):
        with torch.no_grad():
            argmx = log_probs.argmax(dim=-1, keepdim=False)
            return argmx
