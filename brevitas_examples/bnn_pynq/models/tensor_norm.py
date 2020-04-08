# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.init as init


class TensorNorm(nn.Module):
    def __init__(self, eps=1e-4, momentum=0.1):
        super().__init__()

        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.rand(1))
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            mean = x.mean()
            unbias_var = x.var(unbiased=True)
            biased_var = x.var(unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
            inv_std = 1 / (biased_var + self.eps).pow(0.5)
            return (x - mean) * inv_std * self.weight + self.bias
        else:
            return ((x - self.running_mean) / (self.running_var + self.eps).pow(0.5)) * self.weight + self.bias