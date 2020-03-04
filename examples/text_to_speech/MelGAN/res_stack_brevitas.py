"""BSD 3-Clause License
Source: https://github.com/seungwonpark/melgan

Copyright (c) 2020 Xilinx, Inc (Giuseppe Franco)
Copyright (c) 2019, Seungwon Park 박승원
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

from .common import *
from brevitas.quant_tensor import QuantTensor


class ResStack(nn.Module):
    def __init__(self, channel, bit_width):
        super(ResStack, self).__init__()
        self.scale_norm = make_hardtanh_activation(bit_width=bit_width, return_quant_tensor=True)
        self.layers = nn.ModuleList([
            nn.Sequential(
                make_leakyRelu_activation(bit_width),

                nn.utils.weight_norm(make_quantconv1d(channel, channel, kernel_size=3, stride=1, padding=3 ** i,
                                                      dilation=3 ** i, bit_width=bit_width)),

                make_leakyRelu_activation(bit_width),
                nn.utils.weight_norm(make_quantconv1d(channel, channel, kernel_size=3, stride=1, padding=1,
                                                      dilation=1, bit_width=bit_width)),
            )
            for i in range(3)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = self.scale_norm(x)
            if isinstance(x, QuantTensor):
                x_unp, _, _ = x
            else:
                x_unp = x
            x_layer = self.scale_norm(layer(x_unp))

            if isinstance(x_layer, QuantTensor):
                x_layer_unp, _, _ = x_layer
            else:
                x_layer_unp = x_layer

            if self.training:
                x = x_unp + x_layer_unp
            else:
                x = x + x_layer

        if isinstance(x, QuantTensor):
            x, _, _ = x

        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            nn.utils.remove_weight_norm(layer[1])
            nn.utils.remove_weight_norm(layer[3])
