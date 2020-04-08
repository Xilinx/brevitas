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

import torch
from .res_stack_brevitas import ResStack
from .common import *

MAX_WAV_VALUE = 32768.0


class Generator(nn.Module):
    def __init__(self, mel_channel, bit_width, last_layer_bit_width):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel

        self.generator = nn.Sequential(
            nn.utils.weight_norm(make_quantconv1d(mel_channel, 512, kernel_size=7, stride=1, padding=3,
                                                  bit_width=bit_width)),

            make_leakyRelu_activation(bit_width=bit_width),

            nn.utils.weight_norm(make_transpconv1d(512, 256, kernel_size=16, stride=8, padding=4,
                                                   bit_width=bit_width)),

            ResStack(256, bit_width=bit_width),

            make_leakyRelu_activation(bit_width),
            nn.utils.weight_norm(make_transpconv1d(256, 128, kernel_size=16, stride=8, padding=4,
                                                   bit_width=bit_width)),

            ResStack(128, bit_width=bit_width),

            make_leakyRelu_activation(bit_width),
            nn.utils.weight_norm(make_transpconv1d(128, 64, kernel_size=4, stride=2, padding=1,
                                                   bit_width=bit_width)),

            ResStack(64, bit_width=bit_width),

            make_leakyRelu_activation(bit_width),
            nn.utils.weight_norm(
                make_transpconv1d(64, 32, kernel_size=4, stride=2, padding=1, bit_width=bit_width)),

            ResStack(32, bit_width=bit_width),

            make_leakyRelu_activation(bit_width),
            nn.utils.weight_norm(
                make_quantconv1d(32, 1, kernel_size=7, stride=1, padding=3, bit_width=bit_width)),
            make_tanh_activation(bit_width=last_layer_bit_width),
        )

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0  # roughly normalize spectrogram
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        hop_length = 256
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device)
        mel = torch.cat((mel, zero), dim=2)

        audio = self.forward(mel)
        audio = audio.squeeze()  # collapse all dimension except time axis
        audio = audio[:-(hop_length * 10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
        audio = audio.short()

        return audio


'''
    to run this, fix 
    from . import ResStack
    into
    from res_stack import ResStack
'''
if __name__ == '__main__':
    model = Generator(7)

    x = torch.randn(3, 7, 10)
    print(x.shape)

    y = model(x)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])
