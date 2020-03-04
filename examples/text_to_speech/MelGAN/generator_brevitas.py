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
