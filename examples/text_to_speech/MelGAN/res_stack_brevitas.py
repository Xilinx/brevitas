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
