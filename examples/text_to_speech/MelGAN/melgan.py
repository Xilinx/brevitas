from .generator_brevitas import Generator


def melgan(cfg):
    bit_width = cfg.getint('QUANT', 'BIT_WIDTH')
    last_layer_bit_width = cfg.getint('QUANT', 'LAST_LAYER_BIT_WIDTH')
    mel_channels = cfg.getint('AUDIO', 'n_mel_channels')
    model = Generator(mel_channels, bit_width, last_layer_bit_width)

    return model
