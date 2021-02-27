# Adapted from https://github.com/NVIDIA/NeMo/blob/r0.9/collections/nemo_asr/nemo_asr/jasper.py
# Copyright (C) 2020 Xilinx (Giuseppe Franco)
# Copyright (C) 2019 NVIDIA CORPORATION.
#
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .audio_preprocessing import AudioToMelSpectrogramPreprocessor
from .parts.quartznet import JasperBlock, init_weights
from .parts.common import *
from .greedy_ctc_decoder import GreedyCTCDecoder


class JasperEncoder(nn.Module):
    """
    Jasper Encoder creates the pre-processing (prologue), Jasper convolution
    block, and the first 3 post-processing (epilogue) layers as described in
    Jasper (https://arxiv.org/abs/1904.03288)

    Args:
        jasper (list): A list of dictionaries. Each element in the list
            represents the configuration of one Jasper Block. Each element
            should contain::

                {
                    # Required parameters
                    'filters' (int) # Number of output channels,
                    'repeat' (int) # Number of sub-blocks,
                    'kernel' (int) # Size of conv kernel,
                    'stride' (int) # Conv stride
                    'dilation' (int) # Conv dilation
                    'dropout' (float) # Dropout probability
                    'residual' (bool) # Whether to use residual or not.
                    # Optional parameters
                    'residual_dense' (bool) # Whether to use Dense Residuals
                        # or not. 'residual' must be True for 'residual_dense'
                        # to be enabled.
                        # Defaults to False.
                    'separable' (bool) # Whether to use separable convolutions.
                        # Defaults to False
                    'groups' (int) # Number of groups in each conv layer.
                        # Defaults to 1
                    'heads' (int) # Sharing of separable filters
                        # Defaults to -1
                    'tied' (bool)  # Whether to use the same weights for all
                        # sub-blocks.
                        # Defaults to False
                }

        activation (str): Activation function used for each sub-blocks. Can be
            one of ["hardtanh", "relu", "selu"].
        feat_in (int): Number of channels being input to this module
        normalization_mode (str): Normalization to be used in each sub-block.
            Can be one of ["batch", "layer", "instance", "group"]
            Defaults to "batch".
        residual_mode (str): Type of residual connection.
            Can be "add" or "max".
            Defaults to "add".
        norm_groups (int): Number of groups for "group" normalization type.
            If set to -1, number of channels is used.
            Defaults to -1.
        conv_mask (bool): Controls the use of sequence length masking prior
            to convolutions.
            Defaults to True.
        frame_splicing (int): Defaults to 1.
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def __init__(
            self, *,
            jasper,
            outer_bit_width,
            inner_bit_width,
            weight_scaling_per_output_channel,
            absolute_act_val,
            activation_inner_scaling_per_output_channel,
            activation_other_scaling_per_output_channel,
            activation,
            feat_in,
            fused_bn=False,
            normalization_mode="batch",
            residual_mode="add",
            norm_groups=-1,
            conv_mask=True,
            frame_splicing=1,
            init_mode='xavier_uniform',
            **kwargs
    ):
        nn.Module.__init__(self)

        feat_in = feat_in * frame_splicing

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False

        for it, lcfg in enumerate(jasper):
            if it == 0:
                bit_width = outer_bit_width
            else:
                bit_width = inner_bit_width

            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            heads = lcfg.get('heads', -1)
            encoder_layers.append(
                JasperBlock(feat_in,
                            lcfg['filters'],
                            repeat=lcfg['repeat'],
                            kernel_size=lcfg['kernel'],
                            stride=lcfg['stride'],
                            dilation=lcfg['dilation'],
                            dropout=lcfg['dropout'],
                            residual=lcfg['residual'],
                            groups=groups,
                            fused_bn=fused_bn,
                            separable=separable,
                            heads=heads,
                            residual_mode=residual_mode,
                            normalization=normalization_mode,
                            norm_groups=norm_groups,
                            activation=activation,
                            residual_panes=dense_res,
                            conv_mask=conv_mask,
                            bit_width=bit_width,
                            absolute_act_val=absolute_act_val,
                            activation_inner_scaling_per_output_channel=activation_inner_scaling_per_output_channel,
                            activation_other_scaling_per_output_channel=activation_other_scaling_per_output_channel,
                            weight_scaling_per_output_channel=weight_scaling_per_output_channel),
                            )
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))
        # self.to(self._device)

    def forward(self, audio_signal, length=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor, Optional[Tensor]

        s_input, length = self.encoder(([audio_signal], length))
        if length is None:
            return s_input[-1]
        return s_input[-1], length


class JasperDecoderForCTC(nn.Module):
    """
    Jasper Decoder creates the final layer in Jasper that maps from the outputs
    of Jasper Encoder to the vocabulary of interest.

    Args:
        feat_in (int): Number of channels being input to this module
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def __init__(
            self, *,
            feat_in,
            num_classes,
            bit_width,
            weight_scaling_per_channel,
            init_mode="xavier_uniform",
            **kwargs
    ):
        nn.Module.__init__(self)

        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        self.decoder_layers = nn.Sequential(
            make_quantconv1d(self._feat_in, self._num_classes, kernel_size=1,bias=True, bit_width=bit_width,
                             scaling_per_channel=weight_scaling_per_channel))
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encoder_output):
        return F.log_softmax(self.decoder_layers(encoder_output).
                             transpose(1, 2), dim=-1)

class Quartznet(nn.Module):
    def __init__(self, preprocessing, encoder, decoder, greedyctcdecoder):
        super(Quartznet, self).__init__()
        self.preprocessing = preprocessing
        self.encoder = encoder
        self.decoder = decoder
        self.greedy_ctc_decoder = greedyctcdecoder

    def forward(self, input_tensors):
        if self.preprocessing is not None:  # with export mode disabled
            audio_signal_e1, a_sig_length_e1, _, _ = input_tensors
            processed_signal_e1, p_length_e1 = self.preprocessing(
            input_signal=audio_signal_e1,
            length=a_sig_length_e1)
            encoded_e1, encoded_len_e1 = self.encoder(
                audio_signal=processed_signal_e1,
                length=p_length_e1)
        else:  # with export mode enabled, no preprocessing, no length
            encoded_e1 = self.encoder(input_tensors)
        log_probs_e1 = self.decoder(encoder_output=encoded_e1)
        predictions_e1 = self.greedy_ctc_decoder(log_probs=log_probs_e1)
        return predictions_e1

    def restore_checkpoints(self, encoder_state_dict, decoder_state_dict):
        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)
        print("Checkpoint restored")


def quartznet(cfg, quartzet_params, export_mode):

    outer_bit_width = cfg.getint('QUANT', 'OUTER_LAYERS_BIT_WIDTH')
    inner_bit_width = cfg.getint('QUANT', 'INNER_LAYERS_BIT_WIDTH')
    activation_inner_scaling_per_output_channel = cfg.getboolean('ACTIVATIONS', 'INNER_SCALING_PER_CHANNEL')
    activation_other_scaling_per_output_channel = cfg.getboolean('ACTIVATIONS', 'OTHER_SCALING_PER_CHANNEL')
    absolute_act_val = cfg.getint('ACTIVATIONS', 'ABS_ACT_VAL')
    encoder_weight_scaling_per_output_channel = cfg.getboolean('WEIGHT', 'ENCODER_SCALING_PER_OUTPUT_CHANNEL')
    decoder_weight_scaling_per_output_channel = cfg.getboolean('WEIGHT', 'DECODER_SCALING_PER_OUTPUT_CHANNEL')
    fused_bn = cfg.getboolean('QUANT', 'FUSED_BN')

    vocab = quartzet_params['labels']
    sample_rate = quartzet_params['sample_rate']
    feat_in_encoder = quartzet_params["AudioToMelSpectrogramPreprocessor"]["features"]
    feat_in_decoder = quartzet_params["JasperEncoder"]["jasper"][-1]["filters"]

    if export_mode:
        quartzet_params['JasperEncoder']['conv_mask'] = False  # no conv masking in export mode
        data_preprocessor = None  # no built in preprocessing in export mode
    else:
        data_preprocessor = AudioToMelSpectrogramPreprocessor(
        sample_rate=sample_rate,
        **quartzet_params["AudioToMelSpectrogramPreprocessor"])

    encoder = JasperEncoder(
        feat_in=feat_in_encoder,
        weight_scaling_per_output_channel=encoder_weight_scaling_per_output_channel,
        inner_bit_width=inner_bit_width,
        outer_bit_width=outer_bit_width,
        absolute_act_val=absolute_act_val,
        activation_inner_scaling_per_output_channel=activation_inner_scaling_per_output_channel,
        activation_other_scaling_per_output_channel=activation_other_scaling_per_output_channel,
        fused_bn=fused_bn,
        **quartzet_params["JasperEncoder"])

    decoder = JasperDecoderForCTC(
        feat_in=feat_in_decoder,
        bit_width=outer_bit_width,
        weight_scaling_per_channel=decoder_weight_scaling_per_output_channel,
        num_classes=len(vocab))

    greedy_decoder = GreedyCTCDecoder()

    model = Quartznet(data_preprocessor, encoder, decoder, greedy_decoder)
    return model

