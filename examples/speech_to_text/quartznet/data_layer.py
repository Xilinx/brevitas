# Copyright (c) 2019 NVIDIA Corporation
"""
This package contains Neural Modules responsible for ASR-related
data layers.
"""
__all__ = ['AudioToTextDataLayer'] #,
           # 'KaldiFeatureDataLayer',
           # 'TranscriptDataLayer']

from functools import partial
import torch
import torch.nn as nn

# from nemo.backends.pytorch import DataLayerNM
# from nemo.core import DeviceType
# from nemo.core.neural_types import *
from .parts.dataset import (AudioDataset, seq_collate_fn)  # , KaldiFeatureDataset, TranscriptDataset)
from .parts.features import WaveformFeaturizer

def pad_to(x, k=8):
    """Pad int value up to divisor of k.

    Examples:
        >>> pad_to(31, 8)
        32

    """

    return x + (x % k > 0) * (k - x % k)

class AudioToTextDataLayer(nn.Module):
    """Data Layer for general ASR tasks.

    Module which reads ASR labeled data. It accepts comma-separated
    JSON manifest files describing the correspondence between wav audio files
    and their transcripts. JSON files should be of the following format::

        {"audio_filepath": path_to_wav_0, "duration": time_in_sec_0, "text": \
transcript_0}
        ...
        {"audio_filepath": path_to_wav_n, "duration": time_in_sec_n, "text": \
transcript_n}

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        labels (list): Dataset parameter.
            List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        eos_id (str): Dataset parameter.
            End of string symbol used for seq2seq models.
            Defaults to None.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        normalize_transcripts (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Dataset parameter.
            Controls whether the dataloader loads the audio signal and
            transcript or just the transcript.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        perturb_config (dict): Currently disabled.
    """

    # @staticmethod
    # def create_ports():
    #     input_ports = {}
    #     output_ports = {
    #         "audio_signal": NeuralType({0: AxisType(BatchTag),
    #                                     1: AxisType(TimeTag)}),
    #
    #         "a_sig_length": NeuralType({0: AxisType(BatchTag)}),
    #
    #         "transcripts": NeuralType({0: AxisType(BatchTag),
    #                                    1: AxisType(TimeTag)}),
    #
    #         "transcript_length": NeuralType({0: AxisType(BatchTag)})
    #     }
    #     return input_ports, output_ports

    def __init__(
            self, *,
            manifest_filepath,
            labels,
            batch_size,
            sample_rate=16000,
            int_values=False,
            bos_id=None,
            eos_id=None,
            pad_id=None,
            min_duration=0.1,
            max_duration=None,
            normalize_transcripts=True,
            trim_silence=False,
            load_audio=True,
            drop_last=False,
            shuffle=True,
            num_workers=0,
            placement='cpu',
            # perturb_config=None,
            **kwargs
    ):
        super().__init__()

        self._featurizer = WaveformFeaturizer(
            sample_rate=sample_rate, int_values=int_values, augmentor=None)

        # Set up dataset
        dataset_params = {'manifest_filepath': manifest_filepath,
                          'labels': labels,
                          'featurizer': self._featurizer,
                          'max_duration': max_duration,
                          'min_duration': min_duration,
                          'normalize': normalize_transcripts,
                          'trim': trim_silence,
                          'bos_id': bos_id,
                          'eos_id': eos_id,
                          'logger': None,
                          'load_audio': load_audio}

        self._dataset = AudioDataset(**dataset_params)

        # Set up data loader
        if placement == 'cuda':
            print('Parallelizing DATALAYER')
            sampler = torch.utils.data.distributed.DistributedSampler(
                self._dataset)
        else:
            sampler = None

        pad_id = 0 if pad_id is None else pad_id
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(seq_collate_fn, token_pad_value=pad_id),
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader


# class KaldiFeatureDataLayer(DataLayerNM):
#     """Data layer for reading generic Kaldi-formatted data.
#
#     Module that reads ASR labeled data that is in a Kaldi-compatible format.
#     It assumes that you have a directory that contains:
#
#     - feats.scp: A mapping from utterance IDs to .ark files that
#             contain the corresponding MFCC (or other format) data
#     - text: A mapping from utterance IDs to transcripts
#     - utt2dur (optional): A mapping from utterance IDs to audio durations,
#             needed if you want to filter based on duration
#
#     Args:
#         kaldi_dir (str): Directory that contains the above files.
#         labels (list): List of characters that can be output by the ASR model,
#             e.g. {a-z '} for Jasper. The CTC blank symbol is automatically
#             added later for models using CTC.
#         batch_size (int): batch size
#         eos_id (str): End of string symbol used for seq2seq models.
#             Defaults to None.
#         min_duration (float): All training files which have a duration less
#             than min_duration are dropped. Can't be used if the `utt2dur` file
#             does not exist. Defaults to None.
#         max_duration (float): All training files which have a duration more
#             than max_duration are dropped. Can't be used if the `utt2dur` file
#             does not exist. Defaults to None.
#         normalize_transcripts (bool): Whether to use automatic text cleaning.
#             It is highly recommended to manually clean text for best results.
#             Defaults to True.
#         drop_last (bool): See PyTorch DataLoader. Defaults to False.
#         shuffle (bool): See PyTorch DataLoader. Defaults to True.
#         num_workers (int): See PyTorch DataLoader. Defaults to 0.
#     """
#
#     @staticmethod
#     def create_ports():
#         input_ports = {}
#         output_ports = {
#             "processed_signal": NeuralType({0: AxisType(BatchTag),
#                                             1: AxisType(SpectrogramSignalTag),
#                                             2: AxisType(ProcessedTimeTag)}),
#
#             "processed_length": NeuralType({0: AxisType(BatchTag)}),
#
#             "transcripts": NeuralType({0: AxisType(BatchTag),
#                                        1: AxisType(TimeTag)}),
#
#             "transcript_length": NeuralType({0: AxisType(BatchTag)})
#         }
#         return input_ports, output_ports
#
#     def __init__(
#             self, *,
#             kaldi_dir,
#             labels,
#             batch_size,
#             min_duration=None,
#             max_duration=None,
#             normalize_transcripts=True,
#             drop_last=False,
#             shuffle=True,
#             num_workers=0,
#             **kwargs
#     ):
#         super().__init__(**kwargs)
#
#         # Set up dataset
#         dataset_params = {'kaldi_dir': kaldi_dir,
#                           'labels': labels,
#                           'min_duration': min_duration,
#                           'max_duration': max_duration,
#                           'normalize': normalize_transcripts,
#                           'logger': self._logger}
#         self._dataset = KaldiFeatureDataset(**dataset_params)
#
#         # Set up data loader
#         if self._placement == DeviceType.AllGpu:
#             self._logger.info('Parallelizing DATALAYER')
#             sampler = torch.utils.data.distributed.DistributedSampler(
#                 self._dataset)
#         else:
#             sampler = None
#
#         self._dataloader = torch.utils.data.DataLoader(
#             dataset=self._dataset,
#             batch_size=batch_size,
#             collate_fn=self._collate_fn,
#             drop_last=drop_last,
#             shuffle=shuffle if sampler is None else False,
#             sampler=sampler,
#             num_workers=num_workers
#         )
#
#     @staticmethod
#     def _collate_fn(batch):
#         """Collate batch of (features, feature len, tokens, tokens len).
#         Kaldi generally uses MFCC (and PLP) features.
#
#         Args:
#             batch: A batch of elements, where each element is a tuple of
#                 features, feature length, tokens, and token
#                 length for a single sample.
#
#         Returns:
#             The same batch, with the features and token length padded
#             to the maximum of the batch.
#         """
#         # Find max lengths of features and tokens in the batch
#         _, feat_lens, _, token_lens = zip(*batch)
#         max_feat_len = max(feat_lens).item()
#         max_tokens_len = max(token_lens).item()
#
#         # Pad features and tokens to max
#         features, tokens = [], []
#         for feat, feat_len, tkns, tkns_len in batch:
#             feat_len = feat_len.item()
#             if feat_len < max_feat_len:
#                 pad = (0, max_feat_len - feat_len)
#                 feat = torch.nn.functional.pad(feat, pad)
#             features.append(feat)
#
#             tkns_len = tkns_len.item()
#             if tkns_len < max_tokens_len:
#                 pad = (0, max_tokens_len - tkns_len)
#                 tkns = torch.nn.functional.pad(tkns, pad)
#             tokens.append(tkns)
#
#         features = torch.stack(features)
#         feature_lens = torch.stack(feat_lens)
#         tokens = torch.stack(tokens)
#         token_lens = torch.stack(token_lens)
#
#         return features, feature_lens, tokens, token_lens
#
#     def __len__(self):
#         return len(self._dataset)
#
#     @property
#     def dataset(self):
#         return None
#
#     @property
#     def data_iterator(self):
#         return self._dataloader
#
#
# class TranscriptDataLayer(DataLayerNM):
#     """A simple Neural Module for loading textual transcript data.
#     The path, labels, and eos_id arguments are dataset parameters.
#
#     Args:
#         pad_id (int): Label position of padding symbol
#         batch_size (int): Size of batches to generate in data loader
#         drop_last (bool): Whether we drop last (possibly) incomplete batch.
#             Defaults to False.
#         num_workers (int): Number of processes to work on data loading (0 for
#             just main process).
#             Defaults to 0.
#     """
#
#     @staticmethod
#     def create_ports():
#         input_ports = {}
#         output_ports = {
#             'texts': NeuralType({
#                 0: AxisType(BatchTag),
#                 1: AxisType(TimeTag)
#             }),
#
#             "texts_length": NeuralType({0: AxisType(BatchTag)})
#         }
#         return input_ports, output_ports
#
#     def __init__(self,
#                  path,
#                  labels,
#                  batch_size,
#                  bos_id=None,
#                  eos_id=None,
#                  pad_id=None,
#                  drop_last=False,
#                  num_workers=0,
#                  shuffle=True,
#                  **kwargs):
#         super().__init__(**kwargs)
#
#         # Set up dataset
#         dataset_params = {'path': path,
#                           'labels': labels,
#                           'bos_id': bos_id,
#                           'eos_id': eos_id}
#
#         self._dataset = TranscriptDataset(**dataset_params)
#
#         # Set up data loader
#         if self._placement == DeviceType.AllGpu:
#             sampler = torch.utils.data.distributed.DistributedSampler(
#                 self._dataset)
#         else:
#             sampler = None
#
#         pad_id = 0 if pad_id is None else pad_id
#
#         # noinspection PyTypeChecker
#         self._dataloader = torch.utils.data.DataLoader(
#             dataset=self._dataset,
#             batch_size=batch_size,
#             collate_fn=partial(self._collate_fn, pad_id=pad_id, pad8=True),
#             drop_last=drop_last,
#             shuffle=shuffle if sampler is None else False,
#             sampler=sampler,
#             num_workers=num_workers
#         )
#
#     @staticmethod
#     def _collate_fn(batch, pad_id, pad8=False):
#         texts_list, texts_len = zip(*batch)
#         max_len = max(texts_len)
#         if pad8:
#             max_len = pad_to(max_len, 8)
#
#         texts = torch.empty(len(texts_list), max_len,
#                             dtype=torch.long)
#         texts.fill_(pad_id)
#
#         for i, s in enumerate(texts_list):
#             texts[i].narrow(0, 0, s.size(0)).copy_(s)
#
#         if len(texts.shape) != 2:
#             raise ValueError(
#                 f"Texts in collate function have shape {texts.shape},"
#                 f" should have 2 dimensions."
#             )
#
#         return texts, torch.stack(texts_len)
#
#     def __len__(self):
#         return len(self._dataset)
#
#     @property
#     def dataset(self):
#         return None
#
#     @property
#     def data_iterator(self):
#         return self._dataloader
