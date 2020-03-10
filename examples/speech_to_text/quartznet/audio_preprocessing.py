# Adapted from https://github.com/NVIDIA/NeMo/blob/r0.9/collections/nemo_asr/
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

"""
This file contains neural modules responsible for preprocessing audio data.
"""
__all__ = ['AudioPreprocessing',
           'AudioPreprocessor',
           'AudioToMFCCPreprocessor',
           'AudioToMelSpectrogramPreprocessor',
           'AudioToSpectrogramPreprocessor',
           'MultiplyBatch',
           'SpectrogramAugmentation']

from abc import abstractmethod
import math
import torch
import torch.nn as nn
try:
    import torchaudio
    have_torchaudio = True
except ModuleNotFoundError:
    have_torchaudio = False
    print('Could not import torchaudio. Some features might not work.')

from .parts.features import FilterbankFeatures
from .parts.spectr_augment import SpecAugment, SpecCutout


class AudioPreprocessor(nn.Module):
    """
    A base class for Neural Modules that performs audio preprocessing,
    transforming the wav files to features.
    """
    def __init__(self, win_length, hop_length, **kwargs):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length

        # self.disable_casts = (self._opt_level == Optimization.mxprO1)

        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones
        }

    @torch.no_grad()
    def forward(self, input_signal, length):
        # if self.disable_casts:
        #     with amp.disable_casts():
        #         processed_signal = self.get_features(
        #                 input_signal.to(torch.float), length)
        # else:
        processed_signal = self.get_features(input_signal, length)

        processed_length = self.get_seq_len(length.float())
        return processed_signal, processed_length

    @abstractmethod
    def get_features(self, input_signal, length):
        # Called by forward(). Subclasses should implement this.
        pass

    def get_seq_len(self, length):
        # Called by forward()
        return torch.ceil(length / self.hop_length).to(dtype=torch.long)


class AudioToSpectrogramPreprocessor(AudioPreprocessor):
    """Preprocessor that converts wavs to spectrograms.
    Uses torchaudio's Spectrogram class as a featurizer.

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett', 'none', 'null']
            Defaults to "hann"
        normalized (bool): Whether to normalize by magnitude after stft
    """

    def __init__(
            self, *,
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            n_window_size=None,
            n_window_stride=None,
            n_fft=None,
            window="hann",
            normalized=True,
            **kwargs
    ):
        if not have_torchaudio:
            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary for "
                "AudioToSpectrogramPreprocessor. We recommend you try "
                "building it from source for the PyTorch version you have.")
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and "
                             f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(f"{self} received both window_stride and "
                             f"n_window_stride. Only one should be specified.")
        if window_size:
            n_window_size = int(window_size * sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * sample_rate)

        super().__init__(n_window_size, n_window_stride, **kwargs)

        self.win_length = n_window_size
        self.hop_length = n_window_stride

        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        # Set window_fn. None defaults to torch.ones.
        window_fn = self.torch_windows.get(window, None)
        if window_fn is None:
            raise ValueError(
                f"Window argument for AudioProcessor is invalid: {window}."
                f"For no window function, use 'ones' or None.")

        # Create featurizer.
        # Calls torch.stft under the hood, and is hard-coded to use center=True
        self.featurizer = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=window_fn,
            normalized=normalized
        )
        self.featurizer.to(self._device)

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal)


class AudioToMelSpectrogramPreprocessor(AudioPreprocessor):
    """Featurizer that converts wavs to mel spectrograms.
    We don't use torchaudio's implementation here because the original
    implementation is not the same, so for the sake of backwards-compatibility
    this will use the old FilterbankFeatures for now.

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett']
            Defaults to "hann"
        normalize (str): Can be one of ['per_feature', 'all_features']; all
            other options disable feature normalization. 'all_features'
            normalizes the entire spectrogram to be mean 0 with std 1.
            'pre_features' normalizes per channel / freq instead.
            Defaults to "per_feature"
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        preemph (float): Amount of pre emphasis to add to audio. Can be
            disabled by passing None.
            Defaults to 0.97
        features (int): Number of mel spectrogram freq bins to output.
            Defaults to 64
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        log (bool): Log features.
            Defaults to True
        log_zero_guard_type(str): Need to avoid taking the log of zero. There
            are two options: "add" or "clamp".
            Defaults to "add".
        log_zero_guard_value(float, or str): Add or clamp requires the number
            to add with or clamp to. log_zero_guard_value can either be a float
            or "tiny" or "eps". torch.finfo is used if "tiny" or "eps" is
            passed.
            Defaults to 2**-24.
        dither (float): Amount of white-noise dithering.
            Defaults to 1e-5
        pad_to (int): Ensures that the output size of the time dimension is
            a multiple of pad_to.
            Defaults to 16
        frame_splicing (int): Defaults to 1
        stft_conv (bool): If True, uses pytorch_stft and convolutions. If
            False, uses torch.stft.
            Defaults to False
        pad_value (float): The value that shorter mels are padded with.
            Defaults to 0
        mag_power (float): The power that the linear spectrogram is raised to
            prior to multiplication with mel basis.
            Defaults to 2 for a power spec
    """

    def __init__(
            self, *,
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            n_window_size=None,
            n_window_stride=None,
            window="hann",
            normalize="per_feature",
            n_fft=None,
            preemph=0.97,
            features=64,
            lowfreq=0,
            highfreq=None,
            log=True,
            log_zero_guard_type="add",
            log_zero_guard_value=2**-24,
            dither=1e-5,
            pad_to=16,
            frame_splicing=1,
            stft_conv=False,
            pad_value=0,
            mag_power=2.,
            **kwargs
    ):
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and "
                             f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(f"{self} received both window_stride and "
                             f"n_window_stride. Only one should be specified.")
        if window_size:
            n_window_size = int(window_size * sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * sample_rate)

        super().__init__(n_window_size, n_window_stride, **kwargs)

        self.featurizer = FilterbankFeatures(
            sample_rate=sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            stft_conv=stft_conv,
            pad_value=pad_value,
            mag_power=mag_power,
            logger=None
        )
        # self.featurizer.to(self._device)

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)

    def get_seq_len(self, seq_len):
        return self.featurizer.get_seq_len(seq_len)

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks


class AudioToMFCCPreprocessor(AudioPreprocessor):
    """Preprocessor that converts wavs to MFCCs.
    Uses torchaudio.transforms.MFCC.

    Args:
        sample_rate: The sample rate of the audio.
            Defaults to 16000.
        window_size: Size of window for fft in seconds. Used to calculate the
            win_length arg for mel spectrogram.
            Defaults to 0.02
        window_stride: Stride of window for fft in seconds. Used to caculate
            the hop_length arg for mel spect.
            Defaults to 0.01
        n_window_size: Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride: Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window: Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett', 'none', 'null'].
            Defaults to 'hann'
        n_fft: Length of FT window. If None, it uses the smallest power of 2
            that is larger than n_window_size.
            Defaults to None
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        n_mels: Number of mel filterbanks.
            Defaults to 64
        n_mfcc: Number of coefficients to retain
            Defaults to 64
        dct_type: Type of discrete cosine transform to use
        norm: Type of norm to use
        log: Whether to use log-mel spectrograms instead of db-scaled.
            Defaults to True.
    """

    def __init__(
            self, *,
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            n_window_size=None,
            n_window_stride=None,
            window='hann',
            n_fft=None,
            lowfreq=0.,
            highfreq=None,
            n_mels=64,
            n_mfcc=64,
            dct_type=2,
            norm='ortho',
            log=True,
            **kwargs):
        if not have_torchaudio:
            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary for "
                "AudioToMFCCPreprocessor. We recommend you try "
                "building it from source for the PyTorch version you have.")
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and "
                             f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(f"{self} received both window_stride and "
                             f"n_window_stride. Only one should be specified.")
        # Get win_length (n_window_size) and hop_length (n_window_stride)
        if window_size:
            n_window_size = int(window_size * sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * sample_rate)

        super().__init__(n_window_size, n_window_stride, **kwargs)

        mel_kwargs = {}

        mel_kwargs['f_min'] = lowfreq
        mel_kwargs['f_max'] = highfreq
        mel_kwargs['n_mels'] = n_mels

        mel_kwargs['n_fft'] = n_fft or 2 ** math.ceil(math.log2(n_window_size))

        mel_kwargs['win_length'] = n_window_size
        mel_kwargs['hop_length'] = n_window_stride

        # Set window_fn. None defaults to torch.ones.
        window_fn = self.torch_windows.get(window, None)
        if window_fn is None:
            raise ValueError(
                f"Window argument for AudioProcessor is invalid: {window}."
                f"For no window function, use 'ones' or None.")
        mel_kwargs['window_fn'] = window_fn

        # Use torchaudio's implementation of MFCCs as featurizer
        self.featurizer = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            log_mels=log,
            melkwargs=mel_kwargs
        )
        self.featurizer.to(self._device)

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal)


class SpectrogramAugmentation(nn.Module):
    """
    Performs time and freq cuts in one of two ways.

    SpecAugment zeroes out vertical and horizontal sections as described in
    SpecAugment (https://arxiv.org/abs/1904.08779). Arguments for use with
    SpecAugment are `freq_masks`, `time_masks`, `freq_width`, and `time_width`.

    SpecCutout zeroes out rectangulars as described in Cutout
    (https://arxiv.org/abs/1708.04552). Arguments for use with Cutout are
    `rect_masks`, `rect_freq`, and `rect_time`.

    Args:
        freq_masks (int): how many frequency segments should be cut.
            Defaults to 0.
        time_masks (int): how many time segments should be cut
            Defaults to 0.
        freq_width (int): maximum number of frequencies to be cut in one
            segment.
            Defaults to 10.
        time_width (int): maximum number of time steps to be cut in one
            segment
            Defaults to 10.
        rect_masks (int): how many rectangular masks should be cut
            Defaults to 0.
        rect_freq (int): maximum size of cut rectangles along the frequency
            dimension
            Defaults to 5.
        rect_time (int): maximum size of cut rectangles along the time
            dimension
            Defaults to 25.
    """

    def __init__(
            self, *,
            freq_masks=0,
            time_masks=0,
            freq_width=10,
            time_width=10,
            rect_masks=0,
            rect_time=5,
            rect_freq=20,
            rng=None,
            **kwargs
    ):
        nn.Module.__init__(self)

        if rect_masks > 0:
            self.spec_cutout = SpecCutout(
                rect_masks=rect_masks,
                rect_time=rect_time,
                rect_freq=rect_freq,
                rng=rng
            )
            # self.spec_cutout.to(self._device)
        else:
            self.spec_cutout = lambda x: x

        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng
            )
            # self.spec_augment.to(self._device)
        else:
            self.spec_augment = lambda x: x

    def forward(self, input_spec):
        augmented_spec = self.spec_cutout(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec


class MultiplyBatch(nn.Module):
    """
    Augmentation that repeats each element in a batch.
    Other augmentations can be applied afterwards.

    Args:
        mult_batch (int): number of repeats
    """

    def __init__(self, *, mult_batch=1):
        nn.Module.__init__(self)
        self.mult = mult_batch

    @torch.no_grad()
    def forward(self, in_x, in_x_len, in_y, in_y_len):
        out_x = in_x.repeat(self.mult, 1, 1)
        out_y = in_y.repeat(self.mult, 1)
        out_x_len = in_x_len.repeat(self.mult)
        out_y_len = in_y_len.repeat(self.mult)

        return out_x, out_x_len, out_y, out_y_len


def AudioPreprocessing(*args, **kwargs):
    raise NotImplementedError(
        "AudioPreprocessing has been deprecated and replaced by: "
        "AudioToMFCCPreprocessor, AudioToMelSpectrogramPreprocessor, and "
        "AudioToSpectrogramPreprocessor. For most ASR purposes "
        "AudioToMelSpectrogramPreprocessor does the same as the old "
        "AudioPreprocessing.")
