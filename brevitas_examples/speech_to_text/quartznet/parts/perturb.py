# Audio dataset and corresponding functions taken from Patter  https://github.com/ryanleary/patter
# Adapted from https://github.com/NVIDIA/NeMo/tree/r0.9/collections/nemo_asr
# MIT License
# Copyright (c) 2020 Xilinx (Giuseppe Franco)
# Copyright (c) 2019 NVIDIA Corporation
# Copyright (c) 2018 Ryan Leary
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

import random

import librosa
from scipy import signal

from .manifest import ManifestEN
from .segment import AudioSegment


class Perturbation(object):
    def max_augmentation_length(self, length):
        return length

    def perturb(self, data):
        raise NotImplementedError


class SpeedPerturbation(Perturbation):
    def __init__(self, min_speed_rate=0.85, max_speed_rate=1.15, rng=None):
        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._rng = random.Random() if rng is None else rng

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def perturb(self, data):
        speed_rate = self._rng.uniform(self._min_rate, self._max_rate)
        if speed_rate <= 0:
            raise ValueError("speed_rate should be greater than zero.")
        # print("DEBUG: speed:", speed_rate)
        data._samples = librosa.effects.time_stretch(data._samples, speed_rate)


class GainPerturbation(Perturbation):
    def __init__(self, min_gain_dbfs=-10, max_gain_dbfs=10, rng=None):
        self._min_gain_dbfs = min_gain_dbfs
        self._max_gain_dbfs = max_gain_dbfs
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        gain = self._rng.uniform(self._min_gain_dbfs, self._max_gain_dbfs)
        # print("DEBUG: gain:", gain)
        data._samples = data._samples * (10. ** (gain / 20.))


class ImpulsePerturbation(Perturbation):
    def __init__(self, manifest_path=None, rng=None):
        self._manifest = ManifestEN(manifest_path)
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        impulse_record = self._rng.sample(self._manifest.data, 1)[0]
        impulse = AudioSegment.from_file(impulse_record['audio_filepath'],
                                         target_sr=data.sample_rate)
        # print("DEBUG: impulse:", impulse_record['audio_filepath'])
        data._samples = signal.fftconvolve(
            data.samples, impulse.samples, "full")


class ShiftPerturbation(Perturbation):
    def __init__(self, min_shift_ms=-5.0, max_shift_ms=5.0, rng=None):
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        if abs(shift_ms) / 1000 > data.duration:
            # TODO: do something smarter than just ignore this condition
            return
        shift_samples = int(shift_ms * data.sample_rate // 1000)
        # print("DEBUG: shift:", shift_samples)
        if shift_samples < 0:
            data._samples[-shift_samples:] = data._samples[:shift_samples]
            data._samples[:-shift_samples] = 0
        elif shift_samples > 0:
            data._samples[:-shift_samples] = data._samples[shift_samples:]
            data._samples[-shift_samples:] = 0


class NoisePerturbation(Perturbation):
    def __init__(self, manifest_path=None, min_snr_db=40, max_snr_db=50,
                 max_gain_db=300.0, rng=None):
        self._manifest = ManifestEN(manifest_path)
        self._rng = random.Random() if rng is None else rng
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db

    def perturb(self, data):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        noise_record = self._rng.sample(self._manifest.data, 1)[0]
        noise = AudioSegment.from_file(noise_record['audio_filepath'],
                                       target_sr=data.sample_rate)
        noise_gain_db = min(data.rms_db - noise.rms_db - snr_db,
                            self._max_gain_db)
        # print("DEBUG: noise:", snr_db, noise_gain_db, noise_record[
        # 'audio_filepath'])

        # calculate noise segment to use
        start_time = self._rng.uniform(0.0, noise.duration - data.duration)
        noise.subsegment(start_time=start_time,
                         end_time=start_time + data.duration)

        # adjust gain for snr purposes and superimpose
        noise.gain_db(noise_gain_db)
        data._samples = data._samples + noise.samples


perturbation_types = {
    "speed": SpeedPerturbation,
    "gain": GainPerturbation,
    "impulse": ImpulsePerturbation,
    "shift": ShiftPerturbation,
    "noise": NoisePerturbation
}


class AudioAugmentor(object):
    def __init__(self, perturbations=None, rng=None):
        self._rng = random.Random() if rng is None else rng
        self._pipeline = perturbations if perturbations is not None else []

    def perturb(self, segment):
        for (prob, p) in self._pipeline:
            if self._rng.random() < prob:
                p.perturb(segment)
        return

    def max_augmentation_length(self, length):
        newlen = length
        for (prob, p) in self._pipeline:
            newlen = p.max_augmentation_length(newlen)
        return newlen

    @classmethod
    def from_config(cls, config):
        ptbs = []
        for p in config:
            if p['aug_type'] not in perturbation_types:
                print(p['aug_type'], "perturbation not known. Skipping.")
                continue
            perturbation = perturbation_types[p['aug_type']]
            ptbs.append((p['prob'], perturbation(**p['cfg'])))
        return cls(perturbations=ptbs)
