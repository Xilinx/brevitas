import os
import glob
import tqdm
import torch
import argparse
import numpy as np
import configparser

from .utilities.stft import TacotronSTFT
from .utilities.audio_processing import read_wav_np


def main(cfg, args):
    filter_length = cfg.getint('AUDIO', 'filter_length')
    hop_length = cfg.getint('AUDIO', 'hop_length')
    win_length = cfg.getint('AUDIO', 'win_length')
    n_mel_channels = cfg.getint('AUDIO', 'n_mel_channels')
    sampling_rate = cfg.getint('AUDIO', 'sampling_rate')
    mel_fmin = cfg.getfloat('AUDIO', 'mel_fmin')
    mel_fmax = cfg.getfloat('AUDIO', 'mel_fmax')

    segment_length = cfg.getint('AUDIO', 'segment_length')
    pad_short = cfg.getint('AUDIO', 'pad_short')

    stft = TacotronSTFT(filter_length=filter_length,
                        hop_length=hop_length,
                        win_length=win_length,
                        n_mel_channels=n_mel_channels,
                        sampling_rate=sampling_rate,
                        mel_fmin=mel_fmin,
                        mel_fmax=mel_fmax)

    wav_files = glob.glob(os.path.join(args.data_path, '**', '*.wav'), recursive=True)

    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        sr, wav = read_wav_np(wavpath)
        assert sr == sampling_rate, \
            "sample rate mismatch. expected %d, got %d at %s" % \
            (sampling_rate, sr, wavpath)

        if len(wav) < segment_length + pad_short:
            wav = np.pad(wav, (0, segment_length + pad_short - len(wav)), mode='constant', constant_values=0.0)

        wav = torch.from_numpy(wav).unsqueeze(0)
        mel = stft.mel_spectrogram(wav)

        melpath = wavpath.replace('.wav', '.mel')
        torch.save(mel, melpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-d', '--data-path', type=str, required=True,
                        help="root directory of wav files")
    args = parser.parse_args()

    assert os.path.exists(args.config)
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    main(cfg, args)
