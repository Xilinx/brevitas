import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write

from MelGAN import *
import torch.backends.cudnn as cudnn
import brevitas.config

brevitas.config.IGNORE_MISSING_KEYS = False
MAX_WAV_VALUE = 32768.0
import configparser
import random
import os

SEED = 123456
models = {'melgan': melgan}

parser = argparse.ArgumentParser()
parser.add_argument('--input-folder', help='path to folder containing the val folder')
parser.add_argument('--model-cfg', type=str, help='Path to pretrained model .ini configuration file')
parser.add_argument('--workers', default=32, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=16, type=int, help='Minibatch size')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    assert os.path.exists(args.model_cfg)
    cfg = configparser.ConfigParser()
    cfg.read(args.model_cfg)
    sampling_rate = cfg.getint('AUDIO', 'sampling_rate')

    arch = cfg.get('MODEL', 'ARCH')

    model = models[arch](cfg)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        cudnn.benchmark = True
    pretrained_url = cfg.get('MODEL', 'PRETRAINED_URL')
    print("=> Loading checkpoint from:'{}'".format(pretrained_url))
    if args.gpu is None:
        loc = 'cpu'
        checkpoint = torch.hub.load_state_dict_from_url(pretrained_url)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.hub.load_state_dict_from_url(pretrained_url, map_location=loc)

    model.load_state_dict(checkpoint, strict=True)
    model.eval(inference=True)

    with torch.no_grad():
        for melpath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, '*.mel'))):
            mel = torch.load(melpath)
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)

            mel = mel.to(loc)

            audio = model.inference(mel)
            audio = audio.cpu().detach().numpy()

            out_path = melpath.replace('.mel', '_reconstructed.wav')
            write(out_path, sampling_rate, audio)


if __name__ == '__main__':
    main()
