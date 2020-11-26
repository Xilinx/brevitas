import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write

import torch.backends.cudnn as cudnn
import brevitas.config
from .melgan import model_with_cfg

brevitas.config.IGNORE_MISSING_KEYS = False
MAX_WAV_VALUE = 32768.0
import random
import os

SEED = 123456

parser = argparse.ArgumentParser()
parser.add_argument('--input-folder', help='path to folder containing the val folder')
parser.add_argument('--workers', default=32, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=16, type=int, help='Minibatch size')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--pretrained', action='store_true', default=True, help='Load pretrained checkpoint')
parser.add_argument('--model', type=str, default='quant_melgan_8b', help='Name of the model')


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    model, cfg = model_with_cfg(args.model, args.pretrained)

    sampling_rate = cfg.getint('AUDIO', 'sampling_rate')

    if args.gpu is not None:
        loc = 'cuda:{}'.format(args.gpu)
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        cudnn.benchmark = True
    else:
        loc = 'cpu'

    model.to(loc)
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
