"""
Copyright (c) 2023-     Advanced Micro Devices, Inc. (Ian Colbert)
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of AMD, Facebook, Deepmind Technologies, NYU,
   NEC Laboratories America and IDIAP Research Institute nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import os
import tarfile
from typing import Tuple, Type

from PIL import Image
from six.moves import urllib
from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import RandomCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

__all__ = ["get_bsd300_dataloaders"]

url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"


def is_valid_image_file(filename: str):
    extensions = [".png", ".jpg", ".jpeg"]
    for extension in extensions:
        if filename.endswith(extension):
            return True
    return False


def load_img_ycbcr(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def load_img_rbg(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):

    def __init__(self, image_dir, shared_transform, input_transform, target_transform):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [
            os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_valid_image_file(x)]

        self.shared_transform = shared_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img_rbg(self.image_filenames[index])
        input = self.shared_transform(input)
        target = input.copy()
        input = self.input_transform(input)
        target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.image_filenames)


def download_bsd300(dest: str = "data", download: bool = False):
    if not isinstance(dest, str):
        raise ValueError("Specify dataset directory with --data-dir")
    output_image_dir = os.path.join(dest, "BSDS300/images")
    if not os.path.exists(output_image_dir) and download:
        os.makedirs(dest, exist_ok=True)
        print("Data does not exist. Downloading from ", url)
        data = urllib.request.urlopen(url)
        file_path = os.path.join(dest, os.path.basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)
        os.remove(file_path)
    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_transforms(crop_size):
    return Compose([
        RandomCrop(crop_size, pad_if_needed=True), RandomHorizontalFlip(), RandomVerticalFlip()])


def test_transforms(crop_size):
    return Compose([CenterCrop(crop_size)])


def input_transform(crop_size, upscale_factor):
    return Compose([
        Resize(crop_size // upscale_factor),
        ToTensor(),])


def target_transform():
    return Compose([ToTensor()])


def get_training_set(upscale_factor: int, root_dir: str, crop_size: int):
    train_dir = os.path.join(root_dir, "train")
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
    return DatasetFromFolder(
        train_dir,
        shared_transform=train_transforms(crop_size),
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform())


def get_test_set(upscale_factor: int, root_dir: str, crop_size: int):
    test_dir = os.path.join(root_dir, "test")
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
    return DatasetFromFolder(
        test_dir,
        shared_transform=test_transforms(crop_size),
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform())


def get_bsd300_dataloaders(
        data_root: str,
        num_workers: int = 0,
        batch_size: int = 32,
        batch_size_test: int = 100,
        pin_memory: bool = True,
        upscale_factor: int = 3,
        crop_size: int = 512,
        download: bool = False) -> Tuple[Type[DataLoader]]:
    """Function that loads BSD300 dataset from data_root folder and returns the training
    and testing dataloaders. If <data_root>/BSD300/images does not exist, then the data is
    downloaded.

    Args:
        data_root (str): Root folder containing the BSD300 dataset.
        num_workers (int): Number of workers to use for both dataloaders. Default: 0
        batch_size (int): Size of batches to use for the training dataloader. Default: 32
        batch_size_test (int): Size of batches to use for the testing dataloader. When
            None, then batch_size_test = batch_size. Default: 100
        pin_memory (bool): Whether or not to pin the memory for both dataloaders. Default: True
        upscale_factor (int): The upscale factor for the super resolution task. Default: 3
        crop_size (int): The size to crop images for upscaling. Default: 512
        download (bool): Whether or not to download the dataset. Default: False
    """
    data_root = download_bsd300(data_root, download)
    trainset = get_training_set(upscale_factor, data_root, crop_size)
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0)
    testset = get_test_set(upscale_factor, data_root, crop_size)
    if batch_size_test is None:
        batch_size_test = batch_size
    testloader = DataLoader(
        testset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0)
    return trainloader, testloader
