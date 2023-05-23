# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from time import sleep
import warnings

import torch
import torchvision

from brevitas_examples.imagenet_classification import models

TIME_BETWEEN_RETRIES = 2 * 60  # in seconds


def get_model_config(model_name):
    config = dict()
    # Set-up config parameters
    if model_name == 'inception_v3' or model_name == 'googlenet':
        config['inception_preprocessing'] = True
    else:
        config['inception_preprocessing'] = False

    if model_name == 'inception_v3':
        input_shape = 299
        resize_shape = 342
    else:
        input_shape = 224
        resize_shape = 256
    config.update({'resize_shape': resize_shape, 'center_crop_shape': input_shape})
    return config


def get_torchvision_model(model_name):
    model_fn = getattr(torchvision.models, model_name)
    if model_name == 'inception_v3' or model_name == 'googlenet':
        model = model_fn(pretrained=True, transform_input=False)
    else:
        model = model_fn(pretrained=True)

    return model


def get_quant_model(model_name, bit_width):
    model_fn = getattr(models, model_name)
    model = model_fn(float_pretrained=True, bit_width=bit_width)
    return model


def add_bool_arg(parser, name, default, help, str_true=False):
    dest = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    if str_true:
        group.add_argument('--' + name, dest=dest, type=str, help=help)
    else:
        group.add_argument('--' + name, dest=dest, action='store_true', help='Enable ' + help)
    group.add_argument('--no-' + name, dest=dest, action='store_false', help='Disable ' + help)
    parser.set_defaults(**{dest: default})


def get_gpu_index(idx):
    gpu_world_size = torch.cuda.device_count()
    if gpu_world_size == 0:
        # If no GPU is available, execute on CPU
        return None

    gpu_id = idx % gpu_world_size
    try:
        # If GPUtil is instealled, fetch the next available GPU
        return get_next_available_gpu(gpu_id)
    except ModuleNotFoundError:
        warnings.warn("GPUtil not installed, multiple jobs might end up on the same GPU")
        # If GPUtil is not installed, cycle through GPUs
        return gpu_id


def get_next_available_gpu(gpu_id):
    import GPUtil
    GPUs = GPUtil.getGPUs()
    available_gpus = GPUtil.getAvailability(
        GPUs, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
    if available_gpus[gpu_id] == 0:
        next_available_gpu = GPUtil.getAvailable(
            order='memory',
            limit=3,
            maxLoad=0.5,
            maxMemory=0.5,
            includeNan=False,
            excludeID=[],
            excludeUUID=[])
        while len(next_available_gpu) == 0:
            sleep(TIME_BETWEEN_RETRIES)
            next_available_gpu = GPUtil.getAvailable(
                order='memory',
                limit=3,
                maxLoad=0.5,
                maxMemory=0.5,
                includeNan=False,
                excludeID=[],
                excludeUUID=[])

        gpu_id = next_available_gpu[0]
    return gpu_id
