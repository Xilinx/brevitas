"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import torch


def unet_input_shape(resolution):
    return (4, resolution // 8, resolution // 8)


def generate_latents(seeds, device, dtype, input_shape):
    """
    Generate a concatenation of latents of a given input_shape
    (batch size excluded) on a target device from one or more seeds.
    """
    latents = None
    if not isinstance(seeds, (list, tuple)):
        seeds = [seeds]
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator = generator.manual_seed(seed)
        image_latents = torch.randn((1, *input_shape),
                                    generator=generator,
                                    device=device,
                                    dtype=dtype)
        latents = image_latents if latents is None else torch.cat((latents, image_latents))
    return latents


def generate_unet_rand_inputs(
        embedding_shape,
        unet_input_shape,
        batch_size=1,
        device='cpu',
        dtype=torch.float32,
        with_return_dict_false=False):
    sample = torch.randn(batch_size, *unet_input_shape, device=device, dtype=dtype)
    unet_rand_inputs = {
        'sample':
            sample,
        'timestep':
            torch.tensor(1, dtype=torch.int64, device=device),
        'encoder_hidden_states':
            torch.randn(batch_size, *embedding_shape, device=device, dtype=dtype)}
    if with_return_dict_false:
        unet_rand_inputs['return_dict'] = False
    return unet_rand_inputs
