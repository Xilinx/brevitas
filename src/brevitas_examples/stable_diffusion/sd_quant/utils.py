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


def generate_unet_21_rand_inputs(
        embedding_shape,
        unet_input_shape,
        batch_size=1,
        device='cpu',
        dtype=torch.float32,
        with_return_dict_false=False):
    unet_rand_inputs = generate_unet_rand_inputs(
        embedding_shape, unet_input_shape, batch_size, device, dtype, with_return_dict_false)
    return tuple(unet_rand_inputs.values())


def generate_unet_xl_rand_inputs(
        embedding_shape,
        unet_input_shape,
        batch_size=1,
        device='cpu',
        dtype=torch.float32,
        with_return_dict_false=False):
    # We need to pass a combination of args and kwargs to ONNX export
    # If we pass all kwargs, something breaks
    # If we pass only the last element as kwargs, since it is a dict, it has a weird interaction and something breaks
    # The solution is to pass only one argument as args, and everything else as kwargs
    unet_rand_inputs = generate_unet_rand_inputs(
        embedding_shape, unet_input_shape, batch_size, device, dtype, with_return_dict_false)
    sample = unet_rand_inputs['sample']
    del unet_rand_inputs['sample']
    unet_rand_inputs['timestep_cond'] = None
    unet_rand_inputs['cross_attention_kwargs'] = None
    unet_rand_inputs['added_cond_kwargs'] = {
        "text_embeds": torch.randn(1, 1280, dtype=dtype, device=device),
        "time_ids": torch.randn(1, 6, dtype=dtype, device=device)}
    inputs = (sample, unet_rand_inputs)
    return inputs
