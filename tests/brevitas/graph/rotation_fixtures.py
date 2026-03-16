# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools

import pytest_cases
from pytest_cases import fixture_union
import torch
import torch.nn as nn

IN_FEATURES = 24


@pytest_cases.fixture
def linear_rms():

    class LinearRMSModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=True)
            self.linear.weight.data.fill_(2.)
            self.linear.bias.data.fill_(1.)
            self.rms = nn.RMSNorm(4)
            self.rms.weight.data = torch.randn_like(
                self.rms.weight.data)  # Change learned parameters
            self.linear_1 = nn.Linear(4, 8, bias=False)
            self.linear_1.weight.data.fill_(2.)
            self.linear_2 = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            x = self.linear(x)
            x = self.rms(x)
            x = self.linear_1(x)
            x = self.linear_2(x) * x
            x = torch.matmul(x.flatten(1), x.flatten(1).t())

            return x

    return LinearRMSModel


@pytest_cases.fixture
def attention_sdpa():

    class AttentionSDPA(nn.Module):

        def __init__(self) -> None:
            super().__init__()

            self.q_proj = nn.Linear(4, 16, bias=False)
            self.k_proj = nn.Linear(4, 16, bias=False)
            self.v_proj = nn.Linear(4, 16, bias=False)
            self.o_proj = nn.Linear(16, 4, bias=False)

        def forward(self, x):
            hidden_shape = (1, 4, -1, 4)

            query_states = self.q_proj(x).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(x).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(x).view(hidden_shape).transpose(1, 2)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=True)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(1, 4, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            return attn_output

    return AttentionSDPA


list_of_rotation_mixtures = ['linear_rms', 'attention_sdpa']

rotation_fixtures = fixture_union(
    'rotation_fixtures', list_of_rotation_mixtures, ids=list_of_rotation_mixtures)

RESIDUAL_MODEL_REGION_DICTS = [
    {
        "srcs": ["embedding", "block1_linear2", "block2_linear2"],
        "sinks": ["block1_linear1", "block2_linear1", "head"],},
    {
        "srcs": ["block1_linear1"], "sinks": ["block1_linear2"]},
    {
        "srcs": [], "sinks": ["block2_linear2"]},]


class BlockResidualModel(nn.Module):

    def __init__(self, is_tied: bool = False) -> None:
        super().__init__()
        self.embedding = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)

        self.block1_linear1 = nn.Linear(IN_FEATURES, IN_FEATURES, bias=True)
        self.block1_linear2 = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)

        self.block2_linear1 = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)
        self.act = nn.SiLU()
        self.block2_linear2 = nn.Linear(IN_FEATURES, IN_FEATURES, bias=True)

        self.head = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)
        if is_tied:
            self.head.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        r = x
        x = self.block1_linear1(x)
        x = self.block1_linear2(x) + r
        r = x
        x = self.block2_linear1(x)
        x = self.act(x)
        x = self.block2_linear2(x) + r
        x = self.head(x)
        return x


@pytest_cases.fixture
@pytest_cases.parametrize('is_tied', [True, False])
def block_residual_model(is_tied):
    return functools.partial(BlockResidualModel, is_tied=is_tied)


list_of_rotation_fixtures = ["block_residual_model"]

rotation_model = fixture_union(
    'rotation_model', list_of_rotation_fixtures, ids=list_of_rotation_fixtures)
