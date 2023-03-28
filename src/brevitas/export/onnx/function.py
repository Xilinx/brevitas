# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.autograd import Function


class LSTMCellFn(Function):

    @staticmethod
    def symbolic(
            g,
            inp,
            weight_i,
            weight_h,
            bias,
            sequence_lens,
            hidden_state,
            cell_state,
            direction,
            hidden_size,
            cifg,
            layout,
            output_dir_axes,
            state_dir_axes):
        outputs, hidden_state, cell_state = g.op(
            f'LSTM',
            inp,
            weight_i,
            weight_h,
            bias,
            sequence_lens,
            hidden_state,
            cell_state,
            direction_s=direction,
            hidden_size_i=hidden_size,
            input_forget_i=cifg,
            layout_i=layout,
            outputs=3)
        outputs = g.op(f'Squeeze', outputs, output_dir_axes)
        hidden_state = g.op(f'Squeeze', hidden_state, state_dir_axes)
        cell_state = g.op(f'Squeeze', cell_state, state_dir_axes)
        return outputs, hidden_state, cell_state

    @staticmethod
    def forward(
            ctx,
            inp,
            weight_i,
            weight_h,
            bias,
            sequence_lens,
            hidden_state,
            cell_state,
            direction,
            hidden_size,
            cifg,
            layout,
            output_dir_axes,
            state_dir_axes):
        outputs = torch.zeros(inp.size(0), inp.size(1), hidden_state.size(1), device=inp.device)
        hidden_state = hidden_state.squeeze(state_dir_axes.item())
        cell_state = cell_state.squeeze(state_dir_axes.item())
        return outputs, hidden_state, cell_state
