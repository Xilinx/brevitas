# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Based on https://github.com/jafermarq/WinogradAwareNets
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Tuple, Union

import numpy as np
import torch
from torch.nn import Conv2d


def construct_vandermonde(rows: int, cols: int, poly_points: list):
    """ Constructs Vandermonde matrix as described in [1]
        Args:
            rows (int): rows of matrix
            cols (int): columns of matrix
            polyPoints (list): polynomial points used for matrix generation (x,y) format
        Return:
            V (np.ndarray): rows x cols Vandermonde matrix
    """

    assert rows == len(poly_points)

    for r in range(rows):
        row = []
        for c in range(cols):
            f = poly_points[r][0]
            g = poly_points[r][1]
            row.append(f ** c * g ** (cols - c - 1))

        if r == 0:
            V = np.array(row)
        else:
            V = np.vstack([V, row])

    return V.astype(np.float32)


def get_winograd_transforms(m: int, n: int, diagonals: list, poly_points: list) -> np.ndarray:
    """ Generates Winograd (Cook-Toom) transformation matrices given:
        Args:
            m (int): the number of outputs
            n (int): the kernle size
            diagonals (list): a list of values for diagonal matrices Sy, Sx, Sw
            polyPoints (list): a list of polynomial points for Vandermonde matrix generation
        Returns:
            Y_t (np.ndarray): final transform matrix to original space
            X_t (np.ndarray): data transformation matrix to Winograd space
            W (np.ndarray): kernel trasnformation matrix to Winograd space
    """

    Sy = np.diag(diagonals[0]).astype(np.float32)
    Sx = np.diag(diagonals[1]).astype(np.float32)
    Sw = np.diag(diagonals[2]).astype(np.float32)

    V_full_m = construct_vandermonde(m + n - 1, m, poly_points)
    A_t = np.dot(np.transpose(V_full_m), Sy)  # Y_t in [1]

    V_sqr = construct_vandermonde(m + n - 1, m + n - 1, poly_points)
    V_sqrt_inv = np.linalg.inv(V_sqr)
    B_t = np.dot(Sx, np.transpose(V_sqrt_inv))  # X_t in [1]

    V_full_n = construct_vandermonde(m + n - 1, n, poly_points)
    G = np.dot(Sw, V_full_n)  # W in [1]

    return A_t, B_t, G


def get_transforms(m: int, n: int):
    ''' Given number of outpus and kernel size, construct the Winograd transformation matrices (A,B,G) as described in [1] '''

    if m == 2 and n == 3:  # generate transforms A,B,G for F[2x2, 3x3] as in [2]
        diagonals = [[1, 1, 1, -1], [1, 2, 2, -1], [1, 0.5, 0.5, 1]]
        polyPoints = [[0, 1], [1, 1], [-1, 1], [1, 0]]

    elif m == 4 and n == 3:  # generate transforms A,B,G for F[4x4, 3x3] as in [2]
        diagonals = [[1, 1, 1, 1, 1, 1], [4, -6, -6, 24, 24, 1], [1 / 4, -1 / 6, -1 / 6, 1 / 24, 1 / 24, 1]]
        polyPoints = [[0, 1], [1, 1], [-1, 1], [2, 1], [-2, 1], [1, 0]]

    elif m == 6 and n == 3:  # empirically found F(6x6, 3x3) parameters from F[4x4, 3x3] parameters and recipe in [3]
        diagonals = [[1, 1, 1, 1, 1, 1, 1, 1], [4, -6, -6, 24, 24, 1, -12, -12],
                     [1 / 4, -1 / 6, -1 / 6, 1 / 24, 1 / 24, -1 / 12, -1 / 12, 1]]
        polyPoints = [[0, 1], [1, 1], [-1, 1], [2, 1], [-2, 1], [3 / 2, 1], [-3 / 2, 1], [1, 0]]
    else:
        raise ValueError(
            'Need to define parameters for F(' + str(m) + ' x ' + str(m) + ',' + str(n) + ' x ' + str(n) + ')')

    A_t, B_t, G = get_winograd_transforms(m, n, diagonals, polyPoints)

    return A_t, B_t, G


class WinogradConv2d(Conv2d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            F: int,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True) -> None:
        Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self._implicit_padding = (self.kernel_size[0] - 1) / 2
        # Generating the Winograd Transforms
        self.F = F
        A_t, B_t, G = get_transforms(F, kernel_size)
        self.a_matrix_t = torch.nn.Parameter(torch.from_numpy(A_t))
        self.b_matrix_t = torch.nn.Parameter(torch.from_numpy(B_t))
        self.g_matrix = torch.nn.Parameter(torch.from_numpy(G))
        # Internal temp variables to track tile shapes
        self._tiled_shape = None
        self._num_chunks = None

    def input_transform(self, tiled_input: torch.Tensor) -> torch.Tensor:
        V = torch.matmul(self.b_matrix_t, torch.matmul(tiled_input, self.b_matrix_t.t()))
        return V

    def weight_transform(self, tiled_weight: torch.Tensor) -> torch.Tensor:
        U = torch.matmul(self.g_matrix, torch.matmul(tiled_weight, self.g_matrix.t()))
        return U

    def output_inverse_transform(self, transformed_output: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.a_matrix_t, torch.matmul(transformed_output, self.a_matrix_t.t()))

    def tile_forward(self, x: torch.Tensor) -> torch.Tensor:
        unfold_size = self.F + self.kernel_size[0] - 1
        tiled_x = x.unfold(dimension=2, size=unfold_size, step=self.F)
        tiled_x = tiled_x.unfold(dimension=3, size=unfold_size, step=self.F)
        # Save shapes required during untile op
        self._tiled_shape = tiled_x.shape
        self._num_chunks = tiled_x.shape[2]
        tiled_x = tiled_x.contiguous().view(
            tiled_x.size(0), tiled_x.size(1), -1, tiled_x.size(4), tiled_x.size(5))
        return tiled_x

    def untile_forward(self, x):
        output = x.reshape(x.shape[0], x.shape[1], self._tiled_shape[2], self._tiled_shape[3], self.F, self.F)
        return output.transpose(4, 3).contiguous().squeeze().view(
            output.shape[0], output.shape[1], self.F * self._num_chunks, self.F * self._num_chunks)

    def input_pad(self, x):
        ''' Here we calculate the necessary padding given the padding for the filter dimensions and the F configuration '''
        number_tiling_positions = (x.shape[3] - 2 * self._implicit_padding) / self.F
        if number_tiling_positions.is_integer():
            self._tiling_padding = 0
        else:
            '''We need to add additional padding to the one added already to account for the size of the filter.'''
            decimal_part = number_tiling_positions - int(number_tiling_positions)
            to_pad = round((1.0 - decimal_part) * self.F)
            to_pad_even = round(to_pad / 2)
            self._tiling_padding = to_pad_even
        self._expected_output_width = x.shape[3] - 2 * self._implicit_padding
        pad = tuple([int(self._tiling_padding + self.padding[0])] * 4)
        x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
        return x

    def output_crop(self, x):
        # crop output if required. This is necessary if we added additional
        # padding to accommodate for an integer number of (F+k-1) x (F+k-1) tiles
        if x.shape[3] is not self._expected_output_width:
            padding = self._tiling_padding
            x = x[:, :, padding:-padding, padding:-padding]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded_input = self.input_pad(x)
        tiled_input = self.tile_forward(padded_input)
        tiled_weight = self.weight.unsqueeze(2)
        transformed_input = self.input_transform(tiled_input)
        transformed_weight = self.weight_transform(tiled_weight)
        transformed_input = transformed_input.unsqueeze(1).expand(-1, transformed_weight.shape[0], -1, -1, -1, -1)
        transformed_output = transformed_input * transformed_weight
        transformed_output = transformed_output.sum(dim=2, keepdim=False)
        output = self.output_inverse_transform(transformed_output)
        output = self.untile_forward(output)
        output = self.output_crop(output)
        output = output + self.bias.view(1, -1, 1, 1)
        return output

if __name__ == '__main__':
    m = WinogradConv2d(2, 4, 3, 4, padding=0)
    m_orig = torch.nn.Conv2d(2, 4, 3, padding=0)
    m.load_state_dict(m_orig.state_dict(), strict=False)
    inp = torch.randn(1, 2, 32, 32)
    assert torch.allclose(m(inp), m_orig(inp), atol=1e-5)
