# Copyright (c) 2020-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import torch
import math

from brevitas.core.stats import AbsPercentile, NegativePercentileOrZero, PercentileInterval



def test_abs_percentile_per_tensor():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for v in values:
        tensor = torch.Tensor(values)
        abs_percentile = AbsPercentile(v * 10, None)
        out = abs_percentile(tensor)
        assert v == out.item()


def test_abs_percentile_per_channel():
    v = 90
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tensor = torch.Tensor(values)
    tensor = tensor.repeat(2, 1)
    abs_percentile = AbsPercentile(v, stats_reduce_dim=1)
    out = abs_percentile(tensor)
    assert out.isclose(torch.Tensor([9, 9])).all().item()
    

class TestPercentile:
    
    def compute_percentile(self, x, low_q=None, high_q=None):
        low_p, high_p = None, None
        if low_q is not None:
            k = int(math.ceil(.01 * low_q * x.numel()))
            low_p = x.view(-1).kthvalue(k).values
        if high_q is not None:
            k = int(math.floor(.01 * high_q * x.numel() + 0.5))
            high_p = x.view(-1).kthvalue(k).values          
        return low_p, high_p
    
    def test_negative_percentile(self):
        values = [-1., -2., 5]
        values = torch.tensor(values)
        neg_percentile = NegativePercentileOrZero(0.01)
        out = neg_percentile(values)
        
        expected_out = torch.min(torch.tensor(0.), self.compute_percentile(values, low_q = 0.01)[0])
        
        assert torch.allclose(out, expected_out)

    def test_zero_percentile(self):
        values = [1., 2., 5]
        values = torch.tensor(values)
        neg_percentile = NegativePercentileOrZero(0.01)
        out = neg_percentile(values)
        
        expected_out = torch.tensor(0.)
        
        assert torch.allclose(out, expected_out)


    def test_interval_percentile(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        values = torch.tensor(values)
        interval_percentile = PercentileInterval(low_percentile_q = 0.01, high_percentile_q = 99.9)
        out = interval_percentile(values)
        
        range = self.compute_percentile(values, low_q = 0.01, high_q = 99.9)
        expected_out = torch.abs(range[1] - range[0])
        assert torch.allclose(out, expected_out)
