# Copyright (c) 2018-     Xilinx, Inc             

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx nor the names of its contributors 
#    may be used to endorse or promote products derived from this 
#    software without specific prior written permission.

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
from torch.autograd.function import Function
from torch.autograd import Variable

class Identity(Function):

    @staticmethod  
    def to_int(q_params, x):
        raise Exception

    @staticmethod  
    def forward(ctx, q_params, x):
        return x

    @staticmethod  
    def backward(ctx, grad):
        return None, grad
        

class FixedUnitWeight(Function):

    @staticmethod  
    def to_int(q_params, x):
        return x.clamp(q_params.min_val, q_params.max_val).mul(q_params.prescale).round()

    @staticmethod  
    def forward(ctx, q_params, x):
        return FixedUnitWeight.to_int(q_params, x).mul(q_params.postscale)

    @staticmethod  
    def backward(ctx, grad):
        return None, grad


class FixedUnitActivation(FixedUnitWeight):
    
    @staticmethod  
    def forward(ctx, q_params, x):
        ctx.save_for_backward(x)
        ctx.q_params = q_params
        return super(FixedUnitActivation, FixedUnitActivation).forward(ctx, q_params, x)

    @staticmethod  
    def backward(ctx, grad):
        q_params = ctx.q_params
        x, = ctx.saved_tensors

        min_tensor = x.new([q_params.min_val])
        max_tensor = x.new([q_params.max_val])
        
        #Mask has to be a Variable with a float tensor data
        mask = x.le(max_tensor) * x.ge(min_tensor)
        mask = Variable(mask.type(type(x)), requires_grad=False)

        return None, grad * mask
