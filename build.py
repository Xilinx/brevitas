##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Mostly based on the Pytorch-Encoding source code, due MIT copyright below
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found below
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MIT License

# Copyright (c) 2017 Hang Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software. 

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import torch
import platform
import subprocess
from torch.utils.ffi import create_extension

lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
cwd = os.path.dirname(os.path.realpath(__file__))
quantization_lib_path = os.path.join(cwd, "quantization", "lib")

# clean the build files
clean_cmd = ['bash', 'clean.sh']
subprocess.check_call(clean_cmd)

# build CUDA library
os.environ['TORCH_LIB_DIR'] = lib_path
if platform.system() == 'Darwin':
    os.environ['TH_LIBRARIES'] = os.path.join(lib_path,'libATen.1.dylib')
    QUANTIZATION_LIB = os.path.join(cwd, 'quantization/lib/libQUANTIZATION.dylib')

else:
    os.environ['TH_LIBRARIES'] = os.path.join(lib_path,'libATen.so.1')
    QUANTIZATION_LIB = os.path.join(cwd, 'quantization/lib/libQUANTIZATION.so')

build_all_cmd = ['bash', 'make.sh']
subprocess.check_call(build_all_cmd, env=dict(os.environ))

# build FFI
sources = [
    'quantization/src/quantized_fused_rnn_cuda_wrapper.cpp'
    ]
headers = [
    'quantization/src/quantized_fused_rnn_cuda_wrapper.h'
]
defines = [('WITH_CUDA', None)]
with_cuda = True 

include_path = [os.path.join(lib_path, 'include'),
                os.path.join(cwd,'quantization/kernel'), 
                os.path.join(cwd,'quantization/kernel/include'), 
                os.path.join(cwd,'quantization/kernel/thcunn_include'), 
                os.path.join(cwd,'quantization/src')]

def make_relative_rpath(path):
    if platform.system() == 'Darwin':
        return '-Wl,-rpath,' + path
    else:
        return '-Wl,-rpath,' + path

ffi = create_extension(
    'xilinx.torch',
    package=False,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    include_dirs = include_path,
    extra_link_args = [
        make_relative_rpath(lib_path),
        make_relative_rpath(quantization_lib_path),
        QUANTIZATION_LIB,
    ],
)

if __name__ == '__main__':
    ffi.build()