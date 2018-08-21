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

# Set the envrionment variable via python
SET(TORCH_LIB_DIR "$ENV{TORCH_LIB_DIR}")
MESSAGE(STATUS "TORCH_LIB_DIR: " ${TORCH_LIB_DIR})

# Find the include files
SET(TORCH_TH_INCLUDE_DIR "${TORCH_LIB_DIR}/include/TH")
SET(TORCH_THC_INCLUDE_DIR "${TORCH_LIB_DIR}/include/THC")
SET(TORCH_ATEN_INCLUDE_DIR "${TORCH_LIB_DIR}/include/THC")
SET(TORCH_INCLUDE_DIR "${TORCH_LIB_DIR}/include")

# Find the libs. We need to find libraries one by one.
SET(TH_LIBRARIES "$ENV{TH_LIBRARIES}")
SET(THC_LIBRARIES "$ENV{THC_LIBRARIES}")
