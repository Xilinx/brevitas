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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTorch.cmake)

IF(NOT CUDA_FOUND)
FIND_PACKAGE(CUDA 6.5 REQUIRED)
ENDIF()

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Boilerplate compiler flags
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Detect CUDA architecture and get best NVCC flags
IF(NOT COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS OR MSVC)
  INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/select_compute_arch.cmake)
ENDIF()
LIST(APPEND CUDA_NVCC_FLAGS $ENV{TORCH_NVCC_FLAGS})
CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA $ENV{TORCH_CUDA_ARCH_LIST})
LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
    if(CUDA_VERSION VERSION_LESS "8.0")
      MESSAGE(STATUS "Found gcc >=5 and CUDA <= 7.5, adding workaround C++ flags")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
    endif(CUDA_VERSION VERSION_LESS "8.0")
  endif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

IF(MSVC)
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler /wd4819")
  ADD_DEFINITIONS(-DTH_EXPORTS)
ENDIF()

SET(CMAKE_MACOSX_RPATH 1)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Actual compilation stuff
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

FILE(GLOB kernel_src kernel/*.cu)

MESSAGE(STATUS "TORCH_INCLUDE_DIR:" ${TORCH_INCLUDE_DIR})
MESSAGE(STATUS "TORCH_TH_INCLUDE_DIR:" ${TORCH_TH_INCLUDE_DIR})
MESSAGE(STATUS "TORCH_THC_INCLUDE_DIR:" ${TORCH_THC_INCLUDE_DIR})


CUDA_INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}/kernel
	${CMAKE_CURRENT_SOURCE_DIR}/kernel/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/thcunn_include    
    ${TORCH_INCLUDE_DIR}
    ${TORCH_TH_INCLUDE_DIR}
    ${TORCH_THC_INCLUDE_DIR} 
)

CUDA_ADD_LIBRARY(QUANTIZATION SHARED ${kernel_src})

IF(MSVC)
  SET_TARGET_PROPERTIES(QUANTIZATION PROPERTIES PREFIX "lib" IMPORT_PREFIX "lib")
ENDIF()

TARGET_LINK_LIBRARIES(QUANTIZATION 
	${THC_LIBRARIES} 
	${TH_LIBRARIES} 
)