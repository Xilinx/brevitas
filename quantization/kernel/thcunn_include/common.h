/* 
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, 
   NEC Laboratories America and IDIAP Research Institute nor the names 
   of its contributors may be used to endorse or promote products derived 
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE. 
*/

#ifndef THCUNN_COMMON_H
#define THCUNN_COMMON_H

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define THCUNN_assertSameGPU(...) THAssertMsg(THCTensor_(checkGPU)(__VA_ARGS__), \
  "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.")

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define THCUNN_resizeAs_indices(STATE, I1, I2)              \
  THLongStorage *size2 = THCTensor_(newSizeOf)(STATE, I2);  \
  if (!THCIndexTensor_(isSize)(STATE, I1, size2))           \
  { \
    THCudaLongTensor_resize(STATE, I1, size2, NULL);        \
  } \
  THLongStorage_free(size2);

#define THCUNN_check_shape(STATE, I1, I2)                 \
  if (I1 != NULL && I2 != NULL && !THCTensor_(isSameSizeAs)(STATE, I1, I2))	\
  { \
       THCDescBuff s1 = THCTensor_(sizeDesc)(STATE, I1);  \
       THCDescBuff s2 = THCTensor_(sizeDesc)(STATE, I2);  \
       THError(#I1 " and " #I2 " shapes do not match: "   \
               #I1 " %s, " #I2 " %s", s1.str, s2.str);    \
  }


#define THCUNN_check_shape_indices(STATE, I1, I2)              \
  THLongStorage *size2 = THCTensor_(newSizeOf)(STATE, I2);     \
  if (!THCIndexTensor_(isSize)(STATE, I1, size2))              \
  { \
       THCDescBuff s1 = THCIndexTensor_(sizeDesc)(STATE, I1);  \
       THCDescBuff s2 = THCTensor_(sizeDesc)(STATE, I2);       \
       THError(#I1 " and " #I2 " shapes do not match: "        \
               #I1 " %s, " #I2 " %s", s1.str, s2.str);         \
  } \
  THLongStorage_free(size2);

#define THCUNN_check_nElement(STATE, I1, I2)                \
  if (I1 != NULL && I2 != NULL ) {                          \
    ptrdiff_t n1 = THCTensor_(nElement)(STATE, I1);              \
    ptrdiff_t n2 = THCTensor_(nElement)(STATE, I2);              \
    if (n1 != n2)                                           \
    {	\
      THCDescBuff s1 = THCTensor_(sizeDesc)(state, I1);     \
      THCDescBuff s2 = THCTensor_(sizeDesc)(state, I2);     \
      THError(#I1 " and " #I2 " have different number of elements: "	\
              #I1 "%s has %ld elements, while "             \
              #I2 "%s has %ld elements", s1.str, n1, s2.str, n2); \
    }	\
  }

#define THCUNN_check_dim_size(STATE, T, DIM, DIM_SIZE, SIZE) \
  if (THCTensor_(nDimension)(STATE, T) != DIM ||             \
      THCTensor_(size)(STATE, T, DIM_SIZE) != SIZE) {        \
      THCDescBuff s1 = THCTensor_(sizeDesc)(state, T);       \
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"	\
              " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THCUNN_check_dim_size_indices(STATE, T, DIM, DIM_SIZE, SIZE)  \
  if (THCIndexTensor_(nDimension)(STATE, T) != DIM ||                 \
      THCIndexTensor_(size)(STATE, T, DIM_SIZE) != SIZE) {            \
      THCDescBuff s1 = THCIndexTensor_(sizeDesc)(state, T);           \
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d" \
              " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THCUNN_argCheck(STATE, COND, ARG, T, FORMAT) \
  if (!(COND)) { \
    THCDescBuff s1 = THCTensor_(sizeDesc)(state, T); \
    THArgCheck(COND, ARG, FORMAT, s1.str);           \
  }

#endif
