/* 
Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
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

3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU, 
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

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/quantized_generic_fused_rnn_kernel.cu"
#else

#include <cstdarg>
#include <quantized_generic_fused_rnn_kernel_helper.cuh>
#include <quantized_generic_fused_rnn_kernel_impl.cuh>
#include <quantized_generic_fused_rnn_kernel_internal_wrap.cuh>

void THNN_(QuantizedLSTMFused_updateOutput)(
   THCState *state,
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *bias1,
   THCTensor *bias2,
   THCTensor *cx,
   THCTensor *hy,
   THCTensor *cy,
   THCTensor *quantizationBitWidth)
{
  THCTensor_(resizeAs)(state, hy, cx);
  THCTensor_(resizeAs)(state, cy, cx);
  THNN_(FusedRNNAssertSizes)(state, 4, 5, input, hidden, hy, cy, cx);

  bool canUse32bi = THNN_(canUse32BitIndexMath)
      (state, 7, input, hidden, bias1, bias2, hy, cy, cx);
  

  if (canUse32bi) {
    THNN_(QuantizedLSTM_forw_ind_wrap)<uint32_t>
      (state, input, hidden, bias1, bias2, cx, hy, cy, quantizationBitWidth);
  } else {
    THNN_(QuantizedLSTM_forw_ind_wrap)<uint64_t>
      (state, input, hidden, bias1, bias2, cx, hy, cy, quantizationBitWidth);
  }
  THCudaCheck(cudaGetLastError());
}

void THNN_(QuantizedLSTMFusedNoBias_updateOutput)(
   THCState *state,
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *cx,
   THCTensor *hy,
   THCTensor *cy,
   THCTensor *quantizationBitWidth)
{
  THCTensor_(resizeAs)(state, hy, cx);
  THCTensor_(resizeAs)(state, cy, cx);
  THNN_(FusedRNNAssertSizes)(state, 4, 5, input, hidden, hy, cy, cx);

  bool canUse32bi = THNN_(canUse32BitIndexMath)
      (state, 5, input, hidden, hy, cy, cx);
  
  if(canUse32bi){
    THNN_(QuantizedLSTM_forw_ind_wrap)<uint32_t>
      (state, input, hidden, NULL, NULL, cx, hy, cy, quantizationBitWidth);
  } else {
    THNN_(QuantizedLSTM_forw_ind_wrap)<uint64_t>
      (state, input, hidden, NULL, NULL, cx, hy, cy, quantizationBitWidth);
  }
  THCudaCheck(cudaGetLastError());
}

void THNN_(QuantizedLSTMFused_updateGradInput)(
   THCState *state,
   THCTensor *storage,
   THCTensor *gradInGates,
   THCTensor *cx,
   THCTensor *cy,
   THCTensor *gradOutput,
   THCTensor *gradOutputCell,
   THCTensor *gradInputCx)
{
  THCTensor_(resizeAs)(state, gradInputCx, gradOutput);
  THCUNN_assertSameGPU(state, 7, storage, gradInGates, cx, cy,
               gradOutput, gradOutputCell, gradInputCx);
  THNN_(FusedRNNAssertSizes)
    (state, 4, 7, storage, gradInGates, cx, cy,
     gradOutput, gradOutputCell, gradInputCx);

  bool canUse32bi = THNN_(canUse32BitIndexMath)
    (state, 7, storage, gradInGates, cx, cy,
     gradOutput, gradOutputCell, gradInputCx);

  if(canUse32bi){
    THNN_(QuantizedLSTM_back_ind_wrap)<uint32_t>
      (state, storage, gradInGates, cx, cy,
       gradOutput, gradOutputCell, gradInputCx);
  }else{
    THNN_(QuantizedLSTM_back_ind_wrap)<uint64_t>
      (state, storage, gradInGates, cx, cy,
       gradOutput, gradOutputCell, gradInputCx);
  }
  THCudaCheck(cudaGetLastError());
}

void THNN_(QuantizedGRUFused_updateOutput)(
   THCState *state,
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *bias1,
   THCTensor *bias2,
   THCTensor *hx,
   THCTensor *hy,
   THCTensor *storage)
{
  THCTensor_(resizeAs)(state, hy, hx);
  THNN_(FusedRNNAssertSizes)(state, 3, 4, input, hidden, hx, hy);
  THArgCheck(THCTensor_(nElement)(state, storage) ==
             THCTensor_(nElement)(state, hx)*5,
             3, "Storage tensor for fused kernel was not sized correctly.");

  bool canUse32bi = THNN_(canUse32BitIndexMath)
      (state, 7, input, hidden, hx, hy, bias1, bias2, storage);

  if (canUse32bi) {
    THNN_(QuantizedGRU_forw_ind_wrap)<uint32_t>
      (state, input, hidden, bias1, bias2, hx, hy, storage);
  } else {
    THNN_(QuantizedGRU_forw_ind_wrap)<uint64_t>
      (state, input, hidden, bias1, bias2, hx, hy, storage);
  }
  THCudaCheck(cudaGetLastError());
}

void THNN_(QuantizedGRUFusedNoBias_updateOutput)(
   THCState *state,
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *hx,
   THCTensor *hy,
   THCTensor *storage)
{
  THCTensor_(resizeAs)(state, hy, hx);
  THNN_(FusedRNNAssertSizes)(state, 3, 4, input, hidden, hx, hy);
  THArgCheck(THCTensor_(nElement)(state, storage) ==
             THCTensor_(nElement)(state, hx)*5,
             3, "Storage tensor for fused kernel was not sized correctly.");

  bool canUse32bi = THNN_(canUse32BitIndexMath)
      (state, 5, input, hidden, hx, hy, storage);
  
  if (canUse32bi) {
    THNN_(QuantizedGRU_forw_ind_wrap)<uint32_t>
      (state, input, hidden, NULL, NULL, hx, hy, storage);
  } else {
    THNN_(QuantizedGRU_forw_ind_wrap)<uint64_t>
      (state, input, hidden, NULL, NULL, hx, hy, storage);
  }
  THCudaCheck(cudaGetLastError());
}

void THNN_(QuantizedGRUFused_updateGradInput)(
   THCState *state,
   THCTensor *gradInInput,
   THCTensor *gradInHidden,
   THCTensor *gradOutput,
   THCTensor *gradInputHx,
   THCTensor *storage)
{
  THCTensor_(resizeAs)(state, gradInputHx, gradOutput);
  THCUNN_assertSameGPU(state, 5, gradInInput, gradInHidden, gradOutput, gradInputHx, storage);
  THNN_(FusedRNNAssertSizes)(state, 3, 4, gradInInput, gradInHidden, gradOutput, gradInputHx);
  bool canUse32bi = THNN_(canUse32BitIndexMath)(state, 5, gradInInput, gradInHidden,
                                                gradOutput, gradInputHx, storage);
  if(canUse32bi){
    THNN_(QuantizedGRU_back_ind_wrap)<uint32_t>
      (state, gradInInput, gradInHidden, gradOutput, gradInputHx, storage);
  }else{
    THNN_(QuantizedGRU_back_ind_wrap)<uint64_t>
      (state, gradInInput, gradInHidden, gradOutput, gradInputHx, storage);
  }

  THCudaCheck(cudaGetLastError());
}

#include <quantized_fused_rnn_kernel_cleanup.h>

#endif
