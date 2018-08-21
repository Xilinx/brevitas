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

//No pragma once since the header is included multiple times during the compilation process
//once per generated type

#define DATATYPE TensorUtils<THCTensor>::DataType

#define DEVICE_LINEAR_GET(D_TENSOR, INDEX)                              \
  D_TENSOR.data[IndexToOffset<T, IndexType, Dims>::get(INDEX, D_TENSOR)]

//factor will be 3 for GRU and 4 for LSTM
void THNN_(FusedRNNAssertSizes)(THCState *state, int factor, int count, ...)
{
  va_list list;
  va_start(list, count);
  THCTensor *input = va_arg(list, THCTensor*);
  THCTensor *hidden = va_arg(list, THCTensor*);
  THArgCheck(THCTensor_(nElement)(state, input) ==
             THCTensor_(nElement)(state, hidden),
             3, "Input and Hidden tensor sizes should be the same.");

  THAssertMsg(TensorUtils<THCTensor>::getDims(state, input) <= MAX_CUTORCH_DIMS,
              "Tensor dimension is too large.");

  THAssertMsg(TensorUtils<THCTensor>::getDims(state, hidden) <= MAX_CUTORCH_DIMS,
              "Tensor dimension is too large.");

  for (int arg=2; arg < count; ++arg){
    THCTensor *tens = va_arg(list, THCTensor*);
    THArgCheck(THCTensor_(nElement)(state, input) ==
               THCTensor_(nElement)(state, tens)*factor,
               3, "A pointwise tensor was not the right size, should have 1/%u the elements of input/hidden tensor.", arg, factor);
    THAssertMsg(TensorUtils<THCTensor>::getDims(state, tens) <= MAX_CUTORCH_DIMS,
         "Tensor dimension is too large.");
  }

  va_end(list);
}

int THNN_(minIndexType)(THCState *state, int count, ...)
{
  va_list list;
  va_start(list, count);

  THCTensor* tens = va_arg(list, THCTensor*);
  int startDim = TensorUtils<THCTensor>::getDims(state, tens);
  bool canCollapse = THCTensor_(isContiguous)(state,tens);

  for (int arg=1; arg < count; ++arg){
    tens = va_arg(list, THCTensor*);
    canCollapse = canCollapse && THCTensor_(isContiguous)(state, tens);
    if(TensorUtils<THCTensor>::getDims(state, tens) != startDim){
      va_end(list);
      return -1;
    }
  }
  va_end(list);
  if(canCollapse) return -2;
  return startDim;
}

bool THNN_(canUse32BitIndexMath)(THCState *state, int count, ...)
{
  va_list list;
  va_start(list, count);

  for (int arg=0; arg < count; ++arg){
    THCTensor *tens = va_arg(list, THCTensor*);
    if (!TensorUtils<THCTensor>::canUse32BitIndexMath(state, tens)){
      va_end(list);
      return false;
    }
  }
  va_end(list);
  return true;
}

