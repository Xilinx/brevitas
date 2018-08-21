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

// ************ START Create function calls ********** //
#define FILL_FUNCTION(ITYPE, DIM, FUNCTION) FUNCTION(ITYPE, DIM)

#define FILL_DIM(ITYPE, DIM, FUNCTION)          \
  switch (DIM) {                                \
  case -2:                                      \
    FILL_FUNCTION(ITYPE, -2, FUNCTION);         \
    break;                                      \
  case 1:                                       \
    FILL_FUNCTION(ITYPE, 1, FUNCTION);          \
    break;                                      \
  case 2:                                       \
    FILL_FUNCTION(ITYPE, 2, FUNCTION);          \
    break;                                      \
  default:                                      \
    FILL_FUNCTION(ITYPE, -1, FUNCTION);         \
    break;                                      \
  }

#define QUANTIZED_LSTM_FORWARD(ITYPE, DIM) THNN_(QuantizedLSTMForward)             \
  <DATATYPE, ITYPE, DIM>                                        \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>        \
  (inputI, hiddenI,                                             \
   bias1I, bias2I, cxI, hyI, cyI,                               \
   quantizationBitWidthI,                                       \
   hid_size, totalElements);

#define QUANTIZED_LSTM_BACKWARD(ITYPE, DIM) THNN_(QuantizedLSTMBackward)           \
  <DATATYPE, ITYPE, DIM>                                        \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>        \
  (storageI, gradingatesI, cxI, cyI,                            \
   gradoutI, gradoutcI, gradincxI,                              \
   hid_size, totalElements);

#define QUANTIZED_GRU_FORWARD(ITYPE, DIM) THNN_(QuantizedGRUForward) \
  <DATATYPE, ITYPE, DIM> \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>                \
  (inputI, hiddenI, bias1I, bias2I, hxI, hyI, storageI,                 \
   hid_size, totalElements);

#define QUANTIZED_GRU_BACKWARD(ITYPE, DIM) THNN_(QuantizedGRUBackward)                     \
  <DATATYPE, ITYPE, DIM>                                                \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>                \
  (gradininputI, gradinhiddenI, gradoutI, gradinhxI, storageI,          \
   hid_size, totalElements);

// ************ END Create actual function calls ************ //

template<typename INDTYPE>
void THNN_(QuantizedLSTM_forw_ind_wrap)(
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
  bool has_bias = (bias1!=NULL);

  int maxDim;
  if (has_bias) {
    THCUNN_assertSameGPU(state, 7, input, hidden, bias1, bias2, hy, cy, cx, quantizationBitWidth);
    maxDim = THNN_(minIndexType)
      (state, 7, input, hidden, bias1, bias2, hy, cy, cx);
  } else {
    THCUNN_assertSameGPU(state, 5, input, hidden, hy, cy, cx, quantizationBitWidth);
    maxDim = THNN_(minIndexType)
      (state, 5, input, hidden, hy, cy, cx);
  }

  ptrdiff_t totalElements = TensorUtils<THCTensor>::getNumElements(state, cx);

  const dim3 block = getApplyBlock();
  dim3 grid;
  THAssertMsg(getApplyGrid(state, totalElements, grid),
          "Could not get grid size for pointwise apply.");

  TensorInfo<DATATYPE, INDTYPE> inputI =
    getTensorInfo<THCTensor, INDTYPE>(state, input);
  TensorInfo<DATATYPE, INDTYPE> hiddenI =
    getTensorInfo<THCTensor, INDTYPE>(state, hidden);
  TensorInfo<DATATYPE, INDTYPE> cxI =
    getTensorInfo<THCTensor, INDTYPE>(state, cx);
  TensorInfo<DATATYPE, INDTYPE> hyI =
    getTensorInfo<THCTensor, INDTYPE>(state, hy);
  TensorInfo<DATATYPE, INDTYPE> cyI =
    getTensorInfo<THCTensor, INDTYPE>(state, cy);
  TensorInfo<DATATYPE, INDTYPE> quantizationBitWidthI =
    getTensorInfo<THCTensor, INDTYPE>(state, quantizationBitWidth);

  INDTYPE hid_size = cxI.sizes[cxI.dims-1];
  if(has_bias){
    THAssertMsg( hid_size*4 == THCTensor_(nElement)(state, bias1) &&
                 hid_size*4 == THCTensor_(nElement)(state, bias2),
                 "Bias in pointwise operation is an incorrect size, must be 4 x feature size.");
  }

  if(maxDim == -2){
    inputI.collapseDims();
    hiddenI.collapseDims();
    cxI.collapseDims();
    hyI.collapseDims();
    cyI.collapseDims();
    quantizationBitWidthI.collapseDims();
  }

  INDTYPE zero[1] = {0};
  TensorInfo<DATATYPE, INDTYPE> nullinfo =
    TensorInfo<DATATYPE, INDTYPE>(NULL, 1, zero, zero);
  TensorInfo<DATATYPE, INDTYPE> bias1I = nullinfo;
  TensorInfo<DATATYPE, INDTYPE> bias2I = nullinfo;

  if(has_bias){
    bias1I = getTensorInfo<THCTensor, INDTYPE>(state, bias1);
    bias2I = getTensorInfo<THCTensor, INDTYPE>(state, bias2);
    if(maxDim == -2){
      bias1I.collapseDims();
      bias2I.collapseDims();
    }
  }

  FILL_DIM(INDTYPE, maxDim, QUANTIZED_LSTM_FORWARD);

}

template<typename INDTYPE>
void THNN_(QuantizedLSTM_back_ind_wrap)(
   THCState *state,
   THCTensor *storage,
   THCTensor *gradInGates,
   THCTensor *cx,
   THCTensor *cy,
   THCTensor *gradOutput,
   THCTensor *gradOutputCell,
   THCTensor *gradInputCx)
{
  int maxDim = THNN_(minIndexType)
    (state, 7, storage, gradInGates, cx, cy,
     gradOutput, gradOutputCell, gradInputCx);
  ptrdiff_t totalElements = TensorUtils<THCTensor>::getNumElements(state, gradOutput);

  const dim3 block = getApplyBlock();
  dim3 grid;
  THAssertMsg(getApplyGrid(state, totalElements, grid),
              "Could not get grid size for pointwise apply");

  TensorInfo<DATATYPE, INDTYPE> storageI =
    getTensorInfo<THCTensor, INDTYPE>(state, storage);
  TensorInfo<DATATYPE, INDTYPE> gradingatesI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradInGates);
  TensorInfo<DATATYPE, INDTYPE> cxI =
    getTensorInfo<THCTensor, INDTYPE>(state, cx);
  TensorInfo<DATATYPE, INDTYPE> cyI =
    getTensorInfo<THCTensor, INDTYPE>(state, cy);
  TensorInfo<DATATYPE, INDTYPE> gradoutI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradOutput);
  TensorInfo<DATATYPE, INDTYPE> gradoutcI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradOutputCell);
  TensorInfo<DATATYPE, INDTYPE> gradincxI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradInputCx);

  INDTYPE hid_size = gradoutI.sizes[gradoutI.dims-1];

  if(maxDim == -2){
    storageI.collapseDims();
    gradingatesI.collapseDims();
    cxI.collapseDims();
    cyI.collapseDims();
    gradoutI.collapseDims();
    gradoutcI.collapseDims();
    gradincxI.collapseDims();
  }
  FILL_DIM(INDTYPE, maxDim, QUANTIZED_LSTM_BACKWARD);

}

template<typename INDTYPE>
void THNN_(QuantizedGRU_forw_ind_wrap)(
   THCState *state,
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *bias1,
   THCTensor *bias2,
   THCTensor *hx,
   THCTensor *hy,
   THCTensor *storage)
{
  bool has_bias = (bias1!=NULL);
  int maxDim;

  if(has_bias){
    THCUNN_assertSameGPU
      (state, 7, input, hidden, hx, hy, bias1, bias2, storage);
    maxDim = THNN_(minIndexType)
      (state, 7, input, hidden, hx, hy, bias1, bias2, storage);
  }else{
    THCUNN_assertSameGPU
      (state, 5, input, hidden, hx, hy, storage);
    maxDim = THNN_(minIndexType)
      (state, 5, input, hidden, hx, hy, storage);
  }

  ptrdiff_t totalElements = TensorUtils<THCTensor>::getNumElements(state, hx);

  const dim3 block = getApplyBlock();
  dim3 grid;
  THAssertMsg(getApplyGrid(state, totalElements, grid),
              "Could not get grid size for pointwise apply.");

  TensorInfo<DATATYPE, INDTYPE> inputI =
    getTensorInfo<THCTensor, INDTYPE>(state, input);
  TensorInfo<DATATYPE, INDTYPE> hiddenI =
    getTensorInfo<THCTensor, INDTYPE>(state, hidden);
  TensorInfo<DATATYPE, INDTYPE> hxI =
    getTensorInfo<THCTensor, INDTYPE>(state, hx);
  TensorInfo<DATATYPE, INDTYPE> hyI =
    getTensorInfo<THCTensor, INDTYPE>(state, hy);
  TensorInfo<DATATYPE, INDTYPE> storageI =
    getTensorInfo<THCTensor, INDTYPE>(state, storage);

  INDTYPE hid_size = hxI.sizes[hxI.dims-1];
  if(has_bias){
    THAssertMsg( hid_size*3 == THCTensor_(nElement)(state, bias1) &&
                 hid_size*3 == THCTensor_(nElement)(state, bias2),
                 "Bias in pointwise operation is an incorrect size, must be 3 x feature size.");
  }

  if(maxDim == -2){
    inputI.collapseDims();
    hiddenI.collapseDims();
    hyI.collapseDims();
    hxI.collapseDims();
    storageI.collapseDims();
  }

  INDTYPE zero[1] = {0};
  TensorInfo<DATATYPE, INDTYPE> nullinfo =
    TensorInfo<DATATYPE, INDTYPE>(NULL, 1, zero, zero);
  TensorInfo<DATATYPE, INDTYPE> bias1I = nullinfo;
  TensorInfo<DATATYPE, INDTYPE> bias2I = nullinfo;

  if(has_bias){
    bias1I = getTensorInfo<THCTensor, INDTYPE>(state, bias1);
    bias2I = getTensorInfo<THCTensor, INDTYPE>(state, bias2);
    if(maxDim == -2){
      bias1I.collapseDims();
      bias2I.collapseDims();
    }
  }

  FILL_DIM(INDTYPE, maxDim, QUANTIZED_GRU_FORWARD);

}

template<typename INDTYPE>
void THNN_(QuantizedGRU_back_ind_wrap)(
   THCState *state,
   THCTensor *gradInInput,
   THCTensor *gradInHidden,
   THCTensor *gradOutput,
   THCTensor *gradInputHx,
   THCTensor *storage)
{

  int maxDim = THNN_(minIndexType)(state, 5, gradInInput, gradInHidden, gradOutput,
                                   gradInputHx, storage);
  ptrdiff_t totalElements = TensorUtils<THCTensor>::getNumElements(state, gradOutput);

  const dim3 block = getApplyBlock();
  dim3 grid;
  THAssertMsg(getApplyGrid(state, totalElements, grid),
          "Could not get grid size for pointwise apply");

  TensorInfo<DATATYPE, INDTYPE> gradininputI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradInInput);
  TensorInfo<DATATYPE, INDTYPE> gradinhiddenI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradInHidden);
  TensorInfo<DATATYPE, INDTYPE> gradoutI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradOutput);
  TensorInfo<DATATYPE, INDTYPE> gradinhxI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradInputHx);
  TensorInfo<DATATYPE, INDTYPE> storageI =
    getTensorInfo<THCTensor, INDTYPE>(state, storage);

  INDTYPE hid_size = gradoutI.sizes[gradoutI.dims-1];

  if(maxDim == -2){
    gradininputI.collapseDims();
    gradinhiddenI.collapseDims();
    gradoutI.collapseDims();
    gradinhxI.collapseDims();
    storageI.collapseDims();
  }
  FILL_DIM(INDTYPE, maxDim, QUANTIZED_GRU_BACKWARD);
}

