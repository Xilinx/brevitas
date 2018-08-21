/* 
Copyright (c) 2018-     Xilinx, Inc             

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Xilinx nor the names of its contributors 
   may be used to endorse or promote products derived from this 
   software without specific prior written permission.

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
#define THC_GENERIC_FILE "generic/quantized_generic_fused_rnn_cuda_wrapper.c"
#else


int forward_op(quantized_fused_lstm)(
   	THCTensor *input,
   	THCTensor *hidden,
   	THCTensor *bias1,
   	THCTensor *bias2,
   	THCTensor *cx,
   	THCTensor *hy,
   	THCTensor *cy,
    THCTensor *quantizationBitWidth) 
{
	THNN_(QuantizedLSTMFused_updateOutput)(state, input, hidden, bias1, bias2, cx, hy, cy, quantizationBitWidth);
	return 1;
}

int forward_op(quantized_fused_lstm_nobias)(
   	THCTensor *input,
   	THCTensor *hidden,
   	THCTensor *cx,
   	THCTensor *hy,
   	THCTensor *cy,
    THCTensor *quantizationBitWidth) 
{
	THNN_(QuantizedLSTMFusedNoBias_updateOutput)(state, input, hidden, cx, hy, cy, quantizationBitWidth);
	return 1;
}

int backward_op(quantized_fused_lstm)(
   THCTensor *storage,
   THCTensor *gradInGates,
   THCTensor *cx,
   THCTensor *cy,
   THCTensor *gradOutput,
   THCTensor *gradOutputCell,
   THCTensor *gradInputCx)
{
	THNN_(QuantizedLSTMFused_updateGradInput)(state, storage, gradInGates, cx, cy, gradOutput,gradOutputCell, gradInputCx);
	return 1;
}

int forward_op(quantized_fused_gru)(
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *bias1,
   THCTensor *bias2,
   THCTensor *hx,
   THCTensor *hy,
   THCTensor *storage)
{
	THNN_(QuantizedGRUFused_updateOutput)(state, input, hidden, bias1, bias2, hx, hy, storage);
	return 1;
}

int forward_op(quantized_fused_gru_nobias)(
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *hx,
   THCTensor *hy,
   THCTensor *storage)
{
	THNN_(QuantizedGRUFusedNoBias_updateOutput)(state, input, hidden, hx, hy, storage);
	return 1;
}

int backward_op(quantized_fused_gru)(
   THCTensor *gradInInput,
   THCTensor *gradInHidden,
   THCTensor *gradOutput,
   THCTensor *gradInputHx,
   THCTensor *storage)
{
	THNN_(QuantizedGRUFused_updateGradInput)(state, gradInInput, gradInHidden, gradOutput, gradInputHx, storage);
    return 1;
}


#endif
