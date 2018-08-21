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

#include <THC.h>
#include <THCApply.cuh>

#include <common.h>
#include <math_generics.cuh>

#include "quantized_fused_rnn_kernel.h"

template <typename T>
struct TensorSigmoidOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    T one = (T) 1.0;
    *out = one / (one + math_generics::exp(- *in));
  }

  __device__ __forceinline__ void operator()(T* v) const {
    T one = (T) 1.0;
    *v = one / (one + math_generics::exp(- *v));
  }
};

template <typename T>
struct TensorTanhOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    *out = math_generics::tanh(*in);
  }

  __device__ __forceinline__ void operator()(T* v) const {
    *v = math_generics::tanh(*v);
  }
};

template <typename T>
struct FixedPointQuantizationOp {
  __device__ __forceinline__ void operator()(T* out, T* in, T *min_val, T *max_val, T *pre_scale, T *post_scale) const {
    T clipped_value = math_generics::max(math_generics::min(*in, *max_val), *min_val);
    T rounded_value = math_generics::round(clipped_value * (*pre_scale));
    *out =  rounded_value * (*post_scale);
  }
};

template <typename T>
struct QuantizedTensorSigmoidOp {
  __device__ __forceinline__ void operator()(T* out, T* in, T *quantization_bit_width) const {
     if (*quantization_bit_width == (T) 32.0) {
      TensorSigmoidOp<T>()(out, in);
    } else { 
      T one = (T) 1.0;
      T two = (T) 2.0;
      T pre_scale = math_generics::pow(two, *quantization_bit_width);
      T post_scale = one / pre_scale;
      T min_val = (T) 0.0;
      T max_val = one - post_scale;
      TensorSigmoidOp<T>()(out, in);
      FixedPointQuantizationOp<T>()(out, out, &min_val, &max_val, &pre_scale, &post_scale);
   }
  }
};

template <typename T>
struct QuantizedTensorTanhOp {
  __device__ __forceinline__ void operator()(T* out, T* in, T *quantization_bit_width) const {
     if (*quantization_bit_width == (T) 32.0) {
      TensorTanhOp<T>()(out, in);
    } else { 
      T one = (T) 1.0;
      T two = (T) 2.0;
      T pre_scale = math_generics::pow(two, *quantization_bit_width - one);
      T post_scale = math_generics::pow(two, - *quantization_bit_width + one);
      T min_val = (T) -1.0;
      T max_val = one - post_scale;
      TensorTanhOp<T>()(out, in);
      FixedPointQuantizationOp<T>()(out, out, &min_val, &max_val, &pre_scale, &post_scale);
    }
  }
};

#include "generic/quantized_generic_fused_rnn_kernel.cu"
#include <THCGenerateFloatType.h>

#include "generic/quantized_generic_fused_rnn_kernel.cu"
#include <THCGenerateDoubleType.h>
