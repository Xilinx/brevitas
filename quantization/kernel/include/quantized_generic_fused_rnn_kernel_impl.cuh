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

template <typename T, typename IndexType, int Dims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
THNN_(QuantizedGRUForward)(TensorInfo<T, IndexType> Input,
            TensorInfo<T, IndexType> Hidden,
            TensorInfo<T, IndexType> Bias1,
            TensorInfo<T, IndexType> Bias2,
            TensorInfo<T, IndexType> _hx,
            TensorInfo<T, IndexType> _hy,
            TensorInfo<T, IndexType> storage,
            IndexType hsz,
            IndexType totalElements)
{
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x)
    {

      IndexType offset = (linearIndex/hsz)*3*hsz+linearIndex%hsz;

      T ir = DEVICE_LINEAR_GET(Input, offset+0*hsz);
      T ii = DEVICE_LINEAR_GET(Input, offset+1*hsz);
      T in = DEVICE_LINEAR_GET(Input, offset+2*hsz);
      T hr = DEVICE_LINEAR_GET(Hidden,offset+0*hsz);
      T hi = DEVICE_LINEAR_GET(Hidden,offset+1*hsz);
      T hn = DEVICE_LINEAR_GET(Hidden,  offset+2*hsz);

      T hx = DEVICE_LINEAR_GET(_hx, linearIndex);
      T* hy = &DEVICE_LINEAR_GET(_hy, linearIndex);

      bool has_bias = (Bias1.data != NULL);

      T b1r, b1i, b1n, b2r, b2i, b2n;

      if(has_bias) {
        b1r = DEVICE_LINEAR_GET(Bias1, linearIndex%hsz+0*hsz);
        b1i = DEVICE_LINEAR_GET(Bias1, linearIndex%hsz+1*hsz);
        b1n = DEVICE_LINEAR_GET(Bias1, linearIndex%hsz+2*hsz);

        b2r = DEVICE_LINEAR_GET(Bias2, linearIndex%hsz+0*hsz);
        b2i = DEVICE_LINEAR_GET(Bias2, linearIndex%hsz+1*hsz);
        b2n = DEVICE_LINEAR_GET(Bias2, linearIndex%hsz+2*hsz);
      } else {
        b1r = 0.0; b1i = 0.0; b1n = 0.0;
        b2r = 0.0; b2i = 0.0; b2n = 0.0;
      }

      offset = (linearIndex/hsz)*5*hsz+linearIndex%hsz;

      T rg, ig, ng;

      rg = ir + hr + b1r + b2r;
      ig = ii + hi + b1i + b2i;

      TensorSigmoidOp<real>()(&rg, &rg);
      TensorSigmoidOp<real>()(&ig, &ig);
      ng = in + b1n + rg * (hn + b2n);
      ng = math_generics::tanh(ng);
      *hy = ng + ig * (hx - ng);

      //SAVE FOR BACKWARDS
      DEVICE_LINEAR_GET(storage, offset+0*hsz) = rg;
      DEVICE_LINEAR_GET(storage, offset+1*hsz) = ig;
      DEVICE_LINEAR_GET(storage, offset+2*hsz) = ng;
      DEVICE_LINEAR_GET(storage, offset+3*hsz) = hx;
      DEVICE_LINEAR_GET(storage, offset+4*hsz) = hn + b2n;
    }
}

template <typename T, typename IndexType, int Dims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
THNN_(QuantizedGRUBackward)(TensorInfo<T, IndexType> gradInInput,
             TensorInfo<T, IndexType> gradInHidden,
             TensorInfo<T, IndexType> gradOutput,
             TensorInfo<T, IndexType> gradInputHx,
             TensorInfo<T, IndexType> storage,
             IndexType hsz,
             IndexType totalElements)
{
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType offset = (linearIndex/hsz)*5*hsz+linearIndex%hsz;

    T rg = DEVICE_LINEAR_GET(storage, offset+0*hsz);
    T ig = DEVICE_LINEAR_GET(storage, offset+1*hsz);
    T ng = DEVICE_LINEAR_GET(storage, offset+2*hsz);
    T hx = DEVICE_LINEAR_GET(storage, offset+3*hsz);
    T hn = DEVICE_LINEAR_GET(storage, offset+4*hsz);

    T go = DEVICE_LINEAR_GET(gradOutput, linearIndex);

    offset = (linearIndex/hsz)*3*hsz+linearIndex%hsz;

    T gig = go*(hx-ng)*(1-ig)*(ig);
    T ghx = go*(ig);
    T gin = go*(1-ig)*(1-ng*ng);
    T ghn = gin *rg;
    T grg = gin*hn*(1-rg)*rg;

    DEVICE_LINEAR_GET(gradInputHx, linearIndex) = ghx;
    DEVICE_LINEAR_GET(gradInInput, offset+0*hsz) = grg;
    DEVICE_LINEAR_GET(gradInInput, offset+1*hsz) = gig;
    DEVICE_LINEAR_GET(gradInInput, offset+2*hsz) = gin;
    DEVICE_LINEAR_GET(gradInHidden, offset+0*hsz) = grg;
    DEVICE_LINEAR_GET(gradInHidden, offset+1*hsz) = gig;
    DEVICE_LINEAR_GET(gradInHidden, offset+2*hsz) = ghn;
  }
}

template <typename T, typename IndexType, int Dims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
  THNN_(QuantizedLSTMForward)(TensorInfo<T, IndexType> input,
            TensorInfo<T, IndexType> hidden,
            TensorInfo<T, IndexType> bias1,
            TensorInfo<T, IndexType> bias2,
            TensorInfo<T, IndexType> _cx,
            TensorInfo<T, IndexType> _hy,
            TensorInfo<T, IndexType> _cy,
            TensorInfo<T, IndexType> quantizationBitWidth,
            IndexType hsz,
            IndexType totalElements)
{
    T quantization_bit_width = (T) DEVICE_LINEAR_GET(quantizationBitWidth, 0);

    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x)
    {

      IndexType offset = (linearIndex/hsz)*4*hsz+linearIndex%hsz;


      T* iig = &DEVICE_LINEAR_GET(input, offset+0*hsz);
      T* ifg = &DEVICE_LINEAR_GET(input, offset+1*hsz);
      T* icg = &DEVICE_LINEAR_GET(input, offset+2*hsz);
      T* iog = &DEVICE_LINEAR_GET(input, offset+3*hsz);

      T hig = DEVICE_LINEAR_GET(hidden, offset+0*hsz);
      T hfg = DEVICE_LINEAR_GET(hidden, offset+1*hsz);
      T hcg = DEVICE_LINEAR_GET(hidden,  offset+2*hsz);
      T hog = DEVICE_LINEAR_GET(hidden,  offset+3*hsz);

      T cx = DEVICE_LINEAR_GET(_cx, linearIndex);

      T* hy = &DEVICE_LINEAR_GET(_hy, linearIndex);
      T* cy = &DEVICE_LINEAR_GET(_cy, linearIndex);

      bool has_bias = (bias1.data != NULL);

      T b1i, b1f, b1c, b1o;
      T b2i, b2f, b2c, b2o;

      if(has_bias) {
        b1i = DEVICE_LINEAR_GET(bias1, linearIndex%hsz+0*hsz);
        b1f = DEVICE_LINEAR_GET(bias1, linearIndex%hsz+1*hsz);
        b1c = DEVICE_LINEAR_GET(bias1, linearIndex%hsz+2*hsz);
        b1o = DEVICE_LINEAR_GET(bias1, linearIndex%hsz+3*hsz);

        b2i = DEVICE_LINEAR_GET(bias2, linearIndex%hsz+0*hsz);
        b2f = DEVICE_LINEAR_GET(bias2, linearIndex%hsz+1*hsz);
        b2c = DEVICE_LINEAR_GET(bias2, linearIndex%hsz+2*hsz);
        b2o = DEVICE_LINEAR_GET(bias2, linearIndex%hsz+3*hsz);

      } else {
        b1i = 0.0; b1f = 0.0; b1c = 0.0; b1o = 0.0;
        b2i = 0.0; b2f = 0.0; b2c = 0.0; b2o = 0.0;
      }

      T ig, fg, cg, og;

      ig = *iig + hig + b1i + b2i;
      fg = *ifg + hfg + b1f + b2f;
      cg = *icg + hcg + b1c + b2c;
      og = *iog + hog + b1o + b2o;
  
      QuantizedTensorSigmoidOp<real>()(&ig, &ig, &quantization_bit_width);
      QuantizedTensorSigmoidOp<real>()(&fg, &fg, &quantization_bit_width);
      QuantizedTensorTanhOp<real>()(&cg, &cg, &quantization_bit_width);
      QuantizedTensorSigmoidOp<real>()(&og, &og, &quantization_bit_width);

      *cy = (fg * cx) + (ig * cg);
      T cy_tanhed;
      QuantizedTensorTanhOp<real>()(&cy_tanhed, cy, &quantization_bit_width);
      *hy = og * cy_tanhed;

      //SAVE FOR BACKWARDS
      //Also need cy and cx but can be saved easily in python
      *iig = ig;
      *ifg = fg;
      *icg = cg;
      *iog = og;
    }
}

template <typename T, typename IndexType, int Dims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
  THNN_(QuantizedLSTMBackward)(TensorInfo<T, IndexType> storage,
              TensorInfo<T, IndexType> gradInGates,
              TensorInfo<T, IndexType> _cx,
              TensorInfo<T, IndexType> _cy,
              TensorInfo<T, IndexType> gradoutput,
              TensorInfo<T, IndexType> gradoutputcell,
              TensorInfo<T, IndexType> gradInputCx,
              IndexType hsz,
              IndexType totalElements)
{
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType offset = (linearIndex/hsz)*4*hsz+linearIndex%hsz;

    T ig = DEVICE_LINEAR_GET(storage, offset+0*hsz);
    T fg = DEVICE_LINEAR_GET(storage, offset+1*hsz);
    T cg = DEVICE_LINEAR_GET(storage, offset+2*hsz);
    T og = DEVICE_LINEAR_GET(storage, offset+3*hsz);

    T* ih = &DEVICE_LINEAR_GET(gradInGates, offset+0*hsz);
    T* fh = &DEVICE_LINEAR_GET(gradInGates, offset+1*hsz);
    T* ch = &DEVICE_LINEAR_GET(gradInGates, offset+2*hsz);
    T* oh = &DEVICE_LINEAR_GET(gradInGates, offset+3*hsz);

    //will return hidden grads here
    T cx = DEVICE_LINEAR_GET(_cx, linearIndex);
    T cy = DEVICE_LINEAR_GET(_cy, linearIndex);

    T* gi = &DEVICE_LINEAR_GET(gradInputCx, linearIndex);

    T go = DEVICE_LINEAR_GET(gradoutput, linearIndex);
    T goc= DEVICE_LINEAR_GET(gradoutputcell, linearIndex);
    T gcx = math_generics::tanh(cy);

    T gog = go * gcx;
    gcx = go * og * ( 1 - gcx*gcx) + goc;

    T gig = gcx * cg;
    T gfg = gcx * cx;
    T gcg = gcx * ig;

    gcx = gcx * fg;

    gig = gig * (1-ig) * ig;
    gfg = gfg * (1-fg) * fg;
    gcg = gcg * (1-cg*cg);
    gog = gog * (1-og) * og;

    *ih = gig;
    *fh = gfg;
    *ch = gcg;
    *oh = gog;

    *gi = gcx;
  }
}

