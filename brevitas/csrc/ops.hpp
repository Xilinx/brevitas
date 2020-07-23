//Copyright (c) 2020-     Xilinx (Giuseppe Franco)
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//
//* Redistributions of source code must retain the above copyright notice, this
//  list of conditions and the following disclaimer.
//
//* Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
//
//* Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include <ATen/TensorUtils.h>
#include <torch/extension.h>

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class RoundSteFn : public torch::autograd::Function<RoundSteFn> {
 public:
  static variable_list forward(
    AutogradContext* ctx,
    Variable input){
    return {at::round(input)};
   };

   static variable_list backward(
    AutogradContext* ctx,
    variable_list grad_output){
     return {grad_output[0]};
   }
};


class TensorClampSteFn : public torch::autograd::Function<TensorClampSteFn> {
 public:
  static variable_list forward(
    AutogradContext* ctx,
    Variable input,
    Variable min_val,
    Variable max_val){
    Variable output;
    output = at::where(input > max_val, max_val, input);
    output = at::where(output < min_val, min_val, output);
    return {output};
   };

   static variable_list backward(
    AutogradContext* ctx,
    variable_list grad_output){
     return {grad_output[0], Variable(), Variable()};
   }
};


class ScalarClampSteFn : public torch::autograd::Function<ScalarClampSteFn> {
 public:
  static variable_list forward(
    AutogradContext* ctx,
    Variable input,
    const double min_val,
    const double max_val){
    Variable output;
    output = at::clamp(input, min_val, max_val);
    return {output};
   };

   static variable_list backward(
    AutogradContext* ctx,
    variable_list grad_output){
     return {grad_output[0], Variable(), Variable()};
   }
};


class CeilSteFn : public torch::autograd::Function<CeilSteFn> {
 public:
  static variable_list forward(
    AutogradContext* ctx,
    Variable input){
    return {torch::ceil(input)};
   };

   static variable_list backward(
    AutogradContext* ctx,
    variable_list grad_output){
     return {grad_output[0]};
   }
};

class FloorSteFn : public torch::autograd::Function<FloorSteFn> {
 public:
  static variable_list forward(
    AutogradContext* ctx,
    Variable input){
    return {torch::floor(input)};
   };

   static variable_list backward(
    AutogradContext* ctx,
    variable_list grad_output){
     return {grad_output[0]};
   }
};

class BinarySignSteFn : public torch::autograd::Function<BinarySignSteFn> {
 public:
  static variable_list forward(
    AutogradContext* ctx,
    Variable input){
    Variable positive_mask = at::_cast_Float(at::ge(input, 0.0));
    Variable negative_mask = at::_cast_Float(at::lt(input, 0.0));
    Variable output = positive_mask - negative_mask;
    return{output};
    };

   static variable_list backward(
    AutogradContext* ctx,
    variable_list grad_output){
     return {grad_output[0]};
   }
};

class TernarySignSteFn : public torch::autograd::Function<TernarySignSteFn> {
 public:
  static variable_list forward(
    AutogradContext* ctx,
    Variable input){
    return{at::sign(input)};
    };

   static variable_list backward(
    AutogradContext* ctx,
    variable_list grad_output){
     return {grad_output[0]};
   }
};

class RoundToZeroFn : public torch::autograd::Function<RoundToZeroFn> {
 public:
  static variable_list forward(
    AutogradContext* ctx,
    Variable input){
    return {torch::sign(input) * torch::floor(torch::abs(input))};
   };

   static variable_list backward(
    AutogradContext* ctx,
    variable_list grad_output){
     return {grad_output[0]};
   }
};


Tensor ceil_ste(
 const Tensor& input){
 return CeilSteFn::apply(input)[0];
};

Tensor floor_ste(
 const Tensor& input){
 return FloorSteFn::apply(input)[0];
};

Tensor round_ste(
 const Tensor& input){
 return RoundSteFn::apply(input)[0];
};

Tensor binary_sign_ste(
 const Tensor& input){
 return BinarySignSteFn::apply(input)[0];
};

Tensor ternary_sign_ste(
 const Tensor& input){
 return TernarySignSteFn::apply(input)[0];
};

Tensor tensor_clamp_ste(
 const Tensor& input,
 const Tensor& min_val,
 const Tensor& max_val){
 return TensorClampSteFn::apply(input, min_val, max_val)[0];
};

Tensor scalar_clamp_ste(
 const Tensor& input,
 const double min_val,
 const double max_val){
 return ScalarClampSteFn::apply(input, min_val, max_val)[0];
};

Tensor round_to_zero(
 const Tensor& input){
 return RoundToZeroFn::apply(input)[0];
};