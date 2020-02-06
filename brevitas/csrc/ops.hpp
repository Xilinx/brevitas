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