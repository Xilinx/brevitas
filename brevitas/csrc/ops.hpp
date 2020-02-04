#include <ATen/TensorUtils.h>
#include <torch/extension.h>

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class RoundSteClass : public torch::autograd::Function<RoundSteClass> {
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


class TensorClampSteClass : public torch::autograd::Function<TensorClampSteClass> {
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


class ScalarClampSteClass : public torch::autograd::Function<ScalarClampSteClass> {
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


class CeilSteClass : public torch::autograd::Function<CeilSteClass> {
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

class FloorSteClass : public torch::autograd::Function<FloorSteClass> {
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

class BinarySignSteClass : public torch::autograd::Function<BinarySignSteClass> {
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

class TernarySignSteClass : public torch::autograd::Function<TernarySignSteClass> {
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
 return CeilSteClass::apply(input)[0];
};

Tensor floor_ste(
 const Tensor& input){
 return FloorSteClass::apply(input)[0];
};

Tensor round_ste(
 const Tensor& input){
 return RoundSteClass::apply(input)[0];
};

Tensor binary_sign_ste(
 const Tensor& input){
 return BinarySignSteClass::apply(input)[0];
};

Tensor ternary_sign_ste(
 const Tensor& input){
 return TernarySignSteClass::apply(input)[0];
};

Tensor tensor_clamp_ste(
 const Tensor& input,
 const Tensor& min_val,
 const Tensor& max_val){
 return TensorClampSteClass::apply(input, min_val, max_val)[0];
};

Tensor scalar_clamp_ste(
 const Tensor& input,
 const double min_val,
 const double max_val){
 return ScalarClampSteClass::apply(input, min_val, max_val)[0];
};