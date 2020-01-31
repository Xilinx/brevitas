#include <ATen/TensorUtils.h>
#include <torch/extension.h>

at::Tensor SampleInput(const at::Tensor& input);
at::Tensor SampleInput_backward(const at::Tensor& grad);

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class Round_Ste_Class : public torch::autograd::Function<Round_Ste_Class> {
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


class Tensor_Clamp_Ste_Class : public torch::autograd::Function<Tensor_Clamp_Ste_Class> {
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


class Scalar_Clamp_Ste_Class : public torch::autograd::Function<Scalar_Clamp_Ste_Class> {
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


class Ceil_Ste_Class : public torch::autograd::Function<Ceil_Ste_Class> {
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

class Floor_Ste_Class : public torch::autograd::Function<Floor_Ste_Class> {
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

class Binary_Sign_Ste_Class : public torch::autograd::Function<Binary_Sign_Ste_Class> {
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

class Ternary_Sign_Ste_Class : public torch::autograd::Function<Ternary_Sign_Ste_Class> {
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
 return Round_Ste_Class::apply(input)[0];
};

Tensor floor_ste(
 const Tensor& input){
 return Round_Ste_Class::apply(input)[0];
};

Tensor round_ste(
 const Tensor& input){
 return Round_Ste_Class::apply(input)[0];
};

Tensor binary_sign_ste(
 const Tensor& input){
 return Binary_Sign_Ste_Class::apply(input)[0];
};

Tensor ternary_sign_ste(
 const Tensor& input){
 return Ternary_Sign_Ste_Class::apply(input)[0];
};

Tensor tensor_clamp_ste(
 const Tensor& input,
 const Tensor& min_val,
 const Tensor& max_val){
 return Tensor_Clamp_Ste_Class::apply(input, min_val, max_val)[0];
};

Tensor scalar_clamp_ste(
 const Tensor& input,
 const double min_val,
 const double max_val){
 return Scalar_Clamp_Ste_Class::apply(input, min_val, max_val)[0];
};