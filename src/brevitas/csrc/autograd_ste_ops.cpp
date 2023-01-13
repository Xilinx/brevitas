/* Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */


#include <ATen/TensorUtils.h>
#include <torch/extension.h>

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

class RoundSteFn : public torch::autograd::Function<RoundSteFn> {
 public:

  static Tensor forward(AutogradContext* ctx, Tensor input) {
    return at::round(input);
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
     return {grad_output[0]};
  }
};


class TensorClampSteFn : public torch::autograd::Function<TensorClampSteFn> {
 public:

  static Tensor forward(
    AutogradContext* ctx,
    Tensor input,
    Tensor min_val,
    Tensor max_val){
    Tensor output;
    output = at::where(input > max_val, max_val.type_as(input), input);
    output = at::where(output < min_val, min_val.type_as(output), output);
    return output;
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
     return {grad_output[0], Tensor(), Tensor()};
  }
};


class InplaceTensorClampSteFn : public torch::autograd::Function<InplaceTensorClampSteFn> {
 public:

  static Tensor forward(
    AutogradContext* ctx,
    Tensor input,
    Tensor min_val,
    Tensor max_val){
    at::min_out(input, input, max_val);
    at::max_out(input, input, min_val);
    return input;
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
     return {grad_output[0], Tensor(), Tensor()};
  }
};


class ScalarClampSteFn : public torch::autograd::Function<ScalarClampSteFn> {
 public:

  static Tensor forward(
    AutogradContext* ctx,
    Tensor input,
    const double min_val,
    const double max_val){
    Tensor output;
    output = at::clamp(input, min_val, max_val);
    return output;
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
     return {grad_output[0], Tensor(), Tensor()};
  }
};


class ScalarClampMinSteFn : public torch::autograd::Function<ScalarClampMinSteFn> {
 public:

  static Tensor forward(AutogradContext* ctx, Tensor input, const double min_val) {
    Tensor output;
    output = at::clamp_min(input, min_val);
    return output;
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
    return {grad_output[0], Tensor()};
  }
};


class CeilSteFn : public torch::autograd::Function<CeilSteFn> {
 public:

  static Tensor forward(AutogradContext* ctx, Tensor input) {
    return torch::ceil(input);
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
    return {grad_output[0]};
  }
};

class FloorSteFn : public torch::autograd::Function<FloorSteFn> {
 public:

  static Tensor forward(AutogradContext* ctx, Tensor input) {
    return torch::floor(input);
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
    return {grad_output[0]};
  }
};

class BinarySignSteFn : public torch::autograd::Function<BinarySignSteFn> {
 public:

  static Tensor forward(AutogradContext* ctx, Tensor input) {
    Tensor positive_mask = at::ge(input, 0.0);
    Tensor negative_mask = at::lt(input, 0.0);
    Tensor output = positive_mask.type_as(input) - negative_mask.type_as(input);
    return output;
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
    return {grad_output[0]};
  }
};


class TernarySignSteFn : public torch::autograd::Function<TernarySignSteFn> {
 public:

  static Tensor forward(AutogradContext* ctx, Tensor input) {
    return at::sign(input);
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
    return {grad_output[0]};
  }
};


class RoundToZeroSteFn : public torch::autograd::Function<RoundToZeroSteFn> {
 public:

  static Tensor forward(AutogradContext* ctx, Tensor input) {
    return torch::sign(input) * torch::floor(torch::abs(input));
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
    return {grad_output[0]};
  }
};


class DPURoundSteFn : public torch::autograd::Function<DPURoundSteFn> {
 public:

  static Tensor forward(AutogradContext* ctx, Tensor input) {
    Tensor output;
    output = at::where(
        at::logical_and((input < 0.), (input - torch::floor(input) == 0.5)), torch::ceil(input), torch::round(input));
    return output;
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
    return {grad_output[0]};
  }
};


class AbsBinarySignGradFn : public torch::autograd::Function<AbsBinarySignGradFn> {
 public:

  static Tensor forward(AutogradContext* ctx, Tensor input) {
    ctx->save_for_backward({input});
    return torch::abs(input);
  };

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_output) {
    Tensor input = ctx->get_saved_variables()[0];
    return {BinarySignSteFn::apply(input) * grad_output[0]};
  }
};


Tensor ceil_ste_impl(const Tensor& input) {
 return CeilSteFn::apply(input);
};


Tensor floor_ste_impl(const Tensor& input) {
 return FloorSteFn::apply(input);
};


Tensor round_ste_impl(const Tensor& input) {
 return RoundSteFn::apply(input);
};


Tensor dpu_round_ste_impl(const Tensor& input) {
 return DPURoundSteFn::apply(input);
};


Tensor binary_sign_ste_impl(const Tensor& input) {
 return BinarySignSteFn::apply(input);
};


Tensor ternary_sign_ste_impl(const Tensor& input) {
 return TernarySignSteFn::apply(input);
};


Tensor tensor_clamp_ste_impl_(const Tensor& input, const Tensor& min_val, const Tensor& max_val) {
 return InplaceTensorClampSteFn::apply(input, min_val, max_val);
};


Tensor tensor_clamp_ste_impl(const Tensor& input, const Tensor& min_val, const Tensor& max_val) {
 return TensorClampSteFn::apply(input, min_val, max_val);
};


Tensor scalar_clamp_ste_impl(const Tensor& input, const double min_val, const double max_val) {
 return ScalarClampSteFn::apply(input, min_val, max_val);
};


Tensor scalar_clamp_min_ste_impl(const Tensor& input, const double min_val) {
 return ScalarClampMinSteFn::apply(input, min_val);
};


Tensor round_to_zero_ste_impl(const Tensor& input) {
 return RoundToZeroSteFn::apply(input);
};


Tensor abs_binary_sign_grad_impl(const Tensor& input) {
 return AbsBinarySignGradFn::apply(input);
};



TORCH_LIBRARY(autograd_ste_ops, m) {
    m.def("round_ste_impl", &round_ste_impl);
    m.def("tensor_clamp_ste_impl", &tensor_clamp_ste_impl);
    m.def("tensor_clamp_ste_impl_", &tensor_clamp_ste_impl);
    m.def("scalar_clamp_ste_impl", &scalar_clamp_ste_impl);
    m.def("scalar_clamp_min_ste_impl", &scalar_clamp_min_ste_impl);
    m.def("binary_sign_ste_impl", &binary_sign_ste_impl);
    m.def("ternary_sign_ste_impl", &ternary_sign_ste_impl);
    m.def("ceil_ste_impl", &ceil_ste_impl);
    m.def("floor_ste_impl", &floor_ste_impl);
    m.def("round_to_zero_ste_impl", &round_to_zero_ste_impl);
    m.def("dpu_round_ste_impl", &dpu_round_ste_impl);
    m.def("abs_binary_sign_grad_impl", &abs_binary_sign_grad_impl);
}
