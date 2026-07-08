import math

import torch
from torch import Tensor
import torch.nn as nn

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.bit_width.float import ComputeExponentBias
from brevitas.core.utils import StatelessBuffer
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.solver.common import ExponentBitWidthClass
from brevitas.quant.solver.common import MantissaBitWidthClass
from brevitas.utils.float_quant_utils import get_max_available_float
from brevitas.utils.python_utils import Registry
from brevitas_examples.common.generative.quantizers import BaseQuantizer
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY

MAX_AV_FP8 = get_max_available_float(4, 3, 2 ** (4 - 1) - 1, (('111',)), None, True)


def smooth_heaviside_stable(
    x: torch.Tensor,
    T=1.0,
    method: str = "sigmoid",
    theta: float | torch.Tensor = 0.5,
    eps_T: float = 1e-8,
    z_clip: float | None = None,
    grad_clip: float | None = 10.,
) -> torch.Tensor:
    """
    Numerically stabilized smooth approximation to the Heaviside step function H(x - theta).

    H_T(x) ≈ 0 for x << theta and ≈ 1 for x >> theta, with smoothness controlled by T > 0.
    Lower T makes the transition sharper; higher T makes it smoother.

    Args:
        x (torch.Tensor): Input tensor.
        T (float | torch.Tensor): Temperature parameter (> 0). Scalar or tensor
            broadcastable to x. Higher values yield smoother transitions.
        method (str): Which smooth approximation to use:
            - "sigmoid": H_T(x) = 0.5 * (1 + tanh((x-theta) / (2T)))
                         This uses the tanh formulation for sigmoid for improved stability.
            - "erf":    H_T(x) = 0.5 * (1 + erf((x-theta) / (sqrt(2) * T)))
            - "atan":   H_T(x) = 0.5 + (1/pi) * atan((x-theta)/T)
        theta (float | torch.Tensor): Threshold where the step occurs (default 0.0).
        eps_T (float): Small positive floor for T to avoid division by zero.
        z_clip (float | None): Optional clamp for z = (x - theta)/T to avoid extreme magnitudes,
            which can cause overflow in exp/tanh for very small T or large |x|.
            If None, no clamping is applied. A typical safe value is 40.0 for float32.
        grad_clip (float | None): If set, clips the gradient of the output to [-grad_clip, grad_clip]
            to prevent exploding gradients when T is very small.

    Returns:
        torch.Tensor: Tensor of same shape as x with values in [0, 1].

    Notes:
        - This function is differentiable and works with autograd.
        - The "sigmoid" method uses the tanh-based formulation for numerical stability:
            sigmoid(z) = 0.5 * (1 + tanh(z/2)), which avoids exp overflow/underflow.
        - For extreme T (very small or very large), using z_clip can prevent numerical issues
          while minimally affecting the output in saturated regions.
        - grad_clip applies to the backward pass only; it does not change the forward values.
    """
    # # Validate and prepare T and theta
    # if not isinstance(T, (int, float, torch.Tensor)):
    #     raise ValueError("T must be a float or torch.Tensor.")
    # if isinstance(T, (int, float)) and T <= 0:
    #     raise ValueError("T must be > 0.")
    # if isinstance(T, torch.Tensor) and torch.any(T <= 0):
    #     raise ValueError("All elements of T must be > 0.")

    x = torch.as_tensor(x)
    T = torch.as_tensor(T, dtype=x.dtype, device=x.device)
    theta = torch.as_tensor(theta, dtype=x.dtype, device=x.device)

    # Floor T to avoid division by zero (preserves broadcast compatibility)
    T = torch.clamp(T, min=eps_T)

    # Compute normalized argument
    z = (x - theta) * 20 * T

    # Optional clamping to avoid extreme values
    if z_clip is not None:
        z = torch.clamp(z, -float(z_clip), float(z_clip))

    # Compute smooth step
    if method == "sigmoid":
        # Numerically stable sigmoid via tanh formulation
        y = 0.5 * (1.0 + torch.tanh(z / 2.0))
    elif method == "erf":
        y = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    elif method == "atan":
        y = 0.5 + (1.0 / math.pi) * torch.atan(z)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'sigmoid', 'erf', or 'atan'.")

    # # Optional gradient clipping on the output
    # if grad_clip is not None and grad_clip > 0 and y.requires_grad:
    #     y.register_hook(lambda g: torch.clamp(g, -grad_clip, grad_clip))

    return y


class RestrictBitWidth(torch.nn.Module):

    def __init__(self, temperature, min_bit_width):
        super(RestrictBitWidth, self).__init__()
        self.temperature = StatelessBuffer(torch.tensor(temperature))
        self.min_bit_width = min_bit_width

    def restrict_init_float(self, x: float):
        return x

    def restrict_init_tensor(self, x: torch.Tensor):
        return x

    def restrict_init_module(self):
        return torch.nn.Identity()

    def restrict_init_inplace_module(self):
        return torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        x = x - self.min_bit_width
        if self.training:
            x = smooth_heaviside_stable(x / 2., T=self.temperature()) * 2.
        else:
            x = torch.where(x > 1, 2., 0.)
        x = x + self.min_bit_width
        return x


class LearnedMantissaParams(MantissaBitWidthClass):
    min_bit_width = 1
    mantissa_bit_width = 1
    restrict_bit_width_impl = RestrictBitWidth
    temperature = 0.4
    bit_width_min_val = 1
    bit_width_max_val = 3
    bit_width_offset_min_val = 1
    bit_width_offset_max_val = 3


class LearnedExponentParams(ExponentBitWidthClass):
    min_bit_width = 2
    exponent_bit_width = 2
    restrict_bit_width_impl = RestrictBitWidth
    temperature = 0.4
    bit_width_min_val = 2
    bit_width_max_val = 4
    bit_width_offset_min_val = 2
    bit_width_offset_max_val = 4


class MXFP4LearnedbitWeight(MXFloat8e4m3Weight):
    exponent_bit_width_impl_type = BitWidthImplType.PARAMETER
    mantissa_bit_width_impl_type = BitWidthImplType.PARAMETER
    scaling_impl_type = 'parameter_from_stats'
    bit_width = 4
    exponent_bit_width = 2
    mantissa_bit_width = 1
    exponent_bit_class = LearnedExponentParams
    mantissa_bit_class = LearnedMantissaParams
    max_available_float = MAX_AV_FP8
    exponent_bias_impl = ComputeExponentBias


class MXFP4LearnedbitAct(MXFloat8e4m3Act):
    exponent_bit_width_impl_type = BitWidthImplType.PARAMETER
    mantissa_bit_width_impl_type = BitWidthImplType.PARAMETER
    bit_width = 4
    exponent_bit_width = 2
    mantissa_bit_width = 1
    exponent_bit_class = LearnedExponentParams
    mantissa_bit_class = LearnedMantissaParams
    max_available_float = MAX_AV_FP8
    exponent_bias_impl = ComputeExponentBias


@Registry.register(QUANTIZERS_REGISTRY, "learned_float")
class LearnedFloat(BaseQuantizer):
    weight_quant = MXFP4LearnedbitWeight
    linear_input_quant = MXFP4LearnedbitAct


class MXFP6LearnedbitWeight(MXFloat8e4m3Weight):
    exponent_bit_width_impl_type = BitWidthImplType.CONST
    mantissa_bit_width_impl_type = BitWidthImplType.PARAMETER
    scaling_impl_type = 'parameter_from_stats'
    bit_width = 4
    exponent_bit_width = 2
    mantissa_bit_width = 1
    mantissa_bit_class = LearnedMantissaParams
    exponent_bias_impl = ComputeExponentBias


class MXFP6LearnedbitAct(MXFloat8e4m3Act):
    exponent_bit_width_impl_type = BitWidthImplType.CONST
    mantissa_bit_width_impl_type = BitWidthImplType.PARAMETER
    bit_width = 4
    exponent_bit_width = 2
    mantissa_bit_width = 1
    mantissa_bit_class = LearnedMantissaParams
    exponent_bias_impl = ComputeExponentBias


@Registry.register(QUANTIZERS_REGISTRY, "mxfp6_learned_float")
class MXFP6LearnedFloat(BaseQuantizer):
    weight_quant = MXFP6LearnedbitWeight
    linear_input_quant = MXFP6LearnedbitAct
