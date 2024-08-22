from typing import Any, List, Union

from torch import Tensor

from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjectorBase
from brevitas.quant_tensor import GroupwiseFloatQuantTensor


class GroupwiseWeightFloatQuantProxyFromInjector(WeightFloatQuantProxyFromInjectorBase):

    @property
    def group_dim(self):
        return self.quant_injector.group_dim

    @property
    def group_size(self):
        return self.quant_injector.group_size

    def create_quant_tensor(self, qt_args: List[Any]) -> Union[Tensor, GroupwiseFloatQuantTensor]:
        out, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values = qt_args
        return GroupwiseFloatQuantTensor(
            out,
            scale,
            zero_point,
            self.group_size,
            self.group_dim,
            exponent_bit_width,
            mantissa_bit_width,
            exponent_bias,
            saturating,
            inf_values,
            nan_values,
            self.is_signed,
            self.training)
