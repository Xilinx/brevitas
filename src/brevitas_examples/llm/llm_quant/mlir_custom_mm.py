from typing import List, Tuple

import torch
from torch.fx import wrap
import torch.utils.cpp_extension
import torch_mlir
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.registry import \
    _rename_python_keyword_parameter_name
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.registry import JitOperator
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.registry import SIG_ATTR_TYPE


def patched_has_value_semantics_function_signature(self):
    """Gets the Python function signature for this op's has_value_semantics function.
    While this is technically debug-only output, it is useful to copy-paste
    it from the debug dump into the library definitions, as many
    ops have extra default arguments and stuff that are tedious to write out
    right.
    """

    def parameter_decl_builder(arg: "SIG_ATTR_TYPE") -> str:
        parameter_name = _rename_python_keyword_parameter_name(arg["name"])
        return f"{parameter_name}"

    def ret_decl_builder(arg: "SIG_ATTR_TYPE") -> str:
        return "None"

    return self._get_function_signature(
        "has_value_semantics", parameter_decl_builder, ret_decl_builder)


JitOperator.get_has_value_semantics_function_signature = patched_has_value_semantics_function_signature


@wrap
def matmul_rhs_group_quant(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        rhs_scale: torch.Tensor,
        rhs_zero_point: torch.Tensor,
        rhs_bit_width: int,
        rhs_group_size: int):
    # This is just a placeholder for the actual implementation that provides correct shape/device/dtype
    return torch.randn(lhs.shape[0], rhs.shape[0], device=lhs.device, dtype=lhs.dtype)


brevitas_lib = torch.library.Library("brevitas", "DEF")
brevitas_lib.define(
    "matmul_rhs_group_quant(Tensor lhs, Tensor rhs, Tensor rhs_scale, Tensor rhs_zero_point, int rhs_bit_width, int rhs_group_size) -> Tensor"
)
brevitas_lib.impl("matmul_rhs_group_quant", matmul_rhs_group_quant)


def brevitas〇matmul_rhs_group_quant〡shape(
        lhs: List[int],
        rhs: List[int],
        rhs_scale: List[int],
        rhs_zero_point: List[int],
        rhs_bit_width: int,
        rhs_group_size: int) -> List[int]:
    return [lhs[0], rhs[0]]


def brevitas〇matmul_rhs_group_quant〡dtype(
        lhs_rank_dtype: Tuple[int, int],
        rhs_rank_dtype: Tuple[int, int],
        rhs_scale_rank_dtype: Tuple[int, int],
        rhs_zero_point_rank_dtype: Tuple[int, int],
        rhs_bit_width: int,
        rhs_group_size: int) -> int:
    # output dtype is the dtype of the lhs float input
    lhs_rank, lhs_dtype = lhs_rank_dtype
    return lhs_dtype


def brevitas〇matmul_rhs_group_quant〡has_value_semantics(
        lhs, rhs, rhs_scale, rhs_zero_point, rhs_bit_width, rhs_group_size) -> None:
    return


brevitas_matmul_rhs_group_quant_library = [
    brevitas〇matmul_rhs_group_quant〡shape,
    brevitas〇matmul_rhs_group_quant〡dtype,
    brevitas〇matmul_rhs_group_quant〡has_value_semantics]

if __name__ == '__main__':

    class CustomOpExampleModule(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(
                self: torch.nn.Module,
                lhs: torch.Tensor,
                rhs: torch.Tensor,
                rhs_scale: torch.Tensor,
                rhs_zero_point: torch.Tensor):
            return torch.ops.brevitas.matmul_rhs_group_quant(
                lhs, rhs, rhs_scale, rhs_zero_point, rhs_bit_width=8, rhs_group_size=128)

    mod = CustomOpExampleModule()
    mod.eval()

    module = torch_mlir.compile(
        mod, (torch.ones(3, 4), torch.ones(5, 4), torch.ones(1), torch.ones(1)),
        output_type="torch",
        backend_legal_ops=["brevitas.matmul_rhs_group_quant"],
        extra_library=brevitas_matmul_rhs_group_quant_library)
    print(module)
