import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
QONNX_ROOT = REPO_ROOT.parent / "qonnx" / "src"
if QONNX_ROOT.exists():
    sys.path.insert(0, str(QONNX_ROOT))

import numpy as np
import torch

from brevitas.export import export_qonnx
from brevitas.nn import QuantMultiheadAttention
from brevitas.quant_tensor import QuantTensor
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as onnx_exec
from qonnx.transformation.infer_shapes import InferShapes


def tensor_output(output):
    output = output[0]
    return output.value if isinstance(output, QuantTensor) else output


def verify_case(name, model, inputs):
    with torch.no_grad():
        reference = tensor_output(model(*inputs))

    original_names = [tensor.names for tensor in inputs]
    with tempfile.TemporaryDirectory(prefix=f"quant_mha_{name}_") as export_dir:
        export_path = Path(export_dir) / f"{name}.onnx"
        export_qonnx(model, args=inputs, export_path=str(export_path))

        assert [tensor.names for tensor in inputs] == original_names
        assert all(names == (None,) * tensor.dim() for tensor, names in zip(inputs, original_names))

        exported_model = ModelWrapper(str(export_path)).transform(InferShapes())
        input_names = [value.name for value in exported_model.graph.input]
        input_data = {
            input_name: tensor.detach().numpy()
            for input_name, tensor in zip(input_names, inputs)}
        output_data = onnx_exec.execute_onnx(exported_model, input_data, True)
        exported = output_data[exported_model.graph.output[0].name]

    max_abs_error = float(np.max(np.abs(exported - reference.numpy())))
    assert np.allclose(exported, reference.numpy(), atol=1e-5, rtol=1e-5)
    print(f"{name}: export succeeded, max_abs_error={max_abs_error:.3e}, output_shape={exported.shape}")


torch.manual_seed(123)
verify_case(
    "self_attention",
    QuantMultiheadAttention(embed_dim=8, num_heads=2).eval(),
    (torch.randn(3, 2, 8),) * 3)
verify_case(
    "cross_attention",
    QuantMultiheadAttention(
        embed_dim=8, num_heads=2, packed_in_proj=False, kdim=8, vdim=8).eval(),
    (torch.randn(3, 2, 8), torch.randn(4, 2, 8), torch.randn(4, 2, 8)))
print("QuantMultiheadAttention export regression: PASS")
