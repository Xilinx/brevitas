# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

import pytest
import torch

from brevitas.export import export_finn_onnx
from brevitas.nn.target.finn import PWPolyFActivation


@pytest.mark.skipif("FINN_ROOT" not in os.environ, reason="FINN_ROOT is required")
def test_pwpolyf_export_converts_to_finn_hw_layer(tmp_path):
    pytest.importorskip("finn")
    pytest.importorskip("qonnx")

    try:
        from finn.transformation.fpgadataflow.convert_to_hw_layers import InferPWPolyFLayer
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.custom_op.registry import getCustomOp
        from qonnx.transformation.infer_datatypes import InferDataTypes
        from qonnx.transformation.infer_shapes import InferShapes
    except ImportError:
        pytest.skip("FINN PWPolyF conversion is not available")

    export_path = tmp_path / "pwpolyf.onnx"
    model = PWPolyFActivation(func="gelu", K=3, degree=3).eval()
    export_finn_onnx(
        model,
        torch.randn(1, 8),
        export_path,
        input_names=["inp"],
        output_names=["outp"],
        opset_version=13,
        dynamo=False)

    model = ModelWrapper(str(export_path))
    assert model.graph.node[0].op_type == "PWPolyF"
    assert model.graph.node[0].domain != "finn.custom_op.fpgadataflow"

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferPWPolyFLayer())

    node = model.graph.node[0]
    inst = getCustomOp(node)
    assert node.op_type == "PWPolyF"
    assert node.domain == "finn.custom_op.fpgadataflow"
    assert inst.get_nodeattr("func") == "gelu"
    assert inst.get_nodeattr("K") == 3
    assert inst.get_nodeattr("degree") == 3
