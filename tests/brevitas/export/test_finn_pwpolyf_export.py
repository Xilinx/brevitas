# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile

from packaging.version import parse
import pytest
import torch

from brevitas import torch_version
from brevitas.export import export_finn_onnx
from brevitas.export.onnx.finn.custom_ops import DOMAIN_STRING as FINN_PWPOLYF_DOMAIN
from brevitas.nn import QuantIdentity
from brevitas.nn.target.finn import PWPolyFActivation


def _export_pwpolyf(dynamo):
    onnx = pytest.importorskip("onnx")
    mod = PWPolyFActivation(func="gelu", K=3, degree=3)
    mod.eval()
    dummy = torch.randn(1, 8)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        export_path = f.name
    try:
        # Current Dynamo ONNX export lowers through opset >= 18 before any
        # optional version conversion. Keep the legacy path on opset 13 to
        # cover FINN's existing importer contract.
        opset_version = 18 if dynamo else 13
        export_finn_onnx(
            mod,
            dummy,
            export_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            dynamo=dynamo,
            optimize=True)
        return onnx.load(export_path)
    finally:
        os.unlink(export_path)
        try:
            os.unlink(f"{export_path}.data")
        except OSError:
            pass


def _get_pwpolyf_node(onnx_model):
    pwp_nodes = [n for n in onnx_model.graph.node if n.op_type == "PWPolyF"]
    assert len(pwp_nodes) == 1
    return pwp_nodes[0]


def _assert_pwpolyf_contract(node):
    assert node.domain == FINN_PWPOLYF_DOMAIN
    assert len(node.input) == 1
    attrs = {a.name: a for a in node.attribute}
    assert attrs["func"].s.decode("utf-8") == "gelu"
    assert attrs["K"].i == 3
    assert attrs["degree"].i == 3


def _assert_pwpolyf_opset(onnx_model):
    opsets = {opset.domain: opset.version for opset in onnx_model.opset_import}
    assert opsets[FINN_PWPOLYF_DOMAIN] == 1


def test_export_finn_onnx_pwpolyf_torchscript_marker():
    onnx_model = _export_pwpolyf(dynamo=False)
    _assert_pwpolyf_contract(_get_pwpolyf_node(onnx_model))
    _assert_pwpolyf_opset(onnx_model)


def test_export_finn_onnx_pwpolyf_after_quant_tensor_producer():
    onnx = pytest.importorskip("onnx")
    mod = torch.nn.Sequential(
        QuantIdentity(return_quant_tensor=True),
        PWPolyFActivation(func="gelu", K=3, degree=3))
    mod.eval()
    dummy = torch.randn(1, 8)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        export_path = f.name
    try:
        export_finn_onnx(
            mod,
            dummy,
            export_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamo=False)
        onnx_model = onnx.load(export_path)
    finally:
        os.unlink(export_path)
        try:
            os.unlink(f"{export_path}.data")
        except OSError:
            pass

    assert [node.op_type for node in onnx_model.graph.node] == ["Quant", "PWPolyF"]
    _assert_pwpolyf_contract(_get_pwpolyf_node(onnx_model))
    _assert_pwpolyf_opset(onnx_model)


@pytest.mark.skipif(parse("2.6") > torch_version, reason="Dynamo export requires PyTorch>=2.6")
def test_export_finn_onnx_pwpolyf_dynamo_marker():
    onnx_model = _export_pwpolyf(dynamo=True)
    _assert_pwpolyf_contract(_get_pwpolyf_node(onnx_model))
    _assert_pwpolyf_opset(onnx_model)
