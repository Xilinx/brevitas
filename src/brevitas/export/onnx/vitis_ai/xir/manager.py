from typing import Optional, Tuple, Union

from packaging import version
from torch.nn import Module
from torch import Tensor
import numpy as np

try:
    from onnx import numpy_helper
    from onnx import TensorProto
except:
    numpy_helper = None
    TensorProto = None

try:
    import xir
except:
    xir = None

from brevitas.quant_tensor import QuantTensor
from brevitas import torch_version
from ..manager import VitisAIManager
from .handler import XIRQuantLinearHandler, XIRQuantConv2dHandler, XIRQuantConvTranspose2dHandler
from .handler import XIRQuantReLUHandler, XIRQuantIdentityHandler
from .function import XIRFixFn, XIRGemmFn, XIRConv2dFn, XIRConvTranpose2dFn

_nchw_dim_to_name = {
    0: 'n',
    1: 'c',
    2: 'h',
    3: 'w'
}


_nhwc_name_to_dim = {
    'n': 0,
    'h': 1,
    'w': 2,
    'c': 3
}


_nchw_to_nhwc_dims = {
    k: _nhwc_name_to_dim[v] for k, v in _nchw_dim_to_name.items()}


def _onnx_attr(node, name, default):
    for attr in node.attribute:
        if name == attr.name:
            return attr
    if default is None:
        raise RuntimeError(f"Attribute with name {name} not found.")
    else:
        return default


def _onnx_str_attr(node, name, default=None):
    attr = _onnx_attr(node, name, default)
    if isinstance(attr, str):
        return attr
    else:
        return str(attr.s, 'utf-8')


def _onnx_bool_attr(node, name, default=None):
    attr = _onnx_attr(node, name, default)
    if isinstance(attr, bool):
        return attr
    elif isinstance(attr, int):
        return bool(attr)
    else:
        return bool(attr.i)


def _onnx_int_attr(node, name, default=None):
    attr = _onnx_attr(node, name, default)
    if isinstance(attr, int):
        return attr
    else:
        return attr.i


def _onnx_ints_attr(node, name, default=None):
    attr = _onnx_attr(node, name, default)
    if isinstance(attr, list) and [isinstance(i, int) for i in attr]:
        return attr
    else:
        return [i for i in attr.ints]


def _onnx_float_attr(node, name, default=None):
    attr = _onnx_attr(node, name, default)
    if isinstance(attr, float):
        return attr
    else:
        return attr.f


def _onnx_array_attr(node, name, default=None):
    attr = _onnx_attr(node, name, default)
    if isinstance(attr, np.ndarray):
        return attr
    else:
        array = numpy_helper.to_array(attr.t)
        array = np.atleast_1d(array)  # convert any scalar
        return array


def _xir_const_op_to_array(const_op):
    flat_array = np.frombuffer(const_op.get_attr('data'), dtype=const_op.get_attr('data_type'))
    return flat_array.reshape(const_op.get_attr('shape'))


def _onnx_tensor_shape(tensor):
    return [d.dim_value for d in tensor.type.tensor_type.shape.dim]


def _nchw_to_nhwc_shape(shape):
    assert len(shape) == 4, '4-dim shape required'
    return [shape[0], shape[2], shape[3], shape[1]]


def _onnx_tensor_dtype(tensor):
    dtype = TensorProto.DataType.Name(tensor.type.tensor_type.elem_type)
    if dtype == 'FLOAT':
        return 'FLOAT32'
    else:
        return dtype

def otx_constant(xir_graph, node):
    for output_name in node.output:
        value = _onnx_array_attr(node, 'value')
        xir_graph.create_const_op(output_name, value)


def otx_fix_op(xir_graph, onnx_node):
    assert len(onnx_node.input) == 1, 'Fix op can have only one input'
    assert len(onnx_node.output) == 1, 'Fix op can have only one output'
    input_op = xir_graph.get_op(onnx_node.input[0])
    attrs = {}
    attrs['bit_width'] = _onnx_int_attr(onnx_node, 'bit_width')
    attrs['fix_point'] = _onnx_int_attr(onnx_node, 'fix_point')
    attrs['if_signed'] = _onnx_bool_attr(onnx_node, 'signed')
    attrs['round_mode'] = "DPU_ROUND"  # TODO this should taken from ONNX
    input_dict = {'input': [input_op]}
    return xir_graph.create_op(onnx_node.output[0], 'fix', attrs, input_dict)


def otx_gemm_op(xir_graph, onnx_node):
    assert len(onnx_node.output) == 1, f'{onnx_node.type} op can have only one output'
    attrs = {
        'transpose_a': _onnx_bool_attr(onnx_node, 'transA', False),
        'transpose_b': _onnx_bool_attr(onnx_node, 'transB', True)}
    input_dict = {'input': [xir_graph.get_op(i) for i in onnx_node.input[:2]]}
    if len(onnx_node.input) == 3:
        bias_op = xir_graph.get_op(onnx_node.input[2])
        # remove spurious zero bias
        if bias_op.get_type() == 'const' and not _xir_const_op_to_array(bias_op).any():
            xir_graph.remove_op(bias_op)
        else:
            input_dict['bias'] = [bias_op]
    output_name = onnx_node.output[0]
    xir_graph.create_op(name=output_name, kind='matmul', attrs=attrs, input_ops=input_dict)


def _otx_conv_based_op(xir_graph, onnx_node, xir_kind):
    pad_mode = _onnx_str_attr(onnx_node, 'padding_type', 'FLOOR').upper()
    pad_mode = 'FLOOR' if pad_mode == 'STANDARD' else pad_mode
    attrs = {
        'kernel': _onnx_ints_attr(onnx_node, 'kernel_shape'),
        'stride': _onnx_ints_attr(onnx_node, 'strides'),
        'pad': _onnx_ints_attr(onnx_node, 'pads'),
        'pad_mode': pad_mode,
        'dilation': _onnx_ints_attr(onnx_node, 'dilations')}
    input_dict = {
        'input': [xir_graph.get_op(onnx_node.input[0])],
        'weights': [xir_graph.get_op(onnx_node.input[1])]}
    if len(onnx_node.input) == 3:
        input_dict['bias'] = [xir_graph.get_op(onnx_node.input[2])]
    output_name = onnx_node.output[0]
    return xir_graph.create_op(name=output_name, kind=xir_kind, attrs=attrs, input_ops=input_dict)


def otx_conv2d_op(xir_graph, onnx_node):
    return _otx_conv_based_op(xir_graph, onnx_node, 'conv2d')


def otx_depthwise_conv2d_op(xir_graph, onnx_node):
    return _otx_conv_based_op(xir_graph, onnx_node, 'depthwise-conv2d')


def otx_conv_transpose2d_op(xir_graph, onnx_node):
    return _otx_conv_based_op(xir_graph, onnx_node, 'transposed-conv2d')


def otx_depthwise_conv_tranpose2d_op(xir_graph, onnx_node):
    return _otx_conv_based_op(xir_graph, onnx_node, 'transposed-depthwise-conv2d')


def otx_max_pool2d_op(xir_graph, onnx_node):
    input_dict = {'input': [xir_graph.get_op(onnx_node.input[0])]}
    attrs = {
        'kernel': _onnx_ints_attr(onnx_node, 'kernel_shape'),
        'stride': _onnx_ints_attr(onnx_node, 'strides'),
        'pad_mode': 'FLOOR',  # onnx export converts pads to work in floor mode
        'pad': _onnx_ints_attr(onnx_node, 'pads'),
        'global': False}
    return xir_graph.create_op(
        name=onnx_node.output[0], kind='maxpool2d', attrs=attrs, input_ops=input_dict)


def _otx_nary_op(xir_graph, onnx_node, xir_kind, xir_attrs):
    assert len(onnx_node.output) == 1, f'Only one output supported for {xir_kind}'
    input_dict = {'input': [xir_graph.get_op(i) for i in onnx_node.input]}
    return xir_graph.create_op(
        name=onnx_node.output[0], kind=xir_kind, attrs=xir_attrs, input_ops=input_dict)


def otx_relu_op(xir_graph, onnx_node):
    return _otx_nary_op(xir_graph, onnx_node, 'relu', {})


def otx_relu6_op(xir_graph, onnx_node):
    return _otx_nary_op(xir_graph, onnx_node, 'relu6', {})


def otx_sigmoid_op(xir_graph, onnx_node):
    return _otx_nary_op(xir_graph, onnx_node, 'sigmoid', {})


def otx_tanh_op(xir_graph, onnx_node):
    return _otx_nary_op(xir_graph, onnx_node, 'tanh', {})


def otx_add_op(xir_graph, onnx_node):
    return _otx_nary_op(xir_graph, onnx_node, 'add', {})


def otx_sub_op(xir_graph, onnx_node):
    return _otx_nary_op(xir_graph, onnx_node, 'sub', {})


def otx_reshape_op(xir_graph, onnx_node):
    shape_op = xir_graph.get_op(onnx_node.input[1])
    # convert from nchw to nhwc, this approach is very fragile
    # but we don't really have alternatives
    if shape_op.get_attr('shape') == [4]:
        shape = _xir_const_op_to_array(shape_op)
        shape = _nchw_to_nhwc_shape(shape)
        name = shape_op.get_name()
        # since we can't modify data, delete the const op
        # and recreate it under the same name
        xir_graph.remove_op(shape_op)
        shape_op = xir_graph.create_const_op(name, np.array(shape))
    input_dict = {
        'input': [xir_graph.get_op(onnx_node.input[0])],
        'shape': [shape_op]}
    return xir_graph.create_op(
        name=onnx_node.output[0], kind='reshape', input_ops=input_dict)


def otx_transpose_op(xir_graph, onnx_node):
    perm = _onnx_ints_attr(onnx_node, 'perm')
    # convert from nchw to nhwc
    if len(perm) == 4:
        start_names = _nchw_dim_to_name.values()
        end_names = [_nchw_dim_to_name[d] for d in perm]
        perm_name_map = {k: v for k, v in zip(start_names, end_names)}
        perm_nhwc_dims_map = {
            _nhwc_name_to_dim[k]: _nhwc_name_to_dim[v] for k, v in perm_name_map.items()}
        perm = list(dict(sorted(perm_nhwc_dims_map.items())).values())
    input_dict = {'input': [xir_graph.get_op(onnx_node.input[0])]}
    attrs = {'order': perm}
    return xir_graph.create_op(
        name=onnx_node.output[0], kind='transpose', attrs=attrs, input_ops=input_dict)


def otx_concat_op(xir_graph, onnx_node):
    return _otx_axis_nary_op(xir_graph, onnx_node, 'concat')


def otx_stack_op(xir_graph, onnx_node):
    return _otx_axis_nary_op(xir_graph, onnx_node, 'stack')


def _otx_axis_nary_op(xir_graph, onnx_node, xir_kind):
    inp_list = [xir_graph.get_op(i) for i in onnx_node.input]
    axis = _onnx_int_attr(onnx_node, 'axis')
    # get input shape
    for i in inp_list:
        i.infer_shape()
    inp_shape_list = [i.get_output_tensor().dims for i in inp_list]
    assert all([len(s) == len(inp_shape_list[0]) for s in inp_shape_list])
    if len(inp_shape_list[0]) == 4:
        axis = _nchw_to_nhwc_dims[axis]
    input_dict = {'input': inp_list}
    attrs = {'axis': axis}
    return xir_graph.create_op(
        name=onnx_node.output[0], kind=xir_kind, attrs=attrs, input_ops=input_dict)


class XIRManager(VitisAIManager):
    target_name = 'XIR'

    handlers = [
        XIRQuantLinearHandler,
        XIRQuantConv2dHandler,
        XIRQuantConvTranspose2dHandler,
        XIRQuantIdentityHandler,
        XIRQuantReLUHandler]

    otx_converters = {
        'Reshape': otx_reshape_op,
        'Transpose': otx_transpose_op,
        'Concat': otx_concat_op,
        'Stack': otx_stack_op,
        'Constant': otx_constant,
        'Fix': otx_fix_op,
        'Add': otx_add_op,
        'Sub': otx_sub_op,
        'Gemm': otx_gemm_op,
        'Conv2d': otx_conv2d_op,
        'DepthwiseConv2d': otx_depthwise_conv2d_op,
        'ConvTranspose2d': otx_conv_transpose2d_op,
        'DepthwiseConvTranspose2d': otx_depthwise_conv_tranpose2d_op,
        'MaxPool': otx_max_pool2d_op,
        'Relu': otx_relu_op,
        'Relu6': otx_relu6_op,
        'Sigmoid': otx_sigmoid_op,
        'Tanh': otx_tanh_op}

    onnx_passes = [
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    custom_fns = [
        XIRConv2dFn,
        XIRConvTranpose2dFn,
        XIRFixFn,
        XIRGemmFn
    ]

    @classmethod
    def solve_keep_initializers_as_inputs(cls, export_kwargs):
        ka = 'keep_initializers_as_inputs'
        if torch_version >= version.parse('1.3.0') and ka not in export_kwargs:
            export_kwargs[ka] = False

    @classmethod
    def export(
            cls,
            module: Module,
            input_shape: Optional[Tuple[int, ...]] = None,
            export_path: Optional[str] = None,
            input_t: Optional[Union[Tensor, QuantTensor]] = None,
            export_debug_onnx_file: bool = False,
            **kwargs):
        if xir is None:
            raise ModuleNotFoundError("XIR export flow requires xir to be installed.")

        if export_debug_onnx_file:
            onnx_export_path = export_path + '_debug.onnx'
        else:
            onnx_export_path = None
        model = cls.export_onnx(module, input_shape, onnx_export_path, input_t, **kwargs)
        xir_graph = xir.Graph(model.graph.name)
        for inp in model.graph.input:
            shape = _onnx_tensor_shape(inp)
            if len(shape) == 4:
                shape = _nchw_to_nhwc_shape(shape)
            xir_graph.create_op(
                name=inp.name,
                kind='data',
                attrs={'shape': shape, 'data_type': _onnx_tensor_dtype(inp)})
        for node in model.graph.node:
            cls.otx_converters[node.op_type](xir_graph, node)
        if export_path is not None:
            xir_graph.serialize(export_path)
        return xir_graph