"""Shared helpers for the Brevitas workshop notebook.

The main feature is a lightweight ASCII network renderer that draws the
floating-point baseline and its quantized variants without pulling in Netron,
Graphviz, or torchview. Netron shows the exported QDQ/QONNX graph, which mixes in
concepts the tutorial has not introduced yet, so ``draw_network`` renders the
PyTorch model directly instead.

Two levels of detail are provided:

- ``mode="compact"`` summarizes each quantized layer with one weight and one
  activation line. It suits the fixed-format integer workflow.
- ``mode="detailed"`` reports the per-channel minifloat format (``ExMx``) for
  weights and the per-tensor format for activations. It suits the learned
  minifloat workflow. Weights show the first few channel formats followed by
  ``...`` when there are more.

Two orientations are provided:

- ``orientation="vertical"`` stacks the layers top to bottom.
- ``orientation="horizontal"`` reads left to right, which matches the MLP and
  leaves room to draw a skip lane above the main path.

For a plain sequential model the renderer walks ``named_children`` in definition
order. When FX tracing is available it also detects residual merges (``add`` or
``cat`` nodes with two inputs) and draws the skip branch as a separate lane. It
reads quantizer metadata through the public Brevitas tensor/quantizer API and
never executes an exported graph.

The module also hosts a few small workflow and inspection helpers used by the
notebook: parameter snapshots for PTQ/QAT, a rounding-gradient plot, and a
utility that reports where a quantizer attribute is defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import inspect

import torch

import brevitas.nn as qnn

# Number of per-output-channel weight formats shown before truncating with "...".
_MAX_WEIGHT_CHANNELS = 3


@dataclass
class _Block:
    """One rendered layer: a header plus indented detail lines."""

    header: str
    details: list = field(default_factory=list)

    def rows(self) -> list:
        return [self.header] + [f'  {detail}' for detail in self.details]

    def width(self) -> int:
        return max((len(row) for row in self.rows()), default=0)


def _format_minifloat(exponent_bit_width, mantissa_bit_width) -> str:
    """Return an ``ExMx`` label from scalar exponent/mantissa bit-widths."""
    exponent = int(round(float(exponent_bit_width)))
    mantissa = int(round(float(mantissa_bit_width)))
    return f'E{exponent}M{mantissa}'


def _as_flat_list(tensor) -> list:
    """Flatten a tensor to a Python list, tolerating Python scalars."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().flatten().tolist()
    return [tensor]


def _is_float_quant(quant_tensor) -> bool:
    """A minifloat quant-tensor exposes exponent and mantissa bit-widths."""
    return hasattr(quant_tensor, 'exponent_bit_width') and hasattr(
        quant_tensor, 'mantissa_bit_width')


def _weight_detail_lines(module, detailed: bool) -> list:
    """Describe the weight quantizer of a ``QuantLinear`` layer."""
    quant_weight = module.quant_weight()
    scale_shape = tuple(quant_weight.scale.shape)
    per_channel = any(dimension > 1 for dimension in scale_shape)
    granularity = 'per-output-channel' if per_channel else 'per-tensor'

    if not detailed:
        if _is_float_quant(quant_weight):
            exponents = _as_flat_list(quant_weight.exponent_bit_width)
            mantissas = _as_flat_list(quant_weight.mantissa_bit_width)
            fmt = _format_minifloat(exponents[0], mantissas[0])
            return [f'weight: {fmt}, {granularity}']
        bit_width = int(round(float(quant_weight.bit_width)))
        sign = 'INT' if bool(quant_weight.signed) else 'UINT'
        return [f'weight: {sign}{bit_width}, {granularity}']

    # Detailed mode reports one ExMx label per output channel.
    exponents = _as_flat_list(quant_weight.exponent_bit_width)
    mantissas = _as_flat_list(quant_weight.mantissa_bit_width)
    channel_count = max(len(exponents), len(mantissas))
    if len(exponents) == 1:
        exponents = exponents * channel_count
    if len(mantissas) == 1:
        mantissas = mantissas * channel_count

    formats = [
        _format_minifloat(exponents[channel], mantissas[channel])
        for channel in range(min(channel_count, _MAX_WEIGHT_CHANNELS))]
    if channel_count > _MAX_WEIGHT_CHANNELS:
        formats.append('...')
    return [f'weights: {", ".join(formats)}']


def _activation_detail_lines(module, detailed: bool) -> list:
    """Describe the activation quantizer of an activation layer."""
    if not detailed:
        bit_width = int(round(float(module.act_quant.bit_width())))
        sign = 'INT' if module.act_quant.is_signed else 'UINT'
        return [f'activation: {sign}{bit_width}, per-tensor']

    exponent = module.act_quant.exponent_bit_width()
    mantissa = module.act_quant.mantissa_bit_width()
    fmt = _format_minifloat(exponent, mantissa)
    return [f'activation: {fmt}']


def _input_quant_lines(module, detailed: bool) -> list:
    """Describe the input activation quantizer of a ``QuantLinear`` layer.

    Layerwise conversion places the model input and inter-layer activation
    boundaries inside each layer's ``input_quant`` proxy rather than in a
    separate activation module, so without this the diagram would hide them.
    """
    input_quant = getattr(module, 'input_quant', None)
    if input_quant is None or not input_quant.is_quant_enabled:
        return []
    if detailed:
        fmt = _format_minifloat(input_quant.exponent_bit_width(), input_quant.mantissa_bit_width())
        return [f'input: {fmt}']
    bit_width = int(round(float(input_quant.bit_width())))
    sign = 'INT' if input_quant.is_signed else 'UINT'
    return [f'input: {sign}{bit_width}, per-tensor']


def _rounding_name(module) -> str | None:
    """Return the weight rounding module class name, when available."""
    float_to_int = getattr(module.weight_quant.tensor_quant, 'float_to_int_impl', None)
    if float_to_int is None:
        return None
    return type(float_to_int).__name__


def _describe_module(name, module, detailed: bool) -> _Block | None:
    """Build a rendered block for one module, or ``None`` to skip it."""
    if isinstance(module, qnn.QuantLinear):
        header = f'{name}: QuantLinear {module.in_features} -> {module.out_features}'
        details = _input_quant_lines(module, detailed)
        details += _weight_detail_lines(module, detailed)
        if detailed:
            rounding = _rounding_name(module)
            if rounding is not None:
                details.append(f'rounding: {rounding}')
        return _Block(header, details)

    if isinstance(module, (qnn.QuantIdentity, qnn.QuantReLU)):
        kind = type(module).__name__
        header = f'{name}: {kind}'
        return _Block(header, _activation_detail_lines(module, detailed))

    if isinstance(module, torch.nn.Linear):
        header = f'{name}: Linear {module.in_features} -> {module.out_features}'
        return _Block(header, ['weight: FP32'])

    if isinstance(module, torch.nn.ReLU):
        return _Block(f'{name}: ReLU', [])

    return None


def _sequential_blocks(model, detailed: bool) -> list:
    """Collect rendered blocks from ``named_children`` in definition order."""
    blocks = []
    for name, module in model.named_children():
        rendered = _describe_module(name, module, detailed)
        if rendered is not None:
            blocks.append(rendered)
    return blocks


def _render_vertical(blocks, skips, title):
    """Stack blocks top to bottom. ``skips`` is unused in vertical layout."""
    if title:
        print(title)
        print('=' * len(title))
    print('Input')
    for block in blocks:
        print('  |')
        for row in block.rows():
            print(row)
    print('  |')
    print('Output')


def _render_horizontal(blocks, skips, title):
    """Lay blocks out left to right, drawing any skip lanes above the path.

    ``skips`` maps a (source_index, destination_index) pair, both referring to
    positions in ``blocks``, to the merge label (for example ``"add"``). The
    skip lane is drawn as a bracket above the involved blocks.
    """
    if title:
        print(title)
        print('=' * len(title))

    labelled = [_Block('Input', [])] + blocks + [_Block('Output', [])]
    heights = max(len(block.rows()) for block in labelled)
    widths = [block.width() for block in labelled]

    # Vertically center each block's rows within the shared band.
    grids = []
    for block in labelled:
        rows = block.rows()
        pad_top = (heights - len(rows)) // 2
        padded = [''] * pad_top + rows + [''] * (heights - len(rows) - pad_top)
        grids.append([row.ljust(block.width()) for row in padded])

    connector = ' --> '
    skip_lane = _skip_lane(labelled, widths, skips, len(connector)) if skips else None

    if skip_lane is not None:
        print(skip_lane)
    for band_row in range(heights):
        pieces = []
        for column, grid in enumerate(grids):
            pieces.append(grid[band_row])
        arrow = connector if band_row == heights // 2 else ' ' * len(connector)
        print(arrow.join(pieces))


def _skip_lane(labelled, widths, skips, connector_width) -> str | None:
    """Render a single-line bracket above the path for one residual skip.

    Only the first skip is drawn to keep the ASCII output readable; additional
    skips are reported in the destination block instead.
    """
    (source, destination), label = next(iter(skips.items()))
    # Offsets are shifted by one because ``labelled`` prepends the Input block.
    source += 1
    destination += 1

    starts = []
    position = 0
    for index, width in enumerate(widths):
        starts.append(position)
        position += width + connector_width
    centers = [start + widths[index] // 2 for index, start in enumerate(starts)]

    begin = centers[source]
    end = centers[destination]
    lane = [' '] * (end + 1)
    for column in range(begin, end + 1):
        lane[column] = '_'
    lane[begin] = '.'
    lane[end] = '.'
    text = f' skip: {label} '
    middle = (begin + end) // 2 - len(text) // 2
    if middle > begin and middle + len(text) < end:
        for offset, character in enumerate(text):
            lane[middle + offset] = character
    return ''.join(lane)


def draw_network(model, title=None, mode='compact', orientation='vertical'):
    """Print an ASCII diagram of a floating-point or quantized model.

    Args:
        model: a PyTorch module, optionally containing Brevitas quant layers.
        title: optional heading printed above the diagram.
        mode: ``"compact"`` for one summary line per quantizer, or
            ``"detailed"`` for per-channel ``ExMx`` weight formats.
        orientation: ``"vertical"`` to stack top to bottom, or ``"horizontal"``
            to read left to right with an optional skip lane above the path.
    """
    if mode not in ('compact', 'detailed'):
        raise ValueError("mode must be 'compact' or 'detailed'")
    if orientation not in ('vertical', 'horizontal'):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    detailed = mode == 'detailed'

    blocks, skips = _graph_blocks(model, detailed)
    if blocks is None:
        blocks, skips = _sequential_blocks(model, detailed), {}

    if orientation == 'vertical':
        _render_vertical(blocks, skips, title)
    else:
        _render_horizontal(blocks, skips, title)


def _graph_blocks(model, detailed: bool):
    """Trace ``model`` with Brevitas FX and return blocks plus residual skips.

    Returns ``(None, {})`` when tracing is unavailable or the graph has no
    ``call_module`` nodes, so the caller can fall back to the module tree. The
    skip map keys are ``(source_block_index, destination_block_index)`` and its
    values are the merge label.
    """
    try:
        from brevitas.fx import symbolic_trace
        graph_module = symbolic_trace(model)
    except Exception:
        return None, {}

    modules = dict(graph_module.named_modules())
    blocks = []
    node_to_block = {}
    merges = []  # (node, [input_nodes], label)
    for node in graph_module.graph.nodes:
        if node.op == 'call_module':
            rendered = _describe_module(node.target, modules[node.target], detailed)
            if rendered is not None:
                node_to_block[node] = len(blocks)
                blocks.append(rendered)
        elif node.op in ('call_function', 'call_method'):
            label = _merge_label(node)
            if label is not None:
                tensor_inputs = [
                    argument for argument in node.all_input_nodes
                    if argument in node_to_block or argument.op != 'get_attr']
                if len(tensor_inputs) >= 2:
                    merges.append((node, tensor_inputs, label))

    if not blocks:
        return None, {}

    skips = {}
    for node, inputs, label in merges:
        destination = _nearest_following_block(node, node_to_block)
        sources = [node_to_block[i] for i in inputs if i in node_to_block]
        if destination is not None and sources:
            skips[(min(sources), destination)] = label

    return blocks, skips


def _merge_label(node) -> str | None:
    """Return a merge label for add/cat-style nodes, else ``None``."""
    target = node.target
    name = getattr(target, '__name__', str(target))
    if name in ('add', 'add_', '__add__', 'iadd'):
        return 'add'
    if name in ('cat', 'concat'):
        return 'cat'
    return None


def _nearest_following_block(node, node_to_block):
    """Find the block index of the first downstream ``call_module`` user."""
    visited = set()
    frontier = list(node.users)
    while frontier:
        user = frontier.pop(0)
        if user in visited:
            continue
        visited.add(user)
        if user in node_to_block:
            return node_to_block[user]
        frontier.extend(user.users)
    # No downstream module: attach the skip to the last block before the node.
    preceding = [index for block_node, index in node_to_block.items()]
    return max(preceding) if preceding else None


def snapshot_parameters(model):
    """Return a detached CPU copy of every named parameter."""
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def changed_parameter_names(snapshot, model):
    """Return the names whose parameter value differs from ``snapshot``."""
    return [
        name for name,
        parameter in model.named_parameters()
        if not torch.equal(snapshot[name], parameter.detach().cpu())]


def plot_rounding_gradients(modules, comparison_input=None, span=2.5, points=1000):
    """Plot the backward gradient of each rounding module and print a sample.

    ``modules`` maps a label to a rounding ``nn.Module``. Every module uses
    ``torch.round`` in the forward pass, so the rounded outputs match; the
    difference is in the surrogate gradient. The function draws one gradient
    curve per module over ``[-span, span]`` and prints the forward-equality
    check plus the gradient of each module at ``comparison_input``.
    """
    import matplotlib.pyplot as plt

    device = next(iter(modules.values()))(torch.zeros(1)).device

    plt.figure(figsize=(8, 3))
    for label, module in modules.items():
        values = torch.linspace(-span, span, points, device=device, requires_grad=True)
        gradient = torch.autograd.grad(module(values).sum(), values)[0]
        plt.plot(values.detach().cpu(), gradient.detach().cpu(), label=label)
    plt.xlabel('input')
    plt.ylabel('gradient')
    plt.legend()
    plt.grid(True)
    plt.show()

    if comparison_input is None:
        comparison_input = [-1.6, -0.2, 0.2, 1.6]
    sample = torch.tensor(comparison_input, device=device, requires_grad=True)
    outputs = {}
    gradients = {}
    for label, module in modules.items():
        output = module(sample)
        outputs[label] = output
        gradients[label] = torch.autograd.grad(output.sum(), sample, retain_graph=True)[0]

    reference = next(iter(outputs.values()))
    same_forward = all(torch.equal(reference, output) for output in outputs.values())
    print('same rounded forward values:', same_forward)
    for label, gradient in gradients.items():
        print(f'{label} gradient:', gradient.detach().cpu().tolist())


def where_defined(quantizer, name):
    """Report which class in a quantizer's hierarchy defines ``name``.

    Walks the method-resolution order of the quantizer class and returns the
    first class whose own namespace defines ``name``, together with the source
    file of that class. Returns ``None`` when the attribute is not found on any
    class in the hierarchy.
    """
    quantizer_class = quantizer if isinstance(quantizer, type) else type(quantizer)
    for base in quantizer_class.__mro__:
        if name in vars(base):
            try:
                source_file = inspect.getsourcefile(base)
            except TypeError:
                source_file = '<built-in>'
            print(f'{name}: defined in {base.__name__} ({source_file})')
            return base.__name__, source_file
    print(f'{name}: not found in {quantizer_class.__name__} hierarchy')
    return None
