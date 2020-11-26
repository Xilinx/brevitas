from typing import List, Any, Dict, Callable, Tuple, Union, Optional
from dataclasses import dataclass, field

from torch import Tensor
from torch.nn import Module, Parameter

from ..module import Index, FnType


@dataclass
class TraceElem:
    fn: Callable
    fn_type: FnType
    fn_args: List[Any]
    fn_kwargs: [Dict[str, Any]]
    fn_out: Any
    fn_args_index: List[Union[Index, List[Index], Tuple[Index, ...]]]
    fn_kwargs_index: Dict[str, Union[Index, List[Index], Tuple[Index, ...]]]
    fn_out_index: Index
    module_context_list: List[Module]
    prefix_list: List[str]
    module_input_list: List[Any]
    module_output: Any = None  # set after init
    module_fn_name: Optional[str] = None

    @property
    def fn_name(self):
        if isinstance(self.fn, Module):
            return self.fn.__class__.__name__
        elif isinstance(self.fn, str):
            return self.fn
        else:
            return self.fn.__name__

    @property
    def prefix(self):
        return '.'.join(self.prefix_list)


@dataclass
class Trace:
    model_input_list: List[Any] = field(default_factory=list)
    model_output_list: List[Any] = field(default_factory=list)
    trace_elem_list: List[TraceElem] = field(default_factory=list)
    index_map: Dict[Index, Any] = field(default_factory=dict)
    named_buffers: Dict[str, Tensor] = None
    named_params: Dict[str, Parameter] = None
    _index_id = -1

    @property
    def model_input_index_list(self):
        return self.index_from_val(self.model_input_list)

    @property
    def model_output_index_list(self):
        return self.index_from_val(self.model_output_list)

    def is_buffer(self, tensor):
        if not isinstance(tensor, Tensor):
            return False
        for b in self.named_buffers.values():
            if tensor is b:
                return True
        return False

    def name_from_buffer(self, buffer):
        for n, b in self.named_buffers.items():
            if buffer is b:
                return n
        return None

    def name_from_param(self, params):
        for n, p in self.named_params.items():
            if params is p:
                return n
        return None

    def index_from_map(self, val):
        for i, v in self.index_map.items():
            if v is val:  # Note "is" and not "=="
                return i
        return None

    def fresh_index_id(self):
        self._index_id += 1
        return self._index_id

    def index_from_val(self, value: Any, recurse=True):
        index = self.index_from_map(value)
        if isinstance(value, tuple) and recurse and index is None:
            return tuple(self.index_from_val(v, True) for v in value)
        elif isinstance(value, list) and recurse and index is None:
            return [self.index_from_val(v, True) for v in value]
        elif isinstance(value, dict) and recurse and index is None:
            return {k: self.index_from_val(v, True) for k, v in value.items()}
        else:
            if index is None:
                id = self.fresh_index_id()
                self.index_map[Index(id)] = value
                return Index(id)
            else:
                return index

