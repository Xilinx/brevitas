import re

from .parameter_quant import *
from .runtime_quant import *

from brevitas import docstrings

#
# Docs post processing
#

wo_prefix = 'weight_quant_proxy.parameters'
w_prefix = 'weight_quant_proxy.parameters_with_prefix'
docstrings.params[w_prefix] =  docstrings.params[wo_prefix]
docstrings.params[w_prefix] = re.compile(r'\nscaling_stats_reduce_dim\n(.*\n.*\n.*\n)').sub(r"\n", docstrings.params[w_prefix])
docstrings.params[w_prefix] = re.compile(r'\nscaling_shape\n(.*\n)').sub(r"\n", docstrings.params[w_prefix])
docstrings.params[w_prefix] = re.compile(r'\nscaling_stats_input_view_shape_impl\n(.*\n.*\n)').sub(r"\n", docstrings.params[w_prefix])
docstrings.params[w_prefix] = re.compile(r'\nscaling_stats_input_concat_dim\n(.*\n.*\n.*\n)').sub(r"\n", docstrings.params[w_prefix])
docstrings.params[w_prefix] = re.compile(r'\ntracked_parameter_list_init\n(.*\n.*\n.*\n)').sub(r"\n", docstrings.params[w_prefix])
docstrings.params[w_prefix] = re.compile(r'(\n)([a-z_]+)(\n)').sub(r"\1weight_\2\3", docstrings.params[w_prefix])
docstrings.params[w_prefix] = re.compile(r'(`)([a-z_]+)(`)').sub(r"\1weight_\2\3", docstrings.params[w_prefix])

