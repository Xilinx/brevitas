# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from brevitas.graph.base import ModuleInstanceTransformTensor
from brevitas.utils.torch_utils import is_parametrized
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.llm.llm_quant.awq.graph import EqualizeAWQ
from brevitas_examples.llm.llm_quant.run_utils import fix_rewriter


def _get_tensor_weight_id(module, tensor_name):
    if is_parametrized(module) and tensor_name in module.parametrizations:
        return id(module.parametrizations[tensor_name].original)
    elif hasattr(module, tensor_name):
        return id(getattr(module, tensor_name))
    return None


def fix_regions(regions, old_model_ref, tensor_name):
    for region in regions:
        name_to_module_keys = list(region.name_to_module.keys())
        for graph_module_name in name_to_module_keys:
            graph_module = region.name_to_module[graph_module_name]
            tensor_id = _get_tensor_weight_id(graph_module, tensor_name)
            name, module = [
                (n, m) for n, m in old_model_ref.named_modules()
                if hasattr(m, tensor_name) and _get_tensor_weight_id(m, tensor_name) == tensor_id][0]
            # Replace keys, improve this logic
            del region.name_to_module[graph_module_name]
            region.name_to_module[name] = module
            # There are two cases, the names in the region dictionaries can have the name
            srcs_keys = list(region.srcs.keys())
            for srcs_key in srcs_keys:
                split_srcs_key = srcs_key.split("$")
                src_graph_module_name = split_srcs_key[0]
                eq_indexes_str = split_srcs_key[1]
                eq_indexes = region.srcs[srcs_key]
                if graph_module_name == src_graph_module_name:
                    del region.srcs[srcs_key]
                    region.srcs[f"{name}${eq_indexes_str}"] = eq_indexes
                    break
            # And sinks
            sinks_keys = list(region.sinks.keys())
            for sinks_key in sinks_keys:
                split_sinks_key = sinks_key.split("$")
                src_graph_module_name = split_sinks_key[0]
                eq_indexes_str = split_sinks_key[1]
                eq_indexes = region.sinks[sinks_key]
                if graph_module_name == src_graph_module_name:
                    del region.sinks[sinks_key]
                    region.sinks[f"{name}${eq_indexes_str}"] = eq_indexes
                    break
    return regions


def fused_awq_scaling_no_fx(model: nn.Module, calibration_loader: DataLoader, args: Namespace):
    with torch.no_grad():
        new_model, guards = torch._dynamo.export(model)(**calibration_loader[0])
    if hasattr(model, str(torch.nn.functional.scaled_dot_product_attention)):
        m_to_add = getattr(model, str(torch.nn.functional.scaled_dot_product_attention))
        new_model.add_module(str(torch.nn.functional.scaled_dot_product_attention), m_to_add)
    new_model = offload_model(new_model)
    # Insert the identity scaling factors
    eq = EqualizeAWQ(
        sdpa_regions=True,
        weight_group_size=args.weight_group_size
        if args.weight_quant_granularity == 'per_group' else None,
        add_parametrizations_inplace=False,
    )
    new_model, regions, rewriters = eq.apply(model=new_model)
    rewriters = fix_rewriter(rewriters, model, "weight")
    # Map the regions of the graph model (new_model) to modules of the original model (model)
    regions = fix_regions(regions, model, "weight")
    with torch.no_grad():
        for r in rewriters:
            # The weights between model and new_model are tied, so this check prevents
            # scaling twice
            if not isinstance(r, ModuleInstanceTransformTensor):
                model = r.apply(model)
    remove_hooks(new_model)
    return regions
