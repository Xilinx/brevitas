# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

from brevitas.export.inference.handler import GroupwiseIntWeightInferenceHandler
from brevitas.export.inference.manager import InferenceManager
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas_examples.llm.gguf_export.utils import _GGUFCachedIOGroupwiseInt
from brevitas_examples.llm.gguf_export.utils import _GGUFCachedScaleZPGroupwiseInt


class GGUFGroupwiseWeightQuantProxyFromInjector(GroupwiseWeightQuantProxyFromInjector):

    def __init__(self, quant_layer, quant_injector) -> None:
        scaling_cache_impl = _GGUFCachedScaleZPGroupwiseInt()
        zero_point_cache_impl = _GGUFCachedScaleZPGroupwiseInt()

        quant_injector = quant_injector.let(
            gguf_scaling_cache=scaling_cache_impl, gguf_zero_point_cache=zero_point_cache_impl)
        super().__init__(quant_layer, quant_injector)

        self.scaling_cache_impl = scaling_cache_impl
        self.zero_point_cache_impl = zero_point_cache_impl
        self.cache_class = partial(
            _GGUFCachedIOGroupwiseInt,
            scaling_cache_impl=scaling_cache_impl,
            zero_point_cache_impl=zero_point_cache_impl)

    @property
    def gguf_qtype(self):
        return self.quant_injector.gguf_qtype

    @property
    def cache_inference_quant_weight(self):
        return self._cache_inference_quant_weight

    @cache_inference_quant_weight.setter
    def cache_inference_quant_weight(self, enabled):
        # Initialize the outer K-quant scale before caching.
        # Other scaling implementations do not use `init_done`.
        if enabled and not getattr(self.tensor_quant.scaling_impl, 'init_done', True):
            raise RuntimeError(
                "Cannot enable GGUF export caching before scale is initialized. "
                "Run a cache-disabled forward pass first.")
        GroupwiseWeightQuantProxyFromInjector.cache_inference_quant_weight.fset(self, enabled)
        self.scaling_cache_impl.enabled = enabled
        self.zero_point_cache_impl.enabled = enabled

    @property
    def scale(self):
        # (possibly quantized) scale for each sub-group
        return self.retrieve_attribute('scale_')

    @property
    def zero_point(self):
        # (possibly quantized) zero-point for each sub-group; see Q5_KWeightQuant for example
        # NOTE: for K-quants this no longer matches `quant_weight.zero_point_`; simple types are
        # unaffected, since they fall through to the branch below.
        zero_point = self.retrieve_attribute('zero_point')
        if zero_point is not None:
            return zero_point  # see `_GGUFCachedScaleShiftQuantZeroPoint`
        # real-valued (possibly de-quantized) zero-point for each sub-group
        return self.retrieve_attribute('zero_point_')

    @property
    def scale_of_scale(self):
        # if the scale is quantized, this is its scale (one per super-group)
        return self.retrieve_attribute('scale_of_scale')

    @property
    def scale_of_zero_point(self):
        # if the zero-point is quantized, this is its scale (one per super-group)
        return self.retrieve_attribute('scale_of_zero_point')


# TODO: temporary workaround to export weight cache and GGUF qtype tags with custom
# proxy; should revisit if backend is restructured to use registries
class _GGUFGroupwiseIntWeightInferenceHandler(GroupwiseIntWeightInferenceHandler):
    handled_layer = GGUFGroupwiseWeightQuantProxyFromInjector


InferenceManager.handlers.append(_GGUFGroupwiseIntWeightInferenceHandler)
