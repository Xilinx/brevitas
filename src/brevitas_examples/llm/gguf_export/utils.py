# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.utils.quant_utils import _CachedIOGroupwiseInt


# TODO: temporary out-of-source workaround to cache scale and zero-point for GGUF export;
# see GGUFGroupwiseWeightQuantProxyFromInjector in proxy.py and _GGUFCachedQuantRestrictValue
# in base_quantizers.py for usage examples
class _GGUFCachedScaleZPGroupwiseInt:

    def __init__(self):
        self._enabled = False
        self.value = None
        self.scale = None

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        if not value:
            self.value = None
            self.scale = None
        self._enabled = value


# TODO: temporary workaround to cache hierarchical quantizers for GGUF export; should revisit with
# either custom tensor subclasses or a new nested cache implementation in core
class _GGUFCachedIOGroupwiseInt(_CachedIOGroupwiseInt):

    def __init__(self, quant_tensor, metadata_only, scaling_cache_impl, zero_point_cache_impl):
        super().__init__(quant_tensor, metadata_only)
        # NOTE: scale_cache_impl.value isn't populated, see _GGUFCachedQuantRestrictValue
        self.scale_of_scale = scaling_cache_impl.scale
        self.zero_point = zero_point_cache_impl.value
        self.scale_of_zero_point = zero_point_cache_impl.scale
