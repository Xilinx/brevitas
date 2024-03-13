from brevitas.proxy.experimental.base_float_quant import QuantFloatProxyFromInjector
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector


class WeightFloatQuantProxyFromInjector(WeightQuantProxyFromInjector, QuantFloatProxyFromInjector):
    pass


class BiasFloatQuantProxyFromInjector(BiasQuantProxyFromInjector, QuantFloatProxyFromInjector):
    pass
