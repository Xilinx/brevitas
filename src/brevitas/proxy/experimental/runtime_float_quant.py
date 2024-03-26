from brevitas.proxy.experimental.base_float_quant import QuantFloatProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector


class ActFloatQuantProxyFromInjector(ActQuantProxyFromInjector, QuantFloatProxyFromInjector):
    pass
