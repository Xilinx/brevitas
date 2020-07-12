from .parameter_quant import WeightQuantProxy, BiasQuantProxy
from .runtime_quant import ActQuantProxy, ClampQuantProxy, TruncQuantProxy

# retrocompatibility to avoid breaking imports TODO deprecate
from .runtime_quant import ActQuantProxy as ActivationQuantProxy
