from dataclasses import dataclass
from typing import Union, Type


from . import WeightQuantProxy, BiasQuantProxy, ActivationQuantProxy
from .config import WeightQuantConfig, BiasQuantConfig, ActQuantConfig

@dataclass
class WeightQuantConfigSpec:
    type: Type[WeightQuantConfig] = WeightQuantConfig
    prefix = 'weight_'


@dataclass
class WeightQuantSpec:
    type: Type[WeightQuantProxy] = WeightQuantProxy
    config: Union[WeightQuantConfigSpec, WeightQuantConfig] = WeightQuantConfigSpec()


@dataclass
class BiasQuantConfigSpec:
    type: Type[BiasQuantConfig] = BiasQuantConfig
    prefix = 'bias_'


@dataclass
class BiasQuantSpec:
    type: Type[BiasQuantProxy] = BiasQuantProxy
    config: Union[BiasQuantConfigSpec, BiasQuantConfig] = BiasQuantConfigSpec()


@dataclass
class OutputQuantConfigSpec:
    type: Type[ActQuantConfig] = ActQuantConfig
    prefix = 'output_'


@dataclass
class OutputQuantSpec:
    type: Type[ActivationQuantProxy] = ActivationQuantProxy
    config: Union[OutputQuantConfigSpec, ActQuantConfig] = OutputQuantConfigSpec()