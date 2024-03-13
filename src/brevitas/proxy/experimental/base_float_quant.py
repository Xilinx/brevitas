from brevitas.proxy.quant_proxy import QuantProxyFromInjector


class QuantFloatProxyFromInjector(QuantProxyFromInjector):

    def mantissa_bit_width(self):
        return self.quant_injector.mantissa_bit_width

    def exponent_bit_width(self):
        return self.quant_injector.exponent_bit_width

    def saturate(self):
        return self.quant_injector.saturate

    def nan_values(self):
        if 'nan_values' in self.quant_injector:
            return self.quant_injector.nan_values
        return None

    def inf_values(self):
        if 'inf_values' in self.quant_injector:
            return self.quant_injector.inf_values
        return None

    @property
    def is_ocp(self):
        is_e4m3 = self.mantissa_bit_width() == 3 and self.exponent_bit_width() == 4

        is_ocp_e4m3 = is_e4m3 and self.inf_values is None and self.nan_values == (('111',))

        is_e5m2 = self.mantissa_bit_width() == 5 and self.exponent_bit_width() == 2

        is_ocp_e5m2 = is_e5m2 and self.inf_values == (
            ('00',)) and self.nan_values == ('01', '11', '10')

        return is_ocp_e4m3 or is_ocp_e5m2
