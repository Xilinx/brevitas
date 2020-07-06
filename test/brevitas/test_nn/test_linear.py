from brevitas.nn import QuantLinear

OUTPUT_FEATURES = 10
INPUT_FEATURES = 5


class TestQuantLinear:

    def test_default_init(self):
        mod = QuantLinear(out_features=OUTPUT_FEATURES, in_features=INPUT_FEATURES, bias=True)
        for name, m in mod.named_modules():
            print(name)