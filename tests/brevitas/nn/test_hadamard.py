from brevitas.nn import HadamardClassifier

OUTPUT_FEATURES = 10
INPUT_FEATURES = 5
BIT_WIDTH = 5


class TestHadamardStateDict:

    def test_module_state_dict(self):
        """Check that the Hadamard classifier can save and load state_dict without warning."""

        mod = HadamardClassifier(out_channels=OUTPUT_FEATURES, in_channels=INPUT_FEATURES)

        mod2 = HadamardClassifier(out_channels=OUTPUT_FEATURES, in_channels=INPUT_FEATURES)

        mod2.load_state_dict(mod.state_dict())
