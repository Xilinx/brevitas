from brevitas.nn import QuantRNNLayer, BidirRNNLayer
import torch
import time
from collections import namedtuple

SEQ = 1000
INPUT_SIZE = 5
BATCH = 5
HIDDEN = 100
SEED = 123456
LSTMState = namedtuple('LSTMState', ['hx', 'cx'])
torch.manual_seed(SEED)


class TestRNNQuant:
    def test_naiveRNN(self):
        weight_config = {
            'weight_quant_type': 'FP'
        }

        activation_config = {
            'quant_type': 'FP'
        }
        hardtanh_activation_config = {
            'quant_type': 'FP',
            'min_val': -1e32,
            'max_val': 1e32
        }

        input = torch.randn(SEQ, BATCH, INPUT_SIZE)
        states = torch.randn(BATCH, HIDDEN)

        q_gru = QuantRNNLayer(INPUT_SIZE, HIDDEN, activation_config=activation_config,
                                               weight_config=weight_config, norm_scale_input_config=hardtanh_activation_config)
        q_gru.eval()

        # Control
        gru = torch.nn.RNN(INPUT_SIZE, HIDDEN)
        q_gru.load_state_dict(gru.state_dict())
        gru_out, gru_out_state = gru(input, states.unsqueeze(0))
        start = time.time()
        out, out_state = q_gru(input, states)
        end = time.time() - start
        print(end)
        assert torch.allclose(gru_out, out, 1e-05, 1e-05)
        assert torch.allclose(gru_out_state, out_state, 1e-05, 1e-05)
        print("DONE")

    def test_BIRNN(self):
        weight_config = {
            'weight_quant_type': 'FP'
        }

        activation_config = {
            'quant_type': 'FP'
        }
        hardtanh_activation_config = {
            'quant_type': 'FP',
            'min_val': -1e32,
            'max_val': 1e32
        }

        input = torch.randn(SEQ, BATCH, INPUT_SIZE)
        states_fp = torch.randn(2, BATCH, HIDDEN)

        q_gru =BidirRNNLayer(INPUT_SIZE, HIDDEN, activation_config=activation_config,
                                               weight_config=weight_config,
                                               norm_scale_input_config=hardtanh_activation_config)
        q_gru.eval()

        # Control
        gru = torch.nn.RNN(INPUT_SIZE, HIDDEN, bidirectional=True)
        q_gru.load_state_dict(gru.state_dict())
        gru_out, gru_out_state = gru(input, states_fp)
        start = time.time()
        out, out_state = q_gru(input, [states_fp[0].squeeze(0), states_fp[1].squeeze(0)])
        end = time.time() - start
        print(end)
        assert torch.allclose(gru_out, out, 1e-05, 1e-05)
        assert torch.allclose(gru_out_state[0], out_state[0], 1e-05, 1e-05)
        assert torch.allclose(gru_out_state[1], out_state[1], 1e-05, 1e-05)
        print("DONE")