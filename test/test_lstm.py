from brevitas.nn import QuantLSTMLayer
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


class TestLSTMQuant:
    def test_naiveLSTM(self):

        weight_config = {
            'weight_quant_type' : 'QuantType.FP'
        }

        activation_config = {
            'quant_type': 'QuantType.FP'
        }
        hidden_activation_config = {
            'quant_type': 'QuantType.FP',
            'min_val': -1e32,
            'max_val': 1e32
        }
        input = torch.randn(SEQ, BATCH, INPUT_SIZE)
        states = LSTMState(torch.randn(BATCH, HIDDEN),
                           torch.randn(BATCH, HIDDEN))

        q_lstm = torch.jit.script(QuantLSTMLayer(INPUT_SIZE, HIDDEN, activation_config=activation_config,
                                                 hidden_state_activation_config=hidden_activation_config,
                                                 weight_config=weight_config, layer_norm='decompose'))
        q_lstm.eval()

        # Control
        lstm = torch.nn.LSTM(INPUT_SIZE, HIDDEN, 1)
        lstm_state = LSTMState(states.hx.unsqueeze(0), states.cx.unsqueeze(0))
        q_lstm.load_state_dict_new(lstm.state_dict())
        lstm_out, lstm_out_state = lstm(input, lstm_state)
        start = time.time()
        out, custom_state = q_lstm(input, states)
        end = time.time()-start
        print(end)

        assert (out - lstm_out).abs().max() < 1e-5
        assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
        assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5
        print("DONE")
