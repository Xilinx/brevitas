from brevitas.nn import QuantLSTMLayer, BidirLSTMLayer
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

    def test_BILSTM(self):
        weight_config = {
            'weight_quant_type': 'QuantType.FP'
        }

        activation_config = {
            'quant_type': 'QuantType.FP'
        }
        hardtanh_activation_config = {
            'quant_type': 'QuantType.FP',
            'min_val': -1e64,
            'max_val': 1e64
        }

        input = torch.randn(SEQ, BATCH, INPUT_SIZE)
        states = LSTMState(torch.randn(2, BATCH, HIDDEN),
                           torch.randn(2, BATCH, HIDDEN))

        states_quant_direct = LSTMState(states[0][0].squeeze(0), states[1][0].squeeze(0))
        states_quant_reverse = LSTMState(states[0][1].squeeze(0), states[1][1].squeeze(0))
        states_quant = [states_quant_direct, states_quant_reverse]
        q_gru = torch.jit.script(BidirLSTMLayer(INPUT_SIZE, HIDDEN, activation_config=activation_config,
                                               weight_config=weight_config,
                                               hidden_state_activation_config=hardtanh_activation_config))
        q_gru.eval()

        # Control
        gru = torch.nn.LSTM(INPUT_SIZE, HIDDEN, bidirectional=True)
        q_gru.load_state_dict_new(gru.state_dict())
        gru_out, gru_out_state = gru(input, states)
        start = time.time()
        out, out_state = q_gru(input, states_quant)
        end = time.time() - start
        print(end)
        assert torch.allclose(gru_out, out, 1e-05, 1e-05)
        assert torch.allclose(gru_out_state[0][0], out_state[0][0], 1e-05, 1e-05)
        assert torch.allclose(gru_out_state[0][1], out_state[1][0], 1e-05, 1e-05)
        assert torch.allclose(gru_out_state[1][0], out_state[0][1], 1e-05, 1e-05)
        assert torch.allclose(gru_out_state[1][1], out_state[1][1], 1e-05, 1e-05)

        print("DONE")
