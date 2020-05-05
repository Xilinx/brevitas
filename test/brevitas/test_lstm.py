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
            'weight_quant_type': 'FP'
        }

        activation_config = {
            'quant_type': 'FP'
        }
        hidden_activation_config = {
            'quant_type': 'FP',
            'min_val': -1e32,
            'max_val': 1e32
        }
        input = torch.randn(SEQ, BATCH, INPUT_SIZE)
        states = LSTMState(torch.zeros(BATCH, HIDDEN),
                           torch.zeros(BATCH, HIDDEN))

        q_lstm = (QuantLSTMLayer(INPUT_SIZE, HIDDEN, activation_config=activation_config,
                                 norm_scale_out_config=hidden_activation_config,
                                 norm_scale_hidden_config=hidden_activation_config,
                                 weight_config=weight_config))
        q_lstm.eval()

        # Control
        lstm = torch.nn.LSTM(INPUT_SIZE, HIDDEN, 1)
        lstm_state = LSTMState(states.hx.unsqueeze(0), states.cx.unsqueeze(0))
        q_lstm.load_state_dict(lstm.state_dict())
        lstm_out, lstm_out_state = lstm(input, lstm_state)
        out, custom_state = q_lstm(input)

        assert (out - lstm_out).abs().max() < 1e-5
        assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
        assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5

    def test_BILSTM(self):
        weight_config = {
            'weight_quant_type': 'FP'
        }

        activation_config = {
            'quant_type': 'FP'
        }
        hidden_activation_config = {
            'quant_type': 'FP',
            'min_val': -1e32,
            'max_val': 1e32
        }

        input = torch.randn(SEQ, BATCH, INPUT_SIZE)
        states = LSTMState(torch.zeros(2, BATCH, HIDDEN),
                           torch.zeros(2, BATCH, HIDDEN))

        states_quant_direct = LSTMState(states[0][0].squeeze(0), states[1][0].squeeze(0))
        states_quant_reverse = LSTMState(states[0][1].squeeze(0), states[1][1].squeeze(0))
        states_quant = [states_quant_direct, states_quant_reverse]
        q_gru = BidirLSTMLayer(INPUT_SIZE, HIDDEN, activation_config=activation_config,
                               weight_config=weight_config,
                               norm_scale_out_config=hidden_activation_config,
                               norm_scale_hidden_config=hidden_activation_config)
        q_gru.eval()

        # Control
        gru = torch.nn.LSTM(INPUT_SIZE, HIDDEN, bidirectional=True)
        q_gru.load_state_dict(gru.state_dict())
        gru_out, gru_out_state = gru(input, states)
        out, out_state = q_gru(input, states_quant)

        assert torch.allclose(gru_out, out, 1e-05, 1e-05)
        assert torch.allclose(gru_out_state[0][0], out_state[0][0], 1e-05, 1e-05)
        assert torch.allclose(gru_out_state[0][1], out_state[1][0], 1e-05, 1e-05)
        assert torch.allclose(gru_out_state[1][0], out_state[0][1], 1e-05, 1e-05)
        assert torch.allclose(gru_out_state[1][1], out_state[1][1], 1e-05, 1e-05)

    def test_QLSTM(self):
        weight_config = {
            'weight_quant_type': 'INT'
        }

        activation_config = {
            'quant_type': 'INT'
        }
        hidden_activation_config = {
            'quant_type': 'INT',
            'min_val': -1e32,
            'max_val': 1e32
        }

        q_lstm = QuantLSTMLayer(INPUT_SIZE, HIDDEN, activation_config=activation_config,
                                norm_scale_out_config=hidden_activation_config,
                                norm_scale_hidden_config=hidden_activation_config,
                                weight_config=weight_config)
        assert True

    def test_BFQLSTM(self):
        weight_config = {
            'weight_quant_type': 'FP'
        }

        activation_config = {
            'quant_type': 'FP'
        }
        hidden_activation_config = {
            'quant_type': 'FP',
            'min_val': -1e32,
            'max_val': 1e32
        }
        input = torch.randn(SEQ, BATCH, INPUT_SIZE)
        states = LSTMState(torch.zeros(BATCH, HIDDEN),
                           torch.zeros(BATCH, HIDDEN))

        q_lstm = (QuantLSTMLayer(INPUT_SIZE, HIDDEN, activation_config=activation_config,
                                 norm_scale_out_config=hidden_activation_config,
                                 norm_scale_hidden_config=hidden_activation_config,
                                 weight_config=weight_config, batch_first=True))
        q_lstm.eval()

        # Control
        lstm = torch.nn.LSTM(INPUT_SIZE, HIDDEN, 1)
        lstm_state = LSTMState(states.hx.unsqueeze(0), states.cx.unsqueeze(0))
        q_lstm.load_state_dict(lstm.state_dict())
        lstm_out, lstm_out_state = lstm(input, lstm_state)
        out, custom_state = q_lstm(input.transpose(0,1))
        out = out.transpose(0, 1)

        assert (out - lstm_out).abs().max() < 1e-5
        assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
        assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5