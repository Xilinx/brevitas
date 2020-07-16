import torch

from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.nn import QuantReLU
from common import check_expected_pyt_120_fail

BIT_WIDTH = 8
MAX_VAL = 6.0
RANDOM_ITERS = 32


class TestQuantReLU:

    pass
    # @check_expected_pyt_120_fail
    # def test_scaling_stats_to_parameter(self):
    #
    #     stats_act = QuantReLU(bit_width=BIT_WIDTH,
    #                           max_val=MAX_VAL,
    #                           quant_type=QuantType.INT,
    #                           scaling_impl_type=ScalingImplType.STATS)
    #     stats_act.train()
    #     for i in range(RANDOM_ITERS):
    #         inp = torch.randn([8, 3, 64, 64])
    #         stats_act(inp)
    #
    #     stats_state_dict = stats_act.state_dict()
    #
    #     param_act = QuantReLU(bit_width=BIT_WIDTH,
    #                           max_val=MAX_VAL,
    #                           quant_type=QuantType.INT,
    #                           scaling_impl_type=ScalingImplType.PARAMETER)
    #     param_act.load_state_dict(stats_state_dict)
    #
    #     stats_act.eval()
    #     param_act.eval()
    #
    #     assert(torch.allclose(stats_act.quant_act_scale(), param_act.quant_act_scale()))
