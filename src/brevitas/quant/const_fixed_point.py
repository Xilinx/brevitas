import brevitas
import torch
from brevitas.inject import ExtendedInjector
import brevitas.core as bv_core
import brevitas.core.quant as bv_quant
import brevitas.core.scaling as bv_scaling
import brevitas.core.zero_point as bv_zp
import brevitas.core.bit_width as bv_bw
import brevitas.core.function_wrapper.ops_ste as bv_os

round_modes = { 'floor':bv_os.FloorSte,
                'round':bv_os.RoundSte,
                'ceil':bv_os.CeilSte}


def create_int_quant(scale=1, bit_width=8, _narrow_range=False, _signed=True, round_mode='round', dtype=torch.float32):
    """
    Create integer quantizer class
    
    :param scale: max. abs. value of quantized value
    :param bit_width: bit width of quantized value
    :param _narrow_range: if True and signed is True quantization range is symetrical.
                         (like binary coding U1)
                         if False and signed is True quantization range is unsymetrical.
                         (like binary coding U2)
                         If signed is False, narrow_range doesn't metter
    :param _signed: quantize siged or unsigned
    :param round_mode: type of float to int implementation, alloed are ['floor', 'ceil', 'round']
    :param dtype: type of constants values
    
    :return : class of int quantizer
    """
    bw = bit_width
    sc = scale
    float_to_int = round_modes[round_mode]

    class IntQuant(ExtendedInjector):
        tensor_quant = bv_quant.int.RescalingIntQuant
        int_quant = bv_quant.IntQuant
        scaling_impl = bv_scaling.ConstScaling
        int_scaling_impl = bv_scaling.IntScaling
        zero_point_impl = bv_zp.ZeroZeroPoint
        bit_width_impl = bv_bw.BitWidthConst
        
        requires_input_scale = False
        requires_input_bit_width = False
        requires_output_scale = False
        requires_output_bit_width = False
        
        float_to_int_impl = float_to_int

        signed = _signed
        narrow_range = _narrow_range
        scaling_init = torch.tensor([sc], dtype=dtype)
        bit_width = bw
        zero_point = torch.tensor([0], dtype=dtype)

    return IntQuant


def create_fixed_point_quant(bit_width, 
                             int_width=1, 
                             narrow_range=False, 
                             signed=True,
                             round_mode='round',
                             dtype=torch.float32):
    """
    Create fixed point qunatizer class with fixed (const) point position.
    Tensor is limited by range of min and max value depend on poin position, sign etc.
    
    :param bit_width: bit width of quantized value
    :param int_width: integer part bit width of quantized value
    :param narrow_range: if True and signed is True quantization range is symetrical. 
                         (binary coding simmilar to U1)
                         if False and signed is True quantization range is unsymetrical.
                         (bianary codding like U2)
                         If signed is False, narrow_range doesn't metter
    :param signed: qunatize signed or unsigned (signed quantization has 
                   two times smaller max. abs. value of quantized range)
    :param round_mode: type of float to int implementation, alloed are ['floor', 'ceil', 'round']
    :param dtype: type of constants values
    
    :return: class of fixed point quantizer
    
    # example usage
    Q6_2_C = create_fixed_point_quant(bit_width=6, int_width=2, narrow_range=False, signed=True, round_mode='ceil')
    Q6_2_CN = create_fixed_point_quant(bit_width=6, int_width=2, narrow_range=True, signed=True, round_mode='ceil')
    Q8_1_R = create_fixed_point_quant(bit_width=8, int_width=1, narrow_range=False, signed=True, round_mode='round')
    Q8_2_F = create_fixed_point_quant(bit_width=8, int_width=2, narrow_range=False, signed=False, round_mode='floor')
    
    # examle layer which use quantizers
    L_Q6_2_C = brevitas.nn.QuantIdentity(act_quant=Q6_2_C, return_quant_tensor=True)
    L_Q6_2_CN = brevitas.nn.QuantIdentity(act_quant=Q6_2_CN, return_quant_tensor=True)
    L_Q8_1_R = brevitas.nn.QuantIdentity(act_quant=Q8_1_R, return_quant_tensor=True)
    L_Q8_2_F = brevitas.nn.QuantIdentity(act_quant=Q8_2_F, return_quant_tensor=True)
    
    # sample tensor
    t_in = torch.linspace(-5.0, 5.0, 11)
    print(t_in)
    print()
    
    # pass the same tensor by different quantizers
    for L in [L_Q6_2_C, L_Q6_2_CN, L_Q8_1_R, L_Q8_2_F]:
        t_out = L(t_in)
        print(t_out)
        print()
    
    # generated output
    tensor([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.])

    QuantTensor(value=tensor([-2.0000, -2.0000, -2.0000, -2.0000, -1.0000,  0.0000,  1.0000,  1.9375,
             1.9375,  1.9375,  1.9375]), scale=tensor([0.0625]), zero_point=tensor(0.), bit_width=tensor(6.), signed_t=tensor(True), training_t=tensor(True))

    QuantTensor(value=tensor([-1.9375, -1.9375, -1.9375, -1.9375, -1.0000,  0.0000,  1.0000,  1.9375,
             1.9375,  1.9375,  1.9375]), scale=tensor([0.0625]), zero_point=tensor(0.), bit_width=tensor(6.), signed_t=tensor(True), training_t=tensor(True))

    QuantTensor(value=tensor([-1.0000, -1.0000, -1.0000, -1.0000, -1.0000,  0.0000,  0.9922,  0.9922,
             0.9922,  0.9922,  0.9922]), scale=tensor([0.0078]), zero_point=tensor(0.), bit_width=tensor(8.), signed_t=tensor(True), training_t=tensor(True))

    QuantTensor(value=tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.9882, 1.9922, 2.9961,
            3.9843, 4.0000]), scale=tensor([0.0157]), zero_point=tensor(0.), bit_width=tensor(8.), signed_t=tensor(False), training_t=tensor(True))

    """
    scale = (2**(bit_width-signed)-narrow_range-(1-signed)) / 2**(bit_width-int_width)
    
    return create_int_quant(scale=scale, 
                            bit_width=bit_width, 
                            _narrow_range=narrow_range, 
                            _signed=signed,
                            round_mode=round_mode,
                            dtype=dtype)

