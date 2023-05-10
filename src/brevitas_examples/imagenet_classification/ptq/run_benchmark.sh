python ptq_benchmark_torchvision.py hydra.job.chdir=True hydra/launcher=joblib \
 model_name=resnet18,mobilenet_v2,vit_b_32 \
 target_backend=generic,layerwise,flexml  \
 scale_factor_type=float32,po2 \
 weight_bit_width=8b,6b,4b \
 act_bit_width=8b \
 bias_bit_width=int32,int16 \
 scaling_per_output_channel=per_tensor,per_channel \
 act_quant_type=symmetric,asymmetric \
 gptq=enabled,disabled \
 gptq_act_order=enabled,disabled \
 act_quant_percentile=point_9,point_99,point_999 \
 -m
