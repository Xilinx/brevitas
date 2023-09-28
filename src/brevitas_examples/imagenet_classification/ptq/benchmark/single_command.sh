python ptq_benchmark_torchvision.py $1 --calibration-dir /scratch/datasets/imagenet_symlink/calibration --validation-dir /scratch/datasets/imagenet_symlink/val \
--quant_format float \
--target_backend layerwise \
--graph_eq_iterations 50 \
--act_param_method stats mse \
--act_quant_percentile 99.9 99.99
