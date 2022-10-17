import torch

# Avoid fast algorithms that might introduce extra error during fake-quantization
torch.use_deterministic_algorithms(True)
