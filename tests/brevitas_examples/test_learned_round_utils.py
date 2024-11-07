# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import pytest
import pytest_cases
from pytest_cases import fixture
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from brevitas import config
from brevitas.core.function_wrapper.learned_round import AutoRoundSte
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
import brevitas.nn as qnn
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.quant_tensor.base_quant_tensor import QuantTensor
from brevitas_examples.common.learned_round.learned_round_method import AdaRound
from brevitas_examples.common.learned_round.learned_round_method import AdaRoundLoss
from brevitas_examples.common.learned_round.learned_round_method import AutoRound
from brevitas_examples.common.learned_round.learned_round_method import AutoRoundLoss
from brevitas_examples.common.learned_round.learned_round_optimizer import get_blocks
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import \
    LearnedRoundVisionUtils

config.IGNORE_MISSING_KEYS = True


class TestLearnedRound:

    @fixture
    def quant_model():

        class QuantBlock(nn.Module):

            def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
                super().__init__()
                self.layer1 = qnn.QuantLinear(in_features=in_features, out_features=hidden_dim)
                self.layer2 = qnn.QuantLinear(in_features=hidden_dim, out_features=out_features)
                self.relu = qnn.QuantReLU(return_quant_tensor=True)

            def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                out = self.layer1(x)
                out = self.relu(out)
                out = self.layer2(out)
                return self.relu(out)

        class TestQuantModel(nn.Module):

            def __init__(self, in_features: int, out_features: int, hidden_dim: int) -> None:
                super().__init__()
                self.in_proj_mlp = QuantBlock(
                    in_features=in_features, hidden_dim=hidden_dim, out_features=hidden_dim)
                self.hidden_mlp = QuantBlock(
                    in_features=hidden_dim, hidden_dim=hidden_dim, out_features=hidden_dim)
                self.out_proj_mlp = QuantBlock(
                    in_features=hidden_dim, hidden_dim=hidden_dim, out_features=out_features)

            def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                out = self.in_proj_mlp(x)
                out = self.hidden_mlp(out)
                return self.out_proj_mlp(out)

        return TestQuantModel(in_features=2, out_features=1, hidden_dim=4)

    @fixture
    def model():

        class Block(nn.Module):

            def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
                super().__init__()
                self.layer1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
                self.layer2 = nn.Linear(in_features=hidden_dim, out_features=out_features)
                self.relu = F.relu

            def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                out = self.layer1(x)
                out = self.relu(out)
                out = self.layer2(out)
                return self.relu(out)

        class TestModel(nn.Module):

            def __init__(self, in_features: int, out_features: int, hidden_dim: int) -> None:
                super().__init__()
                self.in_proj_mlp = Block(
                    in_features=in_features, hidden_dim=hidden_dim, out_features=hidden_dim)
                self.hidden_mlp = Block(
                    in_features=hidden_dim, hidden_dim=hidden_dim, out_features=hidden_dim)
                self.out_proj_mlp = Block(
                    in_features=hidden_dim, hidden_dim=hidden_dim, out_features=out_features)

            def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                out = self.in_proj_mlp(x)
                out = self.hidden_mlp(out)
                return self.out_proj_mlp(out)

        return TestModel(in_features=2, out_features=1, hidden_dim=4)

    @fixture
    def data_loader():

        class TestDataset(Dataset):

            def __init__(self):
                self.data = torch.tensor([[1.0, 2.0]])
                self.labels = torch.tensor([0])

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]

        return DataLoader(TestDataset(), batch_size=1, shuffle=False)

    def test_get_blocks(self, quant_model: nn.Module):

        def _is_block(module: nn.Module, module_name: str) -> bool:
            return module_name in ["hidden_mlp"]

        expected_blocks = [quant_model.hidden_mlp]
        blocks = get_blocks(quant_model, _is_block)

        assert expected_blocks == blocks

    def test_get_layers(self, quant_model: nn.Module):

        def _is_layer(module: nn.Module, module_name: str) -> bool:
            return isinstance(module, QuantWBIOL)

        expected_layers = [
            quant_model.in_proj_mlp.layer1,
            quant_model.in_proj_mlp.layer2,
            quant_model.hidden_mlp.layer1,
            quant_model.hidden_mlp.layer2,
            quant_model.out_proj_mlp.layer1,
            quant_model.out_proj_mlp.layer2]
        layers = get_blocks(quant_model, _is_layer)

        assert expected_layers == layers

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    # NOTE: DataSaverHook always returns a torch.Tensor for the input, while for the output it can be either a torch.Tensor or
    # a QuantTensor. Is this expected behaviour? For that reason, the argument _assert_type is included in _aux_check_tensors.
    # Also, returning an empty list for the save_inp_out_data does not seem very natural, considering a tensors if the appropiate
    # store option is activated.
    @pytest.mark.parametrize("store_input", [True, False])
    @pytest.mark.parametrize("store_out", [True, False])
    @pytest.mark.parametrize("keep_gpu", [True, False])
    @pytest.mark.parametrize("disable_quant", [True, False])
    def test_save_inp_out_data(
            self, model, quant_model, data_loader, store_input, store_out, keep_gpu, disable_quant):
        # Initialise utils to save tensors
        learned_round_vision_utils = LearnedRoundVisionUtils()
        # Make sure that the quant and FP models share the same weights
        quant_model.load_state_dict(model.state_dict())

        model.eval()
        model = model.cuda()

        quant_model.eval()
        quant_model = quant_model.cuda()

        # Retrieve module from quant_model
        module = quant_model.hidden_mlp

        cache_quant_partial_input = []
        cache_quant_partial_output = []

        cache_fp_partial_input = []
        cache_fp_partial_output = []

        def _aux_check_tensors(
                result_tensor, expected_tensor, keep_gpu, disable_quant, assert_type=False):
            # Verify that tensor is of the appropiate type
            if assert_type:
                assert isinstance(result_tensor, torch.Tensor if disable_quant else QuantTensor)
            # Extract value tensors
            if isinstance(result_tensor, QuantTensor):
                result_tensor, expected_tensor = result_tensor.value, expected_tensor.value
            # Verify that tensor is in appropiate device
            assert result_tensor.is_cuda == keep_gpu
            # Make sure tensors are in the same device before comparison
            if not keep_gpu:
                expected_tensor = expected_tensor.cpu()

            assert torch.allclose(result_tensor, expected_tensor)

        # Compute ground truths inputs and outputs
        with torch.no_grad():
            for batch_data, _ in data_loader:
                batch_data = batch_data.cuda()
                # Compute quant inputs to module
                quant_partial_input = quant_model.in_proj_mlp(batch_data)
                cache_quant_partial_input.append(quant_partial_input)
                # Compute quant outputs of module
                quant_partial_output = quant_model.hidden_mlp(quant_partial_input)
                cache_quant_partial_output.append(quant_partial_output)

                # Compute FP inputs to module
                fp_partial_input = model.in_proj_mlp(batch_data)
                cache_fp_partial_input.append(fp_partial_input)
                # Compute FP outputs of module
                fp_partial_output = model.hidden_mlp(fp_partial_input)
                cache_fp_partial_output.append(fp_partial_output)

        # Inputs and outputs are concatenated along the batch dimension.
        # See https://github.com/quic/aimet/blob/7c9eded51e3d8328746e7ba4cf68c7162f841712/TrainingExtensions/torch/src/python/aimet_torch/v1/adaround/activation_sampler.py#L231
        cache_quant_partial_input = torch.cat(cache_quant_partial_input, dim=0)
        cache_quant_partial_output = torch.cat(cache_quant_partial_output, dim=0)

        cache_fp_partial_input = torch.cat(cache_fp_partial_input, dim=0)
        cache_fp_partial_output = torch.cat(cache_fp_partial_output, dim=0)

        # Retrieve input and output data
        input_data, out_data = learned_round_vision_utils._save_inp_out_data(quant_model, module, data_loader, store_input, store_out, keep_gpu, disable_quant)
        # Verify that empty lists are returned
        if store_input:
            if disable_quant:
                _aux_check_tensors(
                    input_data, fp_partial_input, keep_gpu, disable_quant, assert_type=True)
            else:
                _aux_check_tensors(input_data, quant_partial_input, keep_gpu, disable_quant)
        else:
            assert len(input_data) == 0

        if store_out:
            if disable_quant:
                _aux_check_tensors(out_data, fp_partial_output, keep_gpu, disable_quant)
            else:
                _aux_check_tensors(
                    out_data, quant_partial_output, keep_gpu, disable_quant, assert_type=True)
        else:
            assert len(out_data) == 0

    @pytest.mark.parametrize(
        "learned_round_class, rounding_mode, float_to_int_impl",
        [(AutoRound, "AUTO_ROUND", AutoRoundSte), (AdaRound, "LEARNED_ROUND", LearnedRoundSte)])
    def test_insert_learned_round_quantizer(
            self, quant_model, learned_round_class, rounding_mode, float_to_int_impl):
        block = quant_model.in_proj_mlp
        learned_round = learned_round_class(iters=100)
        learned_round._insert_learned_round_quantizer(block)

        for module in block.modules():
            if hasattr(module, "weight_quant"):
                assert module.weight_quant.rounding_mode == rounding_mode
                assert isinstance(
                    module.weight_quant.tensor_quant.int_quant.float_to_int_impl, float_to_int_impl)

    @pytest.mark.parametrize("learned_round_class", [AutoRound, AdaRound])
    @pytest.mark.parametrize(
        "block_strs, num_round_modules", [([], 0), (["hidden_mlp"], 2),
                                          (["in_proj_mlp", "out_proj_mlp"], 4)])
    def test_find_learned_round_modules(
            self, quant_model, learned_round_class, block_strs, num_round_modules):
        learned_round = learned_round_class(iters=100)
        # Inject quantizers in quant model
        for block_str in block_strs:
            block = getattr(quant_model, block_str)
            learned_round._insert_learned_round_quantizer(block)
        learned_round_modules = learned_round._find_learned_round_modules(quant_model)
        assert len(learned_round_modules) == num_round_modules

    @pytest.mark.parametrize(
        "learned_round_class, learned_round_loss_class", [(AutoRound, AutoRoundLoss)])
    @pytest.mark.parametrize(
        "block_strs, num_round_modules", [([], 0), (["hidden_mlp"], 2),
                                          (["in_proj_mlp", "out_proj_mlp"], 4)])
    def test_learned_round_iter_blockwise(
            self,
            quant_model,
            learned_round_class,
            learned_round_loss_class,
            block_strs,
            num_round_modules):
        # Retrieve blocks from quant model
        blocks = [getattr(quant_model, block_str) for block_str in block_strs]
        learned_round = learned_round_class(iters=100)

        # Counters to verify that the generators returns the appropiate number of items
        blocks_count = 0
        learned_round_modules_count = 0

        for (block, block_loss,
             block_learned_round_modules) in learned_round.learned_round_iterator(blocks):
            assert isinstance(block_loss, learned_round_loss_class)

            for learned_round_module in block_learned_round_modules:
                for params in learned_round_module.parameters():
                    assert params.requires_grad

            blocks_count += 1
            learned_round_modules_count += len(block_learned_round_modules)

        assert blocks_count == len(blocks)
        assert learned_round_modules_count == num_round_modules

    @pytest.mark.parametrize(
        "learned_round_class, learned_round_loss_class", [(AutoRound, AutoRoundLoss),
                                                          (AdaRound, AdaRoundLoss)])
    def test_learned_round_iter_layerwise(
            self, quant_model, learned_round_class, learned_round_loss_class):
        # Retrieve blocks from quant model
        blocks = [module for module in quant_model.modules() if isinstance(module, QuantWBIOL)]
        learned_round = learned_round_class(iters=100)

        # Counters to verify that the generators returns the appropiate number of items
        blocks_count = 0
        learned_round_modules_count = 0

        for (block, block_loss,
             block_learned_round_modules) in learned_round.learned_round_iterator(blocks):
            assert isinstance(block_loss, learned_round_loss_class)

            for learned_round_module in block_learned_round_modules:
                for params in learned_round_module.parameters():
                    assert params.requires_grad

            blocks_count += 1
            learned_round_modules_count += len(block_learned_round_modules)

        assert blocks_count == len(blocks)
        assert learned_round_modules_count == len(blocks)
