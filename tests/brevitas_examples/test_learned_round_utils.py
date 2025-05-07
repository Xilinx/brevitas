# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Union

from accelerate.utils.operations import send_to_device
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
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.inject.enum import LearnedRoundImplType
import brevitas.nn as qnn
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.quant_tensor.base_quant_tensor import QuantTensor
from brevitas_examples.common.learned_round.learned_round_method import LearnedRound
from brevitas_examples.common.learned_round.learned_round_optimizer import get_blocks
from brevitas_examples.common.learned_round.learned_round_optimizer import save_inputs_output

config.IGNORE_MISSING_KEYS = True


class QuantBlock(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.layer1 = qnn.QuantLinear(in_features=in_features, out_features=hidden_dim)
        self.layer2 = qnn.QuantLinear(in_features=hidden_dim, out_features=out_features)
        self.relu = qnn.QuantReLU(return_quant_tensor=True)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return self.relu(out)


class DummyQuantModel(nn.Module):

    def __init__(self, in_features: int, out_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj_mlp = QuantBlock(
            in_features=in_features, hidden_dim=hidden_dim, out_features=hidden_dim)
        self.hidden_mlp = QuantBlock(
            in_features=hidden_dim, hidden_dim=hidden_dim, out_features=hidden_dim)
        self.out_proj_mlp = QuantBlock(
            in_features=hidden_dim, hidden_dim=hidden_dim, out_features=out_features)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.in_proj_mlp(x, block1_kwarg=0., **kwargs)
        out = self.hidden_mlp(out, block2_kwarg=0., **kwargs)
        return self.out_proj_mlp(out, block3_kwarg=0., **kwargs)


class Block(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.layer2 = nn.Linear(in_features=hidden_dim, out_features=out_features)
        self.relu = F.relu

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return self.relu(out)


class DummyModel(nn.Module):

    def __init__(self, in_features: int, out_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj_mlp = Block(
            in_features=in_features, hidden_dim=hidden_dim, out_features=hidden_dim)
        self.hidden_mlp = Block(
            in_features=hidden_dim, hidden_dim=hidden_dim, out_features=hidden_dim)
        self.out_proj_mlp = Block(
            in_features=hidden_dim, hidden_dim=hidden_dim, out_features=out_features)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.in_proj_mlp(x, block1_kwarg=0., **kwargs)
        out = self.hidden_mlp(out, block2_kwarg=0., **kwargs)
        return self.out_proj_mlp(out, block3_kwarg=0., **kwargs)


class DummyDataset(Dataset):

    def __init__(self):
        self.data = [
            {
                "x": torch.tensor([1.0, 2.0]),
                "tensor": torch.tensor([0.0]),
                "bool": True,
                "float": 0.0,},]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DummyCache:

    def __init__(self) -> None:
        self.args = []
        self.kwargs = []
        self.output = []

    def __len__(self) -> int:
        return len(self.args)

    def store_inputs(self, args: Any, kwargs: Any) -> None:
        self.args.append(args)
        self.kwargs.append(kwargs)

    def store_output(self, output: Any) -> None:
        self.output.append(output)

    def sample_batch(self, indices: torch.Tensor) -> Union[Any, torch.Tensor]:
        pass

    def initialize_cache(self) -> None:
        pass

    def clear_cache(self) -> None:
        pass

    def reset_cache(self) -> None:
        pass

    def cache_to_dataset(self) -> Dataset:
        pass

    def collate_fn(self, batch: Any) -> Any:
        pass


class TestLearnedRound:

    @fixture
    def quant_model():
        return DummyQuantModel(in_features=2, out_features=1, hidden_dim=4)

    @fixture
    def model():
        return DummyModel(in_features=2, out_features=1, hidden_dim=4)

    @fixture
    def data_loader():
        return DataLoader(DummyDataset(), batch_size=1, shuffle=False)

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
    @pytest.mark.parametrize("store_inputs", [True, False])
    @pytest.mark.parametrize("store_output", [True, False])
    @pytest.mark.parametrize("keep_gpu", [True, False])
    @pytest.mark.parametrize("disable_quant", [True, False])
    def test_save_inp_out_data(
            self, model, quant_model, data_loader, store_inputs, store_output, keep_gpu,
            disable_quant):

        def _compare_tensors(cache_tensor, gt_tensor, disable_quant, keep_gpu):
            # The elements should be of the same type
            assert isinstance(cache_tensor, torch.Tensor) if disable_quant else isinstance(
                cache_tensor, QuantTensor)
            # If they are QuantTensors extract their data
            if isinstance(cache_tensor, QuantTensor):
                cache_tensor, gt_tensor = cache_tensor.value, gt_tensor.value
            # Verify that the tensor is in GPU if keep_gpu=True
            assert keep_gpu == cache_tensor.is_cuda
            # Make sure tensors are in the same device before comparison
            cache_tensor = cache_tensor.cpu()
            gt_tensor = gt_tensor.cpu()
            # Check that their contents match
            assert torch.allclose(cache_tensor, gt_tensor)
            # Return True to be used in the helper
            return True

        def model_forward(model, inputs):
            device = next(model.parameters()).device
            inputs = send_to_device(inputs, device)
            model(**inputs)

        # Make sure that the quant and FP models share the same weights
        quant_model.load_state_dict(model.state_dict())

        # Prepare models
        model.eval()
        model = model.cuda()
        quant_model.eval()
        quant_model = quant_model.cuda()

        device = next(quant_model.parameters()).device

        # Compute ground-truth inputs/outputs
        fp_args, fp_kwargs, fp_outs = [], [], []
        quant_args, quant_kwargs, quant_outs = [], [], []
        # Compute ground truths inputs and outputs
        with torch.no_grad():
            for inputs in data_loader:
                inputs = send_to_device(inputs, device)
                kwargs = {k: v for k, v in inputs.items() if k != "x"}
                # Compute quant inputs to module
                quant_arg = quant_model.in_proj_mlp(**inputs)
                quant_kwarg = {"block2_kwarg": 0.0, **kwargs}
                # Compute quant outputs of module
                quant_out = quant_model.hidden_mlp(quant_arg, **quant_kwarg)

                quant_args.append((quant_arg,))
                quant_kwargs.append(quant_kwarg)
                quant_outs.append(quant_out)

                # Compute quant inputs to module
                fp_arg = model.in_proj_mlp(**inputs)
                fp_kwarg = {"block2_kwarg": 0.0, **kwargs}
                # Compute quant outputs of module
                fp_out = model.hidden_mlp(fp_arg, **fp_kwarg)

                fp_args.append((fp_arg,))
                fp_kwargs.append(fp_kwarg)
                fp_outs.append(fp_out)

        # Prepare to capture inputs/outputs using DataSaverHook
        cache = DummyCache()

        # Retrieve module from quant_model
        module = quant_model.hidden_mlp

        # Make call to save inputs/outputs
        save_inputs_output(
            model=quant_model,
            model_forward=model_forward,
            module=module,
            dataloader=data_loader,
            cache=cache,
            store_inputs=store_inputs,
            store_output=store_output,
            keep_gpu=keep_gpu,
            disable_quant=disable_quant,
        )

        # Verify that the lengths of the lists match
        if store_inputs:
            assert len(cache.args) == len(fp_args) and len(cache.kwargs) == len(fp_kwargs)
        else:
            assert len(cache.args) == 0 and len(cache.kwargs) == 0

        if store_output:
            assert len(cache.output) == len(fp_outs)
        else:
            assert len(cache.output) == 0

        # Verify that the arguments are the same
        for cache_arg, gt_arg in zip(cache.args, fp_args if disable_quant else quant_args):
            _compare_tensors(cache_arg[0], gt_arg[0], disable_quant, keep_gpu)

        for cache_kwarg, gt_kwarg in zip(cache.kwargs, fp_kwargs if disable_quant else quant_kwargs):
            # Compare the contents within each dictionary
            same_contents = all((
                torch.allclose(cache_kwarg.get(gt_kwarg_k, None).cpu(), gt_kwarg_v.cpu(
                )) if isinstance(gt_kwarg_v, torch.Tensor) else cache_kwarg.get(gt_kwarg_k, None) ==
                gt_kwarg_v) for gt_kwarg_k,
                                gt_kwarg_v in gt_kwarg.items())
            # Verify that the dictionaries have the same keys and content
            assert set(cache_kwarg.keys()) == set(gt_kwarg.keys()) and same_contents

        # Verify that the outputs are the same
        for cache_output, gt_output in zip(cache.output, fp_outs if disable_quant else quant_outs):
            _compare_tensors(cache_output, gt_output, disable_quant, keep_gpu)

    @pytest.mark.parametrize(
        "learned_round",
        [
            LearnedRound(learned_round_impl_type=LearnedRoundImplType.IDENTITY),
            LearnedRound(learned_round_impl_type=LearnedRoundImplType.HARD_SIGMOID)])
    def test_insert_learned_round_quantizers(self, quant_model, learned_round):
        block = quant_model.in_proj_mlp
        learned_round.insert_learned_round_quantizers(block)

        for module in block.modules():
            if hasattr(module, "weight_quant"):
                assert module.weight_quant.rounding_mode == "LEARNED_ROUND"
                assert isinstance(
                    module.weight_quant.tensor_quant.int_quant.float_to_int_impl, LearnedRoundSte)

    @pytest.mark.parametrize(
        "learned_round",
        [
            LearnedRound(learned_round_impl_type=LearnedRoundImplType.IDENTITY),
            LearnedRound(learned_round_impl_type=LearnedRoundImplType.HARD_SIGMOID)])
    @pytest.mark.parametrize(
        "block_strs, num_round_modules", [([], 0), (["hidden_mlp"], 2),
                                          (["in_proj_mlp", "out_proj_mlp"], 4)])
    def test_return_learned_round_quantizers(
            self, quant_model, learned_round, block_strs, num_round_modules):
        # Inject quantizers in quant model
        for block_str in block_strs:
            block = getattr(quant_model, block_str)
            learned_round.insert_learned_round_quantizers(block)
        learned_round_modules = learned_round.return_learned_round_quantizers(quant_model)
        assert len(learned_round_modules) == num_round_modules
