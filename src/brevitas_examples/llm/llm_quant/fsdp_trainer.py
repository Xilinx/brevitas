# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

from collections.abc import Mapping
import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union
import warnings

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

from huggingface_hub import create_repo
from huggingface_hub import ModelCard
from huggingface_hub import upload_folder
import huggingface_hub.utils as hf_hub_utils
import numpy as np
from packaging import version
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.data.data_collator import default_data_collator
from transformers.debug_utils import DebugOption
from transformers.debug_utils import DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS
from transformers.hyperparameter_search import default_hp_search_backend
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import deepspeed_init
from transformers.integrations.deepspeed import deepspeed_load_checkpoint
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor
from transformers.optimization import get_scheduler
from transformers.processing_utils import ProcessorMixin
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_3
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_callback import DefaultFlowCallback
from transformers.trainer_callback import ExportableState
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import distributed_broadcast_scalars
from transformers.trainer_pt_utils import distributed_concat
from transformers.trainer_pt_utils import DistributedTensorGatherer
from transformers.trainer_pt_utils import EvalLoopContainer
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_pt_utils import LayerWiseDummyOptimizer
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_pt_utils import nested_concat
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_pt_utils import nested_numpify
from transformers.trainer_pt_utils import nested_xla_mesh_reduce
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_pt_utils import SequentialDistributedSampler
from transformers.trainer_utils import BestRun
from transformers.trainer_utils import check_target_module_exists
from transformers.trainer_utils import default_compute_objective
from transformers.trainer_utils import denumpify_detensorize
from transformers.trainer_utils import enable_full_determinism
from transformers.trainer_utils import EvalLoopOutput
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_utils import find_executable_batch_size
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_utils import has_length
from transformers.trainer_utils import HPSearchBackend
from transformers.trainer_utils import HubStrategy
from transformers.trainer_utils import neftune_post_forward_hook
from transformers.trainer_utils import number_of_arguments
from transformers.trainer_utils import PredictionOutput
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_utils import RemoveColumnsCollator
from transformers.trainer_utils import SaveStrategy
from transformers.trainer_utils import seed_worker
from transformers.trainer_utils import set_seed
from transformers.trainer_utils import speed_metrics
from transformers.trainer_utils import TrainerMemoryTracker
from transformers.trainer_utils import TrainOutput
from transformers.training_args import OptimizerNames
from transformers.training_args import ParallelMode
from transformers.training_args import TrainingArguments
from transformers.utils import ADAPTER_CONFIG_NAME
from transformers.utils import ADAPTER_SAFE_WEIGHTS_NAME
from transformers.utils import ADAPTER_WEIGHTS_NAME
from transformers.utils import can_return_loss
from transformers.utils import CONFIG_NAME
from transformers.utils import find_labels
from transformers.utils import is_accelerate_available
from transformers.utils import is_apex_available
from transformers.utils import is_bitsandbytes_available
from transformers.utils import is_datasets_available
from transformers.utils import is_galore_torch_available
from transformers.utils import is_grokadamw_available
from transformers.utils import is_in_notebook
from transformers.utils import is_ipex_available
from transformers.utils import is_liger_kernel_available
from transformers.utils import is_lomo_available
from transformers.utils import is_peft_available
from transformers.utils import is_safetensors_available
from transformers.utils import is_sagemaker_dp_enabled
from transformers.utils import is_sagemaker_mp_enabled
from transformers.utils import is_schedulefree_available
from transformers.utils import is_torch_compile_available
from transformers.utils import is_torch_mlu_available
from transformers.utils import is_torch_mps_available
from transformers.utils import is_torch_musa_available
from transformers.utils import is_torch_neuroncore_available
from transformers.utils import is_torch_npu_available
from transformers.utils import is_torch_xla_available
from transformers.utils import is_torch_xpu_available
from transformers.utils import is_torchao_available
from transformers.utils import logging
from transformers.utils import PushInProgress
from transformers.utils import PushToHubMixin
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from transformers.utils import SAFE_WEIGHTS_NAME
from transformers.utils import strtobool
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils import WEIGHTS_NAME
from transformers.utils import XLA_FSDPV2_MIN_VERSION
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.quantization_config import QuantizationMethod

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    from torch_xla import __version__ as XLA_VERSION
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    import smdistributed.modelparallel.torch as smp

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward
    from .trainer_pt_utils import smp_forward_only
    from .trainer_pt_utils import smp_gather
    from .trainer_pt_utils import smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version
    from accelerate import Accelerator
    from accelerate import skip_first_batches
    from accelerate.state import AcceleratorState
    from accelerate.utils import DistributedDataParallelKwargs
    from accelerate.utils import DistributedType
    from accelerate.utils import load_fsdp_model
    from accelerate.utils import load_fsdp_optimizer
    from accelerate.utils import save_fsdp_model
    from accelerate.utils import save_fsdp_optimizer

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(
            inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

logger = logging.get_logger(__name__)

from transformers.trainer import Trainer


class FSDPTrainer(Trainer):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

            <Tip>

            [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
            your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
            models.

            </Tip>

        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `processing_class` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise if the processing_class is a feature extractor or tokenizer.
        train_dataset (Union[`torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `datasets.Dataset`], *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.

            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`, `datasets.Dataset`]), *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
            This supercedes the `tokenizer` argument, which is now deprecated.
        model_init (`Callable[[], PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
            from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
            be able to choose different architectures according to hyper parameters (such as layer count, sizes of
            inner layers, dropout probabilities etc).
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618) used by [`Trainer`].
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).

            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use.
            Overrides `optim` and `optim_args` in `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)

    """

    # Those are used as methods of the Trainer in examples.
    from transformers.trainer_pt_utils import _get_learning_rate
    from transformers.trainer_pt_utils import log_metrics
    from transformers.trainer_pt_utils import metrics_format
    from transformers.trainer_pt_utils import save_metrics
    from transformers.trainer_pt_utils import save_state

    @deprecate_kwarg(
        "tokenizer", new_name="processing_class", version="5.0.0", raise_if_both_names=True)
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase,
                                         BaseImageProcessor,
                                         FeatureExtractionMixin,
                                         ProcessorMixin]] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer],
                          Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[Tuple[Type[torch.optim.Optimizer], Dict[str,
                                                                                   Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor],
                                                         torch.Tensor]] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        if args.batch_eval_metrics and compute_metrics is not None:
            if "compute_result" not in inspect.signature(compute_metrics).parameters.keys():
                raise ValueError(
                    "When using `batch_eval_metrics`, your `compute_metrics` function must take a `compute_result`"
                    " boolean argument which will be triggered after the last batch of the eval set to signal that the"
                    " summary statistics should be returned by the function.")
        if args.eval_strategy is not None and args.eval_strategy != "no" and eval_dataset is None:
            raise ValueError(
                f"You have set `args.eval_strategy` to {args.eval_strategy} but you didn't pass an `eval_dataset` to `Trainer`. Either set `args.eval_strategy` to `no` or pass an `eval_dataset`. "
            )
        if args.save_strategy == SaveStrategy.BEST or args.load_best_model_at_end:
            if args.metric_for_best_model is None:
                raise ValueError(
                    "`args.metric_for_best_model` must be provided when using 'best' save_strategy or if `args.load_best_model_at_end` is set to `True`."
                )

        self.args = args
        self.compute_loss_func = compute_loss_func
        # Seed must be set before instantiating the model when using model
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(
            self.args.seed)

        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        self.create_accelerator_and_postprocess()

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method. This will become a fatal error in the next"
                    " release.",
                    FutureWarning,
                )
            self.model_init = model_init

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto")

        if getattr(model, "is_parallelizable", False) and getattr(model, "model_parallel", False):
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        if getattr(model, "hf_device_map", None) is not None:
            devices = [
                device for device in set(model.hf_device_map.values())
                if device not in ["cpu", "disk"]]
            if len(devices) > 1:
                self.is_model_parallel = True
            elif len(devices) == 1:
                self.is_model_parallel = self.args.device != torch.device(devices[0])
            else:
                self.is_model_parallel = False

            # warn users
            if self.is_model_parallel:
                logger.info(
                    "You have loaded a model on multiple GPUs. `is_model_parallel` attribute will be force-set"
                    " to `True` to avoid any unexpected behavior such as device placement mismatching."
                )

        if self.args.use_liger_kernel:
            if is_liger_kernel_available():
                from liger_kernel.transformers import _apply_liger_kernel_to_instance

                if isinstance(model, PreTrainedModel):
                    # Patch the model with liger kernels. Use the default kernel configurations.
                    _apply_liger_kernel_to_instance(model=model)
                else:
                    logger.warning(
                        "The model is not an instance of PreTrainedModel. No liger kernels will be applied."
                    )
            else:
                raise ImportError(
                    "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
                    "Please install it with `pip install liger-kernel`")

        _is_quantized_and_base_model = getattr(
            model, "is_quantized", False) and not getattr(model, "_hf_peft_config_loaded", False)
        _quantization_method_supports_training = (
            getattr(model, "hf_quantizer", None) is not None and model.hf_quantizer.is_trainable)

        _is_model_quantized_and_qat_trainable = getattr(
            model, "hf_quantizer", None) is not None and getattr(
                model.hf_quantizer, "is_qat_trainable", False)

        # Filter out quantized + compiled models
        if _is_quantized_and_base_model and hasattr(model, "_orig_mod"):
            raise ValueError(
                "You cannot fine-tune quantized model with `torch.compile()` make sure to pass a non-compiled model when fine-tuning a quantized model with PEFT"
            )

        # At this stage the model is already loaded
        if _is_quantized_and_base_model and not _is_peft_model(
                model) and not _is_model_quantized_and_qat_trainable:
            raise ValueError(
                "You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of"
                " the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft"
                " for more details")
        elif _is_quantized_and_base_model and not _quantization_method_supports_training:
            raise ValueError(
                f"The model you are trying to fine-tune is quantized with {model.hf_quantizer.quantization_config.quant_method}"
                " but that quantization method do not support training. Please open an issue on GitHub: https://github.com/huggingface/transformers"
                f" to request the support for training support for {model.hf_quantizer.quantization_config.quant_method}"
            )

        self.is_fsdp_xla_enabled = args.fsdp_config["xla"]
        if len(args.fsdp) > 0:
            if self.is_deepspeed_enabled:
                raise ValueError(
                    "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )
            if not args.fsdp_config["xla"] and args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise ValueError("Using fsdp only works in distributed training.")

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
        # 4. FSDP - same as MP
        self.place_model_on_device = args.place_model_on_device
        if (self.is_model_parallel or self.is_deepspeed_enabled or
            ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train) or
                self.is_fsdp_xla_enabled or self.is_fsdp_enabled):
            self.place_model_on_device = False

        default_collator = (
            DataCollatorWithPadding(processing_class) if processing_class is not None and
            isinstance(processing_class, (PreTrainedTokenizerBase, SequenceFeatureExtractor)) else
            default_data_collator)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

        # Bnb Quantized models doesn't support `.to` operation.
        if (self.place_model_on_device and not getattr(model, "quantization_method", None)
                == QuantizationMethod.BITS_AND_BYTES):
            self._move_model_to_device(model, args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        # Just in case the model was wrapped outside of the `Trainer`
        unwrapped_model = self.accelerator.unwrap_model(model)
        model_forward = (
            unwrapped_model.forward
            if not _is_peft_model(unwrapped_model) else unwrapped_model.get_base_model().forward)
        forward_params = inspect.signature(model_forward).parameters

        # Check if the model has explicit setup for loss kwargs,
        # if not, check if `**kwargs` are in model.forward
        if hasattr(model, "accepts_loss_kwargs"):
            self.model_accepts_loss_kwargs = model.accepts_loss_kwargs
        else:
            self.model_accepts_loss_kwargs = any(
                k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values())

        self.neftune_noise_alpha = args.neftune_noise_alpha

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = optimizer_cls_and_kwargs
        if self.optimizer_cls_and_kwargs is not None and self.optimizer is not None:
            raise RuntimeError(
                "Passing both `optimizers` and `optimizer_cls_and_kwargs` arguments is incompatible."
            )
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if is_torch_xla_available() and self.optimizer is not None:
            for param in self.model.parameters():
                model_device = param.device
                break
            for param_group in self.optimizer.param_groups:
                if len(param_group["params"]) > 0:
                    optimizer_device = param_group["params"][0].device
                    break
            if model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you"
                    " created an optimizer around your model **before** putting on the device and passing it to the"
                    " `Trainer`. Make sure the lines `import torch_xla.core.xla_model as xm` and"
                    " `model.to(xm.xla_device())` is performed before the optimizer creation in your script."
                )
        #if (self.is_fsdp_xla_enabled or self.is_fsdp_enabled) and (
        #    self.optimizer is not None or self.lr_scheduler is not None
        #):
        #    raise RuntimeError(
        #        "Passing `optimizers` is not allowed if PyTorch FSDP is enabled. "
        #        "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
        #    )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler)
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(getattr(
                self.data_collator, "collate_batch", None)):
            raise ValueError(
                "The `data_collator` should be a simple callable (function, class with `__call__`)."
            )

        if args.max_steps > 0 and args.num_train_epochs > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler.")

        if (train_dataset is not None and
                isinstance(train_dataset, torch.utils.data.IterableDataset) and
                args.group_by_length):
            raise ValueError(
                "the `--group_by_length` option is only available for `Dataset`, not `IterableDataset"
            )

        self._signature_columns = None

        # Mixed precision setup
        self.use_apex = False
        self.use_cpu_amp = False

        # Mixed precision setup for SageMaker Model Parallel
        if is_sagemaker_mp_enabled():
            # BF16 + model parallelism in SageMaker: currently not supported, raise an error
            if args.bf16:
                raise ValueError(
                    "SageMaker Model Parallelism does not support BF16 yet. Please use FP16 instead "
                )

            if IS_SAGEMAKER_MP_POST_1_10:
                # When there's mismatch between SMP config and trainer argument, use SMP config as truth
                if args.fp16 != smp.state.cfg.fp16:
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                        f"but FP16 provided in trainer argument is {args.fp16}, "
                        f"setting to {smp.state.cfg.fp16}")
                    args.fp16 = smp.state.cfg.fp16
            else:
                # smp < 1.10 does not support fp16 in trainer.
                if hasattr(smp.state.cfg, "fp16"):
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                        "but SageMaker Model Parallelism < 1.10 does not support FP16 in trainer.")
        if (args.fp16 or args.bf16) and args.half_precision_backend == "auto":
            if args.device == torch.device("cpu"):
                if args.fp16:
                    if not is_torch_greater_or_equal_than_2_3:
                        raise ValueError("Tried to use `fp16` but it is not supported on cpu")
                else:
                    args.half_precision_backend = "cpu_amp"
            logger.info(f"Using {args.half_precision_backend} half precision backend")

        if (args.fp16 or
                args.bf16) and not (self.is_deepspeed_enabled or is_sagemaker_mp_enabled()):
            # deepspeed and SageMaker Model Parallel manage their own half precision
            if args.half_precision_backend == "cpu_amp":
                self.use_cpu_amp = True
                self.amp_dtype = torch.bfloat16
            elif args.half_precision_backend == "apex":
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to"
                        " https://www.github.com/nvidia/apex.")
                self.use_apex = True

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)],
        )
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

        # very last
        self._memory_tracker.stop_and_update_metrics()

        # torch.compile
        if args.torch_compile and not is_torch_compile_available():
            raise RuntimeError("Using torch.compile requires PyTorch 2.0 or higher.")

        self.is_fsdp_xla_v2_enabled = args.fsdp_config.get("xla_fsdp_v2", False)
        if self.is_fsdp_xla_v2_enabled:
            if not IS_XLA_FSDPV2_POST_2_2:
                raise ValueError("FSDPv2 requires `torch_xla` 2.2 or higher.")
            # Prepare the SPMD mesh that is going to be used by the data loader and the FSDPv2 wrapper.
            # Tensor axis is just a placeholder where it will not be used in FSDPv2.
            num_devices = xr.global_runtime_device_count()
            xs.set_global_mesh(
                xs.Mesh(
                    np.array(range(num_devices)), (num_devices, 1), axis_names=("fsdp", "tensor")))
        self.is_fsdp_xla_v1_enabled = self.is_fsdp_xla_enabled and not self.is_fsdp_xla_v2_enabled

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """

        # Overwrite optimizer creation because optimizer is already created
        optimizer = self.optimizer
        self.create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=optimizer,
        )
