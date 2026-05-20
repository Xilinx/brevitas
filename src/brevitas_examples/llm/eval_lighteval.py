# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

#Adapted from https://github.com/huggingface/lighteval, released under the following LICENSE:

# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
import os
import pathlib
import re

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics.metrics import Metrics
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_loader import TransformersModel
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager
from lighteval.pipeline import Pipeline
from lighteval.pipeline import PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.tasks.requests import SamplingMethod
from torch import nn
from transformers import AutoTokenizer

### LightEval Custom Tasks

# In most recent versions of lighteval, some tasks have been changed, differing from what lm_eval does
# and from previous versions of lighteval itself.
# These custom tasks (and relative utility functions) restore those, waiting for lighteval to restore
# them natively.


def hellaswag_preprocess(
    text: str,
    wikihow_artifacts: list[str] = [" [title]"],
    truncate_dots: bool = False,
    strip_text: bool = False,
    dot_replacement: str = ". ",
):
    """Comes from LM Eval Harness"""
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    for wikihow_artifact in wikihow_artifacts:
        text = text.replace(wikihow_artifact, dot_replacement)
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    if truncate_dots:
        text = text.replace(r"\.+", r"\.")
    if strip_text:
        text = text.strip()
    return text


def hellaswag_harness(line, task_name: str = None):
    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=hellaswag_preprocess(line["activity_label"] + ": " + ctx),
        choices=[hellaswag_preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
        # "metric": "choices_loglikelihood",
    )


def piqa_harness(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['goal']}\nAnswer:",
        choices=[f" {line['sol1']}", f" {line['sol2']}"],
        gold_index=int(line["label"]),  # "metric": "choices_loglikelihood",
    )


# For HS, we have always tested against validation split
hellaswag_lm_eval = LightevalTaskConfig(
    name="hellaswag_lm_eval",
    prompt_function=hellaswag_harness,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,],
    stop_sequence=["\n"],
    version=0,
)

# For PIQA, most recent lighteval tests against validation and test split.
# Previous versions of lighteval tested only against validation.
piqa_lm_eval = LightevalTaskConfig(
    name="piqa_lm_eval",
    prompt_function=piqa_harness,
    hf_repo="ybisk/piqa",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metrics=[
        Metrics.loglikelihood_acc,],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [hellaswag_lm_eval, piqa_lm_eval]

### End of LightEval custom tasks


class BrevitasPromptManager(PromptManager):
    """Task-type-aware PromptManager that handles reasoning models like Qwen3.

    Reasoning models (e.g. Qwen3) have two problems with lighteval's default PromptManager:

    1. **Loglikelihood tasks**: When a chat template is used, Qwen3's template ends
       the prompt with ``<|im_start|>assistant\n``, at which point the model's probability
       distribution heavily favours ``<think>`` as the next token.  Passing
       ``enable_thinking=False`` makes it worse by injecting an empty
       ``<think>\\n\\n</think>\\n\\n`` block between context and continuation, corrupting
       the loglikelihood computation.  Plain-text formatting avoids both issues.
    2. **Generative tasks** (e.g. GSM8K): Instruct-tuned models need the chat template
       to produce useful output, but thinking mode must be suppressed so the model does
       not waste the token budget on ``<think>...</think>`` blocks.

    This subclass inspects ``doc.sampling_methods`` and routes accordingly:

    * ``LOGPROBS`` / ``PERPLEXITY`` → plain-text formatting (no chat template).
    * ``GENERATIVE`` → chat template with ``enable_thinking=False``.

    For non-reasoning models the ``enable_thinking`` kwarg is silently ignored by Jinja2,
    so this is safe to use unconditionally.
    """

    def prepare_prompt(self, doc: Doc) -> str:
        is_generative = SamplingMethod.GENERATIVE in doc.sampling_methods
        if is_generative and self.use_chat_template:
            return self._prepare_chat_template_no_thinking(doc)
        else:
            # For loglikelihood / perplexity tasks, always use plain text so
            # that no thinking block or chat framing interferes with the
            # probability computation over continuation tokens.
            return self._prepare_plain_text(doc)

    def _prepare_chat_template_no_thinking(self, doc: Doc) -> str:
        """Format using the chat template with thinking mode explicitly disabled."""
        orig_apply = self.tokenizer.apply_chat_template
        try:
            self.tokenizer.apply_chat_template = partial(orig_apply, enable_thinking=False)
            return self._prepare_chat_template(doc)
        finally:
            self.tokenizer.apply_chat_template = orig_apply


def filter_results(results, tasks):
    # filter out what we actually want to track
    eval_results = dict()
    for task_name in tasks:
        # log all result metrics we have for this task
        for key, val in results["results"][task_name].items():
            if not isinstance(val, str):
                # for mmlu, we don't log results per subtask, but simply overall results
                name = f"{task_name}_{key}"
                eval_results[name] = val
    return eval_results


# TODO: Although we have update lighteval and transformers version, we still rely on custom BrevitasPipeline
# This allows us to set `_cache ` to None and re-compute results based on current configuration, rather
# than loading pre-computed results
class BrevitasPipeline(Pipeline):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model._cache = None

    def _init_model(self, model_config, model):
        # Verify that both the model and model_config are passed
        assert model is not None and model_config is not None, "Provide both a model and a model config."
        assert not isinstance(model, LightevalModel), "A LigthevalModel and a model config cannot be provided simultaneously."

        # Retrieve the original pad_token before lighteval overwrites it.
        # lighteval unconditionally does `tokenizer.pad_token = tokenizer.eos_token`
        # which is wrong for models like Qwen3 that define a distinct pad_token.
        tokenizer_name = model_config.tokenizer or model_config.model_name
        original_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        original_pad_token = original_tokenizer.pad_token

        wrapped_model = TransformersModel.from_model(
            model=model,
            config=model_config,
            accelerator=self.accelerator,
        )

        # Restore the original pad_token if the model explicitly defined one
        if original_pad_token is not None:
            wrapped_model._tokenizer.pad_token = original_pad_token

        # Replace the prompt manager with a task-type-aware variant.
        # - Loglikelihood tasks use plain text (no chat template) so that
        #   thinking tokens / chat framing do not corrupt probability computations.
        # - Generative tasks use the chat template with thinking mode disabled
        #   so instruct models get proper formatting without wasting the token
        #   budget on <think>...</think> blocks.
        wrapped_model.prompt_manager = BrevitasPromptManager(
            use_chat_template=wrapped_model.use_chat_template,
            tokenizer=wrapped_model.tokenizer,
            system_prompt=model_config.system_prompt,
        )

        return wrapped_model


def run_lighteval(
    model_name: str,
    model: nn.Module,
    tasks: list[str],
    output_dir: str = "./results",
    dtype: str | None = None,
    batch_size: int | None = None,
    max_samples: int | None = None,
):
    """Evaluate model using HuggingFace Lighteval with accelerate as backend.

    Returns:
        results (dict): Evaluation results containing metrics and scores for all tasks.
    """

    evaluation_tracker = EvaluationTracker(output_dir=output_dir, save_details=True)
    parent_folder = pathlib.Path(os.path.abspath(__file__)).parent
    full_path = os.path.join(parent_folder, 'eval_lighteval.py')

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=max_samples,
        custom_tasks_directory=full_path)

    model_config = TransformersModelConfig(
        model_name=model_name, dtype=dtype, batch_size=batch_size, model_parallel=True)

    # Pipeline expects a comma-separated list of tasks
    tasks = ",".join(tasks)

    pipeline = BrevitasPipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model=model,
        model_config=model_config,
    )

    pipeline.evaluate()

    results = pipeline.get_results()
    results = filter_results(results, list(results["results"].keys()))

    return results
