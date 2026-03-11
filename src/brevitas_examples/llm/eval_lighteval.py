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
from lighteval.tasks.requests import Doc
from torch import nn

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

        return TransformersModel.from_model(
            model=model,
            config=model_config,
            accelerator=self.accelerator,
        )


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
