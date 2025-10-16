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

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_loader import TransformersModel
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager
from lighteval.pipeline import Pipeline
from lighteval.pipeline import PipelineParameters
from torch import nn


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


# TODO (pml): The implementation of `_init_model` in `BrevitasPipeline` mimics that
# of `Pipeline` in `lighteval>=0.11.0`. However, `lighteval>=0.11.0` requires
# `transformers>=4.54.0`, which clashes with the pin `transformers[sentencepiece]==4.50.0`
# in `requirements-llm.txt`. Therefore, `BrevitasPipeline` can only be removed once the
# `transformers` requirement is upgraded to `transformers>=4.54.0`. The following
# provides extra context on the changes to _init_model: https://github.com/huggingface/lighteval/pull/921.
class BrevitasPipeline(Pipeline):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=max_samples,
    )

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
