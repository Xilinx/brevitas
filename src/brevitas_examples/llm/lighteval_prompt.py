# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.tasks.requests import SamplingMethod


class BrevitasPromptManager(PromptManager):
    """Format LightEval prompts according to request type and thinking policy."""

    def __init__(self, *args, generation_thinking: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generation_thinking = generation_thinking

    def prepare_prompt(self, doc: Doc) -> str:
        is_generative = SamplingMethod.GENERATIVE in doc.sampling_methods
        is_likelihood = any(
            method in doc.sampling_methods
            for method in (SamplingMethod.LOGPROBS, SamplingMethod.PERPLEXITY))
        if is_generative and is_likelihood:
            raise ValueError(
                f"Task document '{doc.task_name}' mixes generative and likelihood requests, "
                'which require incompatible prompt formatting.')
        if is_generative and self.use_chat_template:
            return self._prepare_generation_prompt(doc)
        return self._prepare_plain_text(doc)

    def _prepare_generation_prompt(self, doc: Doc) -> str:
        original_apply = self.tokenizer.apply_chat_template
        try:
            self.tokenizer.apply_chat_template = partial(
                original_apply, enable_thinking=self.generation_thinking)
            return self._prepare_chat_template(doc)
        finally:
            self.tokenizer.apply_chat_template = original_apply
