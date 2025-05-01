# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
import copy
from dataclasses import dataclass
import logging
import os
import platform
import shutil
from typing import Dict, List, Optional
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import onnx
from packaging import version
import pytest
import pytest_cases
import torch
from transformers import AutoTokenizer

from brevitas import config
from brevitas import torch_version
from brevitas_examples.llm.llm_quant.data import _tokenize_and_group_texts
from brevitas_examples.llm.llm_quant.data import get_wikitext2
from brevitas_examples.llm.main import main
from brevitas_examples.llm.main import parse_args
from brevitas_examples.llm.main import quantize_llm
from tests.marker import jit_disabled_for_dynamic_quant_act
from tests.marker import jit_disabled_for_export
from tests.marker import requires_pt_ge

BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1


class TestBatchEncoding:

    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids

    def __getitem__(self, item):
        assert item == "input_ids"
        return self.input_ids


class TestTokenizer:

    def __init__(
        self,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def batch_encode_plus(self, texts: List[str], add_special_tokens: bool,
                          **kwargs) -> Dict[str, list]:
        # Per-character tokenizer
        return {
            "input_ids": [([self.bos_token_id] if
                           (self.bos_token_id is not None and add_special_tokens) else []) +
                          list(map(ord, text)) for text in texts]}

    # Mimics the output of the tokenizer when called with return_tensors='pt'
    def __call__(self, text: str, **kwargs) -> torch.Tensor:
        return TestBatchEncoding(
            torch.tensor(
                self.batch_encode_plus([text], add_special_tokens=False)["input_ids"],
                dtype=torch.int64))


# Expected results for test_clm_tokenization. The nesting order corresponds to bos_preprocessing,
# fuse_documents, add_eos_token
EXPECTED_TOKENIZED_TEXTS = {
    "none": {
        False: {
            False: [np.array([98, 98])],
            True: [np.array([97, EOS_TOKEN_ID]), np.array([98, 98]), np.array([98, EOS_TOKEN_ID])],
        },
        True: {
            False: [np.array([97, 98]), np.array([98, 98])],
            True: [np.array([97, EOS_TOKEN_ID]), np.array([98, 98]), np.array([98, EOS_TOKEN_ID])],
        },},
    "sequence": {
        False: {
            False: [
                np.array([BOS_TOKEN_ID, 97]),
                np.array([BOS_TOKEN_ID, 98]),
                np.array([BOS_TOKEN_ID, 98]),
                np.array([BOS_TOKEN_ID, 98])],
            True: [
                np.array([BOS_TOKEN_ID, 97]),
                np.array([BOS_TOKEN_ID, EOS_TOKEN_ID]),
                np.array([BOS_TOKEN_ID, 98]),
                np.array([BOS_TOKEN_ID, 98]),
                np.array([BOS_TOKEN_ID, 98]),
                np.array([BOS_TOKEN_ID, EOS_TOKEN_ID])],},
        True: {
            False: [
                np.array([BOS_TOKEN_ID, 97]),
                np.array([BOS_TOKEN_ID, 98]),
                np.array([BOS_TOKEN_ID, 98]),
                np.array([BOS_TOKEN_ID, 98])],
            True: [
                np.array([BOS_TOKEN_ID, 97]),
                np.array([BOS_TOKEN_ID, EOS_TOKEN_ID]),
                np.array([BOS_TOKEN_ID, 98]),
                np.array([BOS_TOKEN_ID, 98]),
                np.array([BOS_TOKEN_ID, 98]),
                np.array([BOS_TOKEN_ID, EOS_TOKEN_ID])],}},
    "document": {
        False: {
            False: [np.array([BOS_TOKEN_ID, 97]), np.array([BOS_TOKEN_ID, 98]), np.array([98, 98])],
            True: [np.array([BOS_TOKEN_ID, 97]), np.array([BOS_TOKEN_ID, 98]), np.array([98, 98])],
        },
        True: {
            False: [np.array([BOS_TOKEN_ID, 97]), np.array([BOS_TOKEN_ID, 98]), np.array([98, 98])],
            True: [
                np.array([BOS_TOKEN_ID, 97]),
                np.array([EOS_TOKEN_ID, BOS_TOKEN_ID]),
                np.array([98, 98]),
                np.array([98, EOS_TOKEN_ID])],}}}


@pytest.mark.llm
@pytest_cases.parametrize("bos_token_id", [None, BOS_TOKEN_ID], ids=lambda x: f"bos={x}")
@pytest_cases.parametrize("eos_token_id", [None, EOS_TOKEN_ID], ids=lambda x: f"eos={x}")
@pytest_cases.parametrize(
    "bos_preprocessing", [None, "document", "sequence"], ids=lambda x: f"preprocessing={x}")
@pytest_cases.parametrize("add_eos_token", [False, True], ids=lambda x: f"add_eos={x}")
@pytest_cases.parametrize("fuse_documents", [False, True], ids=lambda x: f"fuse={x}")
def test_clm_tokenization(
        bos_token_id, eos_token_id, bos_preprocessing, fuse_documents, add_eos_token):
    # Sample texts
    texts = ["", "a", "", "bbb"]
    # Instantiate test tokenizer
    tokenizer = TestTokenizer(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    # Expected results
    expected_tokenized_text = EXPECTED_TOKENIZED_TEXTS[
        "none" if bos_token_id is None or bos_preprocessing is None else bos_preprocessing][
            fuse_documents][add_eos_token and eos_token_id is not None]
    tokenized_text = _tokenize_and_group_texts(
        texts=texts,
        tokenizer=tokenizer,
        sequence_length=2,
        bos_preprocessing=bos_preprocessing,
        fuse_documents=fuse_documents,
        add_eos_token=add_eos_token,
    )["input_ids"]
    assert all(map(lambda x: np.array_equal(*x), zip(expected_tokenized_text, tokenized_text)))


@pytest.mark.llm
@pytest_cases.parametrize("add_bos_token", [False, True], ids=lambda x: f"add_bos={x}")
@pytest_cases.parametrize("split", ["train", "validation"], ids=lambda x: f"split={x}")
def test_wikitext2_tokenization(add_bos_token, split):
    # Texts following Wikitext2 format
    texts = ["=a=", "", "bb"]
    raw_dataset = Dataset.from_dict({
        "text": texts,})
    # Instantiate test tokenizer
    tokenizer = TestTokenizer(bos_token_id=BOS_TOKEN_ID,)
    expected_tokenized_texts = {
        "train": {
            False: [
                torch.tensor([[61, 97, 61, 10, 10]], dtype=torch.int64),
                torch.tensor([[10, 10, 10, 98, 98]], dtype=torch.int64)],
            True: [
                torch.tensor([[BOS_TOKEN_ID, 61, 97, 61, 10]], dtype=torch.int64),
                torch.tensor([[BOS_TOKEN_ID, 10, 10, 10, 98]], dtype=torch.int64)]},
        "validation": {
            False: [torch.tensor([[61, 97, 61, 10, 10]], dtype=torch.int64)],
            True: [
                torch.tensor([[BOS_TOKEN_ID, 61, 97, 61, 10]], dtype=torch.int64),
                torch.tensor([[BOS_TOKEN_ID, 10, 10, 10, 98]],
                             dtype=torch.int64)]}}[split][add_bos_token]
    with patch('brevitas_examples.llm.llm_quant.data.random.randint', side_effect=[0, 4]):
        tokenized_texts = get_wikitext2(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            seqlen=5,
            nsamples=2,
            split=split,
            add_bos_token=add_bos_token,
        )
    for tokenized_text, expected_tokenized_text in zip(tokenized_texts, expected_tokenized_texts):
        assert torch.equal(tokenized_text["input_ids"], expected_tokenized_text)
