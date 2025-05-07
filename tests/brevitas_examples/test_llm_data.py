# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, List, Optional
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import pytest_cases
import torch

from brevitas_examples.llm.llm_quant.data import get_wikitext2
from brevitas_examples.llm.llm_quant.data import tokenize_and_group_texts

# Identifiers for the special tokens of DummyTokenizer
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1


# Mimics a part of the functionality of the class BatchEncoding in transformers.tokenization_utils_base,
# since an instance of it is returned by the __call__of PreTrainedTokenizerBase
class DummyBatchEncoding:

    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids

    def __getitem__(self, item) -> torch.Tensor:
        assert item == "input_ids"
        return self.input_ids


# Sample tokenizer which maps to each character in a string to its integer representation
class DummyTokenizer:

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

    def __call__(self, text: str, **kwargs) -> torch.Tensor:
        return DummyBatchEncoding(
            torch.tensor(
                self.batch_encode_plus([text], add_special_tokens=False)["input_ids"],
                dtype=torch.int64))


# Expected results for test_clm_tokenization. The nesting order corresponds to bos_preprocessing,
# fuse_documents, add_eos_token
EXPECTED_CLM_TOKENIZED_TEXTS = {
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


@pytest_cases.parametrize("bos_token_id", [None, BOS_TOKEN_ID], ids=lambda x: f"bos={x}")
@pytest_cases.parametrize("eos_token_id", [None, EOS_TOKEN_ID], ids=lambda x: f"eos={x}")
@pytest_cases.parametrize(
    "bos_preprocessing", [None, "document", "sequence"], ids=lambda x: f"preprocessing={x}")
@pytest_cases.parametrize("add_eos_token", [False, True], ids=lambda x: f"add_eos={x}")
@pytest_cases.parametrize("fuse_documents", [False, True], ids=lambda x: f"fuse={x}")
def test_clm_tokenization(
        bos_token_id: Optional[int],
        eos_token_id: Optional[int],
        bos_preprocessing: bool,
        fuse_documents: bool,
        add_eos_token: bool):
    texts = ["", "a", "", "bbb"]
    tokenizer = DummyTokenizer(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    expected_tokenized_text = EXPECTED_CLM_TOKENIZED_TEXTS[
        "none" if bos_token_id is None or bos_preprocessing is None else bos_preprocessing][
            fuse_documents][add_eos_token and eos_token_id is not None]
    tokenized_text = tokenize_and_group_texts(
        texts=texts,
        tokenizer=tokenizer,
        sequence_length=2,
        bos_preprocessing=bos_preprocessing,
        fuse_documents=fuse_documents,
        add_eos_token=add_eos_token,
    )["input_ids"]
    assert all(map(lambda x: np.array_equal(*x), zip(expected_tokenized_text, tokenized_text)))


@pytest_cases.parametrize("add_bos_token", [False, True], ids=lambda x: f"add_bos={x}")
@pytest_cases.parametrize("split", ["train", "validation"], ids=lambda x: f"split={x}")
def test_wikitext2_tokenization(add_bos_token: bool, split: str):
    # Texts following Wikitext2 format
    texts = ["=a=", "", "bb"]
    raw_dataset = Dataset.from_dict({
        "text": texts,})
    # Instantiate test tokenizer
    tokenizer = DummyTokenizer(bos_token_id=BOS_TOKEN_ID,)
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
