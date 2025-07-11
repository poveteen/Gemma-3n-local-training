"""Dataset helpers for Gemma 3n CPU Training and Finetuning.

Currently supports JSONL files with the following schema per line:
{"instruction": "<instruction>", "response": "<target response>"}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

__all__ = [
    "load_jsonl_dataset",
    "prepare_dataset",
]


def load_jsonl_dataset(path: str | Path) -> Dataset:
    """Load a JSONL dataset using 🤗 Datasets.

    Args:
        path: Path to the JSONL file.

    Returns:
        A ``datasets.Dataset`` object.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)

    dataset = load_dataset("json", data_files=path, split="train")
    return dataset


def _format_example(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """Tokenize a single example for causal LM fine-tuning."""
    prompt = example.get("instruction", "")
    response = example.get("response", "")
    text = f"<|user|> {prompt}\n<|assistant|> {response}"  # Simple chat template

    # Avoid OverflowError when model_max_length is an absurdly large default (e.g. 2**63-1)
    max_len = tokenizer.model_max_length
    if max_len is None or max_len > 4096:  # hard-cap for CPU training
        max_len = 4096

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_attention_mask=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def prepare_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    num_proc: int | None = None,
) -> Dataset:
    """Tokenize and prepare the dataset for training.

    The function maps over the entire dataset to add ``input_ids``, ``attention_mask``, and ``labels``.
    Uses multiprocessing if ``num_proc`` is provided.
    """
    return dataset.map(
        lambda ex: _format_example(ex, tokenizer),
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )
