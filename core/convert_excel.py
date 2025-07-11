"""Utility script to convert WHO SDG Excel data to JSONL instruction/response pairs.

Example usage:
    python -m core.convert_excel \
        --input "WHO - SDG - data.xlsx" \
        --output_train ./data/train.jsonl \
        --output_val ./data/val.jsonl \
        --val_split 0.05

The script constructs a simple natural-language question from each row and
uses the corresponding numeric *estimate* as the answer.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

# Instruction template. You can tweak as needed.
QUESTION_TMPL = (
    "For {setting} in {date}, what was the value of '{indicator_name}' "
    "for subgroup '{subgroup}'?"
)


def build_example(row) -> dict:
    """Construct a dict with `instruction` and `response` keys."""
    estimate = row.get("estimate")
    if pd.isna(estimate):
        return {}

    instruction = QUESTION_TMPL.format(
        setting=row.get("setting", "[Unknown]"),
        date=row.get("date", "[Unknown]"),
        indicator_name=row.get("indicator_name", "an indicator"),
        subgroup=row.get("subgroup", "all population"),
    )
    # Response could be formatted with more context or units; keep simple.
    response = str(estimate)
    return {"instruction": instruction, "response": response}


def convert(
    input_path: str | Path,
    output_train: str | Path,
    output_val: str | Path,
    val_split: float = 0.05,
    shuffle: bool = True,
    max_rows: int | None = None,
):
    df = pd.read_excel(input_path)
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if max_rows:
        df = df.head(max_rows)

    examples: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building examples"):
        ex = build_example(row)
        if ex:
            examples.append(ex)

    split_idx = int(len(examples) * (1 - val_split))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    def dump(lines: List[dict], path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            for item in lines:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    dump(train_examples, output_train)
    dump(val_examples, output_val)
    print(
        f"Wrote {len(train_examples)} train and {len(val_examples)} val examples "
        f"to {output_train}, {output_val}"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Convert WHO SDG Excel to JSONL")
    p.add_argument("--input", required=True, help="Path to Excel file (.xlsx)")
    p.add_argument("--output_train", default="./data/train.jsonl", help="Output train JSONL")
    p.add_argument("--output_val", default="./data/val.jsonl", help="Output val JSONL")
    p.add_argument("--val_split", type=float, default=0.05, help="Validation split fraction")
    p.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optionally limit number of rows for quick experiments",
    )
    return p.parse_args()


def main():
    args = parse_args()
    convert(
        input_path=args.input,
        output_train=args.output_train,
        output_val=args.output_val,
        val_split=args.val_split,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
