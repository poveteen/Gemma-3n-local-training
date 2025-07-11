"""Entry point for LoRA fine-tuning Gemma 3n on CPU.

Recommended usage (CPU):
$ accelerate launch core/train.py --config configs/finetune.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root on sys.path so that `import core` works when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

from core import data as data_utils
from core.utils import Config, ensure_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")





def load_model_and_tokenizer(cfg: Config):
    """Load tokenizer & model according to quantization settings."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For causal LM

    quant_args = {}
    if cfg.quantization == "int8":
        quant_args["load_in_8bit"] = True
    elif cfg.quantization == "4bit":
        quant_args["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        **quant_args,
    )


    return model, tokenizer


def apply_lora(model, cfg: Config):
    lora_cfg = cfg.lora or {}
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
    )
    logger.info("Applying LoRA to model: %s", lora_config)
    return get_peft_model(model, lora_config)


def load_datasets(cfg: Config, tokenizer) -> tuple[Dataset, Dataset | None]:
    train_ds = data_utils.load_jsonl_dataset(cfg.train_data_path)
    train_ds = data_utils.prepare_dataset(train_ds, tokenizer)

    if cfg.val_data_path:
        val_ds = data_utils.load_jsonl_dataset(cfg.val_data_path)
        val_ds = data_utils.prepare_dataset(val_ds, tokenizer)
    else:
        # Split 5% for validation if val path not supplied
        splits = train_ds.train_test_split(test_size=0.05, seed=42)
        train_ds, val_ds = splits["train"], splits["test"]

    return train_ds, val_ds


def run_training_from_app(model_name: str, train_data_path: str, output_dir: str, **kwargs):
    """Run training session from a web app, yielding logs in real-time."""
    # Create a lightweight config object from parameters
    cfg = Config(
        model_name=model_name,
        train_data_path=train_data_path,
        output_dir=output_dir,
        # Set defaults, can be overridden by kwargs
        batch_size=kwargs.get("batch_size", 1),
        epochs=kwargs.get("epochs", 1),
        learning_rate=kwargs.get("learning_rate", 2e-4),
        lora=kwargs.get("lora", {"r": 8, "alpha": 16}),
        quantization=kwargs.get("quantization", "none"),
        val_data_path=kwargs.get("val_data_path", None),
    )

    # Require GPU for this run
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available, but GPU-only execution requested. Please attach a GPU runtime.")

    ensure_dir(cfg.output_dir)

    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    train_ds, val_ds = load_datasets(cfg, tokenizer)

    # Build TrainingArguments with graceful fallback for older transformers versions that
    # may not support some keyword arguments (e.g. evaluation_strategy).
    _args = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        logging_steps=10,
        # GPU-only: ensure we are using CUDA
        fp16=True if torch.cuda.is_available() else False,
        # Add max_steps for smoke test if provided
        max_steps=kwargs.get("max_steps", -1), # -1 means no limit
    )

    # Conditionally add kwargs if supported by this transformers version
    import inspect

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        _args["evaluation_strategy"] = "steps"
    if "save_strategy" in sig.parameters:
        _args["save_strategy"] = "epoch"
    if "report_to" in sig.parameters:
        _args["report_to"] = "none"

    try:
        training_args = TrainingArguments(**_args)
    except TypeError as e:
        # Fallback: remove any unsupported keys and retry
        for k in list(_args.keys()):
            try:
                TrainingArguments(**{k: _args[k]})
            except TypeError:
                _args.pop(k)
        training_args = TrainingArguments(**_args)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # Redirect stdout to capture and yield logs from the Trainer
    f = io.StringIO()
    with redirect_stdout(f):
        yield "Starting training...\n"
        trainer.train()
        yield f.getvalue() # Yield initial logs

        yield "\nTraining complete. Saving adapter..."
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        yield f.getvalue() # Yield final logs
    
    yield f"\nAdapter saved to {cfg.output_dir}"
