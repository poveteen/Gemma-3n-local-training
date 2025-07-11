"""Utility helpers for Gemma 3n CPU Training and Finetuning framework."""
from __future__ import annotations

import logging
import os
import yaml
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")


@dataclass
class Config:
    """Dataclass wrapper around a YAML config dict."""

    model_name: str
    train_data_path: str | None = None
    val_data_path: str | None = None
    output_dir: str = "./models/gemma_finetuned"
    quantization: str = "none"  # none, int8, 4bit
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    lora: Dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
        # Ensure correct types are enforced from YAML
        data["batch_size"] = int(data.get("batch_size", 8))
        data["epochs"] = int(data.get("epochs", 3))
        data["learning_rate"] = float(data.get("learning_rate", 2e-5))
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "output_dir": self.output_dir,
            "quantization": self.quantization,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "lora": self.lora or {},
        }


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path and not os.path.isdir(path):
        logger.info("Creating directory %s", path)
        os.makedirs(path, exist_ok=True)
