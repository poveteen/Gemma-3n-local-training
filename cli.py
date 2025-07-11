"""Typer CLI for Gemma 3n CPU Training and Finetuning framework."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

# Ensure project root is on sys.path when running as script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import infer as infer_module  # pylint: disable=wrong-import-position

app = typer.Typer(add_completion=False, help="Gemma 3n CPU Training & Finetuning CLI")


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    accelerate_args: str = typer.Option("", help="Additional args to pass to accelerate launch"),
):
    """Launch LoRA fine-tuning via Hugging Face Accelerate (CPU-only).

    Equivalent to:
    # Force CPU training as requested
    accelerate launch --cpu -m core.train --config <CONFIG>
    """

    cmd = [
        "accelerate",
        "launch",
        "--cpu",
        "-m",
        "core.train",
    ]
    if accelerate_args:
        cmd.extend(accelerate_args.split())
    cmd.extend(["--config", config])

    typer.echo(f"Running: {' '.join(cmd)}")

    # Create a copy of the current environment and disable GPUs to force CPU
    env = os.environ.copy()
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable MPS
    env["CUDA_VISIBLE_DEVICES"] = ""  # Hide all GPUs

    # Run the command and stream output in real-time.
    # The `env` variable ensures CPU-only execution.
    subprocess.run(cmd, check=True, env=env)


@app.command()
def infer(
    model: str = typer.Option(..., help="Path or HF hub ID of model or adapter"),
    prompt: str = typer.Option(..., help="Prompt text"),
    quantization: str = typer.Option("none", help="Quantization mode: none, int8, 4bit"),
    max_tokens: int = typer.Option(256, help="Max new tokens"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
):
    """Run inference on CPU using the specified model or adapter."""

    result = infer_module.generate_text(
        model_path=model,
        prompt=prompt,
        quantization=quantization,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    typer.echo(result)


@app.command()
def eval():  # noqa: D401
    """Placeholder for future evaluation metrics (Phase 2)."""
    typer.echo("Eval command not implemented yet.")


if __name__ == "__main__":
    app()
