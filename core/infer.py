"""Run inference with Gemma 3n on GPU (strictly)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")


def load_model_for_inference(model_path: str, quantization: str = "none") -> tuple[Any, Any]:
    """Load tokenizer and (optionally LoRA) model strictly on GPU for inference."""
    # Ensure GPU is available
    #if not torch.cuda.is_available():
    #    raise RuntimeError("GPU not available, but GPU-only inference requested.")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    #device = "cuda"
    quant_args: Dict[str, Any] = {}
    if quantization == "int8":
        quant_args["load_in_8bit"] = True
    elif quantization == "4bit":
        quant_args["load_in_4bit"] = True

    # Detect if provided path is an adapter directory
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.is_file():
        logger.info("Loading LoRA adapter from %s", model_path)
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_cfg.base_model_name_or_path
        logger.info("Base model: %s", base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **quant_args)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        logger.info("Loading base model %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **quant_args)

    model.to(device)
    model.eval()
    logger.info("Model loaded successfully on GPU.")
    return model, tokenizer


def run_inference_from_app(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
):
    """Generates and streams response for the Gradio chat interface."""

    #device = "cuda"
    model.to(device)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Run generation in a separate thread for non-blocking streaming
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Yield generated tokens
    for new_text in streamer:
        yield new_text
