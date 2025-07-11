# Gemma 3n GPU Training and Finetuning

A lightweight, GPU-focused framework to run inference and LoRA fine-tuning on Google’s Gemma 3n family (`1.1B`, `2B`). Designed for small-scale deployments such as chatbots, assistants, or domain-specific Q&A systems.

## Features

* **GPU-only** inference and training for high performance.
* **LoRA** fine-tuning with **PEFT** for memory-efficient adaptation.
* Optional **int8 / 4-bit quantization** with **bitsandbytes**.
* Simple **Gradio Web App** for an easy, no-code workflow.
* Functionality to push trained adapters directly to the **Hugging Face Hub**.

## Prerequisites

### 1. Hugging Face Account & Gemma Access

Before you begin, you must have a Hugging Face account and accept the license terms for the Gemma model you intend to use. You will not be able to download the model weights otherwise.

- **Accept Terms for Gemma 3 1B**: [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
- **Accept Terms for Gemma 3 2B**: [google/gemma-3-2b-it](https://huggingface.co/google/gemma-3-2b-it)

### 2. Hardware & System Drivers

This application requires a local GPU to run. The setup process differs based on your hardware:

### NVIDIA GPUs (Recommended)
- **NVIDIA Driver**: You must have the latest NVIDIA drivers installed. You can download them from the [NVIDIA website](https://www.nvidia.com/Download/index.aspx).
- **CUDA Toolkit**: While the PyTorch version in `requirements.txt` bundles many necessary CUDA libraries, a full installation of the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) (version 11.8 or 12.1 is recommended) is best for compatibility.

### Hardware Requirements (VRAM)

Fine-tuning requires a significant amount of GPU memory (VRAM). Here are some estimates:

- **Gemma 3 1B**: At least **8-12 GB** of VRAM.
- **Gemma 3 2B**: At least **12-16 GB** of VRAM.

Using 4-bit quantization (an option in the app) can lower these requirements, but performance may vary.

### Apple Silicon (M1/M2/M3 Macs)
- **macOS & Xcode**: Ensure you are on a recent version of macOS with Xcode and the command line tools installed. No separate GPU drivers are needed, as PyTorch uses the built-in Metal Performance Shaders (MPS) backend.
- **Note**: While functional, performance and stability on MPS can sometimes lag behind CUDA.

## Quick Start

This project uses a Gradio web interface to simplify the fine-tuning process. 

### 1. Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Launch the Web App

Run the `app.py` script to start the Gradio server:

```bash
python3 app.py
```

This will launch a web server and provide a local URL (usually `http://127.0.0.1:7860`). Open this URL in your browser.

### 3. Using the App

The app is organized into three tabs:

1.  **Setup & Data Processing**:
    *   Upload your dataset (CSV, Excel, PDF, or Word).
    *   Select the Gemma model you want to fine-tune.
    *   Enter your Hugging Face token to authenticate.
    *   Click **Process Data & Prepare for Training**.

2.  **Training**:
    *   Select the base model for fine-tuning.
    *   Check the **Run smoke test** box for a quick 10-step run to verify the pipeline.
    *   Click **Start Fine-Tuning** and monitor the logs.

3.  **Inference & Export**:
    *   Once training is complete, chat with your fine-tuned model in the chat interface.
    *   To share your model, enter a repository name (e.g., `your-username/my-gemma-finetune`) and click **Push to Hub**.

## Repository Layout

```
├── configs/            # YAML configuration files
├── core/               # Framework source code
│   ├── data.py         # Dataset loading helpers
│   ├── train.py        # LoRA fine-tuning entry point
│   ├── infer.py        # Inference entry point
│   └── utils.py        # Shared utilities
├── cli.py              # Typer-based CLI wrapper
├── requirements.txt    # Python dependencies
└── setup.py            # Installable package definition
```

## License

This project is released under the Apache 2.0 license.
