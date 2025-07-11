from pathlib import Path

from setuptools import find_packages, setup

PACKAGE_NAME = "gemma3n_cpu_training_finetuning"
DESCRIPTION = "CPU-only inference and LoRA fine-tuning framework for Google's Gemma 3n models"
ROOT_DIR = Path(__file__).resolve().parent
README = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    author="Your Name",
    license="Apache-2.0",
    packages=find_packages(exclude=("tests", "examples", "data")),
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece>=0.1.99",
        "scipy>=1.12.0",
        "typer>=0.9.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "pytest",
        ]
    },
    entry_points={
        "console_scripts": [
            "gemma3n-cli=cli:app",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
