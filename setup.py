#!/usr/bin/env python3
"""
Setup configuration for Micro AI Coder
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="micro-ai-coder",
    version="0.1.0",
    author="Micro AI Coder Team",
    author_email="your-email@example.com",
    description="A 4-phase system for training and deploying specialized AI code generators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/Micro_AI_coder",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "micro-ai-coder-dataset=phase1_dataset_creator.v2_dataset_creator:main",
            "micro-ai-coder-train=phase2_training.v2_train_model:train",
            "micro-ai-coder-infer=phase3_inference.v2_inference:main",
            "micro-ai-coder-agent=phase4_agent.micro_ai_coder_agent:main",
        ],
    },
)
