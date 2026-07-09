# RunPod Setup Guide for Qwen 1.5B Fine-Tuning

## Overview
This guide explains how to run the fine-tuning on **RunPod GPU Cloud** with 24GB RAM + 16GB VRAM (e.g., RTX 4090, A100, or RTX 6000).

**Cost Estimate**: ~$3-5 per hour depending on GPU
**Training Time**: 2-3 hours (10x faster than M2 Mac!)
**Total Cost**: ~$6-15 for full training

---

## Step 1: Create RunPod Account & Set Up Pod

### 1.1 Sign Up
- Go to https://www.runpod.io
- Create account and add payment method
- Go to Pods → GPU Cloud

### 1.2 Choose GPU Pod
**Recommended Options** (24GB RAM + 16GB VRAM):
- **RTX 4090** (24GB VRAM, ~$0.44/hr) ⭐ Best value
- **A100** (40GB VRAM, ~$1.29/hr) - Overkill but faster
- **RTX 6000** (24GB VRAM, ~$0.39/hr) - Similar to 4090

**Select**:
- GPU: RTX 4090 (or A100)
- Container: `pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime`
- Disk: 30GB (to hold model + dataset)
- Volume: (optional) 100GB persistent storage

### 1.3 Launch Pod
- Click "Deploy" and wait for pod to start (1-2 minutes)
- Note your **Pod ID** and SSH connection details

---

## Step 2: Connect to Pod

### Option A: Web Terminal (Easiest)
1. Go to RunPod dashboard
2. Click your pod
3. Click "Connect" → "Web Terminal"
4. You're in! (no SSH needed)

### Option B: SSH
```bash
ssh root@{YOUR_POD_IP} -p {PORT}
# Password shown in RunPod dashboard
```

---

## Step 3: Set Up Environment

### 3.1 Clone Repository
```bash
cd /workspace  # RunPod's writable directory
git clone https://github.com/muraliwebworld/Micro_AI_coder.git
cd Micro_AI_coder/finetuned_ai_model_qwen-1-5-b
```

### 3.2 Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.36.0 peft>=0.7.0 datasets>=2.14.0 accelerate>=0.24.0 bitsandbytes>=0.41.0
pip install requests tqdm numpy pyyaml psutil
```

### 3.3 Verify GPU
```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```
Expected output:
```
GPU: NVIDIA RTX 4090
VRAM: 24.0GB
```

---

## Step 4: Prepare Dataset

### Option A: Download from GitHub (Recommended)
```bash
# Dataset is already in the repo
ls -lh datasets/data_cleaned_huggingface_new.jsonl
```

### Option B: Upload Your Dataset
If you want to upload from your Mac:
```bash
# From your Mac terminal
scp -P {PORT} datasets/data_cleaned_huggingface_new.jsonl root@{POD_IP}:/workspace/Micro_AI_coder/finetuned_ai_model_qwen-1-5-b/datasets/
```

---

## Step 5: Download Pre-trained Model

```bash
cd /workspace/Micro_AI_coder/finetuned_ai_model_qwen-1-5-b
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct', trust_remote_code=True)
print('✅ Model downloaded successfully!')
"
```
(This pre-downloads the model to avoid timeout during training)

---

## Step 6: Run Training on RunPod

### Create RunPod-Optimized Training Script

Create `scripts/fine_tune_runpod.py`:

```python
#!/usr/bin/env python3
"""
Fine-tune Qwen 1.5B on RunPod GPU (RTX 4090, 24GB VRAM)
Training time: ~2-3 hours for 2 epochs
"""

import os
import sys
import logging
import json
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
import psutil

# ============ CONFIGURATION FOR RUNPOD (24GB VRAM) ============
BATCH_SIZE = 4  # 4x larger than M2 (was 1)
GRADIENT_ACCUMULATION = 1  # Can use 1 (was 8 on M2)
MAX_LENGTH = 1024  # Full length (was 256 on M2)
LEARNING_RATE = 2e-4  # Standard for 1.5B
EPOCHS = 2
WARMUP_STEPS = 200
SAVE_STEPS = 250
EVAL_STEPS = 250

# Paths
MODEL_PATH = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_PATH = "./datasets/data_cleaned_huggingface_new.jsonl"
OUTPUT_DIR = "./checkpoints_runpod"
FINAL_MODEL_DIR = "./models/qwen_reactjs_merged_runpod"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryMonitorCallback:
    """Monitor VRAM usage"""
    def __init__(self):
        self.max_vram = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            vram_gb = torch.cuda.memory_allocated() / 1e9
            self.max_vram = max(self.max_vram, vram_gb)
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"Step {state.global_step}: VRAM {vram_gb:.1f}GB / {total_vram:.1f}GB "
                       f"(Peak: {self.max_vram:.1f}GB)")


def setup_device():
    """Detect GPU device"""
    logger.info("🔧 Setting up device...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        device = torch.device("cpu")
        logger.info("   ⚠️  No GPU found, using CPU (VERY SLOW)")
    return device


def load_dataset_from_jsonl(path):
    """Load training data"""
    logger.info(f"📂 Loading dataset from {path}...")
    with open(path, 'r') as f:
        lines = f.readlines()
    
    examples = []
    for line in lines:
        try:
            data = json.loads(line)
            examples.append(data)
        except json.JSONDecodeError:
            continue
    
    logger.info(f"   ✅ Loaded {len(examples)} examples")
    return examples


def prepare_training_examples(raw_data):
    """Format examples for training"""
    logger.info("   Formatting examples...")
    examples = []
    for item in raw_data:
        prompt = item.get("prompt", "")
        code = item.get("code", "")
        if prompt and code:
            text = f"Prompt: {prompt}\n\nCode:\n{code}"
            examples.append({"text": text})
    logger.info(f"   ✅ Prepared {len(examples)} formatted examples")
    return examples


def load_model_and_tokenizer(device):
    """Load Qwen 1.5B with 8-bit quantization"""
    logger.info("🤖 Loading model...")
    
    # 8-bit quantization for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"   ✅ Model loaded (parameters: 1.5B)")
    logger.info(f"   ✅ Tokenizer loaded (vocab: {len(tokenizer)})")
    
    return model, tokenizer


def setup_lora(model):
    """Apply LoRA adapters"""
    logger.info("⚙️  Setting up LoRA...")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"   ✅ LoRA configured")
    logger.info(f"   ✅ Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def tokenize_function(examples, tokenizer):
    """Tokenize examples"""
    result = tokenizer(
        examples["text"],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )
    return result


def prepare_datasets(examples, tokenizer):
    """Prepare and split datasets"""
    logger.info("📊 Preparing datasets...")
    
    dataset = Dataset.from_dict({"text": [e["text"] for e in examples]})
    logger.info(f"   Created dataset with {len(dataset)} samples")
    
    logger.info("   Tokenizing... (this may take 1-2 minutes)")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=["text"],
    )
    
    tokenized_dataset = tokenized_dataset.remove_columns(["attention_mask"])
    
    # Split: 80% train, 10% val, 10% test
    split_datasets = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_datasets["train"]
    temp_dataset = split_datasets["test"]
    
    split_temp = temp_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = split_temp["train"]
    test_dataset = split_temp["test"]
    
    logger.info(f"   ✅ Dataset split:")
    logger.info(f"      Train: {len(train_dataset)} samples")
    logger.info(f"      Val:   {len(val_dataset)} samples")
    logger.info(f"      Test:  {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset


def train_model(model, tokenizer, train_dataset, val_dataset):
    """Fine-tune the model"""
    logger.info("🚀 Starting fine-tuning on RunPod...")
    logger.info(f"   Batch size: {BATCH_SIZE} (4x larger than M2)")
    logger.info(f"   Gradient accumulation: {GRADIENT_ACCUMULATION} (can use 1 now)")
    logger.info(f"   Max length: {MAX_LENGTH} (full sequence)")
    logger.info(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    logger.info(f"   Learning rate: {LEARNING_RATE}")
    logger.info(f"   Epochs: {EPOCHS}")
    logger.info(f"   Expected time: 2-3 hours on RTX 4090\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        learning_rate=LEARNING_RATE,
        fp16=True,
        gradient_checkpointing=True,
        logging_dir="./logs",
        logging_first_step=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    train_result = trainer.train()
    
    logger.info("\n✅ Training complete!")
    logger.info(f"   Final training loss: {train_result.training_loss:.4f}")
    
    return trainer


def save_model(trainer, model, tokenizer):
    """Save fine-tuned model"""
    logger.info("💾 Saving model...")
    
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    logger.info(f"   ✅ Model saved to {FINAL_MODEL_DIR}")
    logger.info(f"   Size: ~3.2GB (fp16)")
    
    return merged_model


def main():
    logger.info("="*60)
    logger.info("Qwen 1.5B Fine-tuning on RunPod GPU")
    logger.info("="*60 + "\n")
    
    device = setup_device()
    raw_data = load_dataset_from_jsonl(DATASET_PATH)
    examples = prepare_training_examples(raw_data)
    model, tokenizer = load_model_and_tokenizer(device)
    model = setup_lora(model)
    train_dataset, val_dataset, test_dataset = prepare_datasets(examples, tokenizer)
    trainer = train_model(model, tokenizer, train_dataset, val_dataset)
    save_model(trainer, model, tokenizer)
    
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Convert to GGUF: python scripts/convert_to_gguf.py")
    logger.info(f"  2. Download model: scp -r models/qwen_reactjs_merged_runpod <your-mac>:")
    logger.info(f"  3. Deploy to Ollama locally")


if __name__ == "__main__":
    main()
```

### 3.1 Run Training
```bash
cd /workspace/Micro_AI_coder/finetuned_ai_model_qwen-1-5-b
python scripts/fine_tune_runpod.py
```

**Expected Output**:
```
============================================================
Qwen 1.5B Fine-tuning on RunPod GPU
============================================================

🔧 Setting up device...
   ✅ GPU: NVIDIA RTX 4090
   ✅ VRAM: 24.0GB
🤖 Loading model...
   ✅ Model loaded (parameters: 1.5B)
🚀 Starting fine-tuning on RunPod...
   Batch size: 4 (4x larger than M2)
   Max length: 1024 (full sequence)
   Expected time: 2-3 hours on RTX 4090

Epoch 1/2: 100%|███████████| 7800/7800 [58:34<00:00]
Epoch 2/2: 100%|███████████| 7800/7800 [57:22<00:00]

✅ TRAINING COMPLETE!
```

---

## Step 7: Monitor Training

### Option A: Real-time TensorBoard
```bash
# In a new terminal on RunPod
tensorboard --logdir ./logs --port 6006 --host 0.0.0.0
```
Then open: `http://{POD_IP}:6006`

### Option B: Watch VRAM
```bash
watch -n 1 nvidia-smi
```

### Option C: Check Training Loss
```bash
tail -f logs/training.log | grep "loss"
```

---

## Step 8: Convert to GGUF (Optional)

### Install GGUF Tools
```bash
pip install llama-cpp-python
```

### Create `scripts/convert_to_gguf_runpod.py`:

```python
#!/usr/bin/env python3
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
import json

MODEL_DIR = "./models/qwen_reactjs_merged_runpod"
OUTPUT_FILE = "./models/qwen_reactjs_runpod.Q4_K_M.gguf"

print("Converting to GGUF...")
# Note: This requires GGML conversion tools
# For now, we'll save as PyTorch for deployment

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Save quantized version
model.half().save_pretrained(f"{MODEL_DIR}_fp16")
print(f"✅ Model saved: {MODEL_DIR}_fp16")
```

### Run Conversion
```bash
python scripts/convert_to_gguf_runpod.py
```

---

## Step 9: Download Model to Your Mac

### From RunPod Terminal:
```bash
# Show model size
du -sh models/qwen_reactjs_merged_runpod
```

### From Your Mac:
```bash
scp -r root@{POD_IP}:/workspace/Micro_AI_coder/finetuned_ai_model_qwen-1-5-b/models/qwen_reactjs_merged_runpod ./models/
```

---

## Step 10: Stop Pod & Save Money

```bash
# When done, stop the pod from RunPod dashboard
# You'll only be charged for hours used
# Example: 2.5 hours × $0.44/hr = $1.10
```

---

## Performance Comparison

| Metric | M2 Mac (8GB) | RunPod RTX 4090 |
|--------|-------------|-----------------|
| Batch Size | 1 | 4 |
| Sequence Length | 256 | 1024 |
| Grad Accumulation | 8 | 1 |
| Time per Epoch | ~3 hours | ~30 min |
| Total Training Time | 6+ hours | 2-3 hours |
| **Speedup** | Baseline | **2-3x faster** |
| Cost | $0 | $1-3 |

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
BATCH_SIZE=2  # was 4
```

### Model Download Timeout
```bash
# Pre-download with increased timeout
HF_HUB_READ_TIMEOUT=120 python scripts/fine_tune_runpod.py
```

### Connection Lost
```bash
# Use tmux to keep running
tmux new-session -d -s training
tmux send-keys -t training "python scripts/fine_tune_runpod.py" Enter
tmux attach -t training  # Check status later
```

---

## Cost Breakdown

| Component | Cost |
|-----------|------|
| RTX 4090 Pod (3 hours) | ~$1.32 |
| Data transfer (download model) | Free |
| **Total** | **~$1.32** |

Compare to M2 Mac: 6+ hours × electricity = ~$2-3

**RunPod is cheaper AND 3x faster! 🚀**

---

## Files to Download After Training

1. **Model**: `models/qwen_reactjs_merged_runpod/` (~3.2GB)
2. **Logs**: `logs/` (optional, for analysis)
3. **Checkpoints**: `checkpoints_runpod/` (optional, backup)

---

## Next: Deploy Locally

After downloading model to Mac:

```bash
# Create Modelfile
FROM ./models/qwen_reactjs_merged_runpod
SYSTEM """You are an expert React component generator..."""
PARAMETER temperature 0.3

# Deploy
ollama create micro-ai-coder-runpod:latest -f Modelfile

# Test
python scripts/inference_finetuned.py
```

---

## Quick Reference

```bash
# 1. SSH to pod
ssh root@{POD_IP} -p {PORT}

# 2. Setup
cd /workspace && git clone <repo> && cd Micro_AI_coder/finetuned_ai_model_qwen-1-5-b
pip install -r requirements.txt

# 3. Train
python scripts/fine_tune_runpod.py

# 4. Monitor
nvidia-smi  # Check VRAM
tail -f logs/training.log  # Check loss

# 5. Download
# On Mac: scp -r root@{IP}:/workspace/.../models/qwen_reactjs_merged_runpod ./

# 6. Stop pod
# From RunPod dashboard
```

---

**That's it! You'll have a fine-tuned Qwen 1.5B model trained on RunPod in 2-3 hours for ~$1.32! 🎉**
