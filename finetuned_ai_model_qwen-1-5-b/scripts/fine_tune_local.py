#!/usr/bin/env python3
"""
Local Fine-Tuning Script for Qwen2.5-Coder-1.5B on M2 Mac
Optimized for 8GB RAM with Metal GPU acceleration
"""

import os
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# Configuration
MODEL_PATH = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Update if using local model
DATASET_PATH = "./datasets/data_cleaned_huggingface_new.jsonl"
OUTPUT_DIR = "./checkpoints"
FINAL_MODEL_DIR = "./models/qwen_reactjs_merged"

# Training Hyperparameters for 1.5B + 8GB RAM
BATCH_SIZE = 1  # Critical for 8GB RAM
GRADIENT_ACCUMULATION = 4  # Effective batch = 4
MAX_LENGTH = 512  # React components fit comfortably
LEARNING_RATE = 5e-4  # Higher LR for smaller model
EPOCHS = 2  # 2 epochs = ~5-6 hours total
WARMUP_STEPS = 200
SAVE_STEPS = 500
EVAL_STEPS = 500

# LoRA Configuration for 1.5B
LORA_R = 16  # Smaller rank for 1.5B
LORA_ALPHA = 8
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_device():
    """Setup and verify device configuration"""
    logger.info("🔧 Setting up device...")
    
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("✅ Metal GPU (MPS) available")
    else:
        device = "cpu"
        logger.warning("⚠️  Metal GPU not available, falling back to CPU (slower)")
    
    logger.info(f"   Device: {device}")
    logger.info(f"   PyTorch version: {torch.__version__}")
    
    return device


def load_dataset_from_jsonl(path):
    """Load JSONL dataset for fine-tuning"""
    logger.info(f"📂 Loading dataset from {path}...")
    
    with open(path) as f:
        raw_data = [json.loads(line) for line in f if line.strip()]
    
    logger.info(f"   ✅ Loaded {len(raw_data)} samples")
    
    # Show sample
    sample = raw_data[0]
    logger.info(f"   Sample format: {list(sample.keys())}")
    logger.info(f"   Prompt length: {len(sample['prompt'])} chars")
    logger.info(f"   Code length: {len(sample['code'])} chars")
    
    return raw_data


def prepare_training_examples(raw_data):
    """Prepare examples in format for fine-tuning"""
    logger.info("📝 Preparing training examples...")
    
    examples = []
    for item in raw_data:
        prompt = item['prompt']
        code = item['code']
        # Format: "Prompt: {prompt}\n\nCode:\n{code}"
        text = f"Prompt: {prompt}\n\nCode:\n{code}"
        examples.append({"text": text})
    
    logger.info(f"   ✅ Prepared {len(examples)} examples")
    return examples


def load_model_and_tokenizer(device):
    """Load base model and tokenizer"""
    logger.info(f"🤖 Loading model: {MODEL_PATH}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("   ✅ Tokenizer loaded")
    
    # Load model with 8-bit quantization to save memory
    try:
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
        )
        logger.info("   ✅ Model loaded with 8-bit quantization")
    except Exception as e:
        logger.warning(f"   ⚠️  8-bit quantization failed ({e}), loading without quantization")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        logger.info("   ✅ Model loaded (fp16)")
    
    # Show model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Model parameters: {total_params/1e9:.2f}B")
    
    return model, tokenizer


def setup_lora(model):
    """Setup LoRA adapters for efficient fine-tuning"""
    logger.info("🎯 Setting up LoRA adapters...")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("   ✅ LoRA adapters applied")
    
    return model


def tokenize_function(examples, tokenizer, max_length=MAX_LENGTH):
    """Tokenization function for dataset"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def prepare_datasets(examples, tokenizer):
    """Prepare and split datasets"""
    logger.info("📊 Preparing datasets...")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict({"text": [e["text"] for e in examples]})
    logger.info(f"   Created dataset with {len(dataset)} samples")
    
    # Tokenize
    logger.info("   Tokenizing... (1-2 minutes)")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=["text"],
    )
    
    # Remove attention_mask (not needed for CLM)
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
    logger.info("🚀 Starting fine-tuning...")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Gradient accumulation: {GRADIENT_ACCUMULATION}")
    logger.info(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    logger.info(f"   Learning rate: {LEARNING_RATE}")
    logger.info(f"   Epochs: {EPOCHS}")
    logger.info(f"   Expected time: 4-6 hours (M2 GPU)\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        learning_rate=LEARNING_RATE,
        fp16=True,  # Metal GPU friendly
        gradient_checkpointing=True,  # Save memory
        logging_dir="./logs",
        logging_first_step=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    train_result = trainer.train()
    
    logger.info("\n✅ Training complete!")
    logger.info(f"   Final training loss: {train_result.training_loss:.4f}")
    
    return trainer


def save_model(trainer, model, tokenizer):
    """Save fine-tuned model"""
    logger.info("💾 Saving model...")
    
    # Create output directory
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    
    # Merge LoRA adapters
    logger.info("   Merging LoRA adapters...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    logger.info(f"   ✅ Model saved to {FINAL_MODEL_DIR}")
    logger.info(f"   Model size: ~1.2 GB (fp16)")
    
    return merged_model


def main():
    """Main fine-tuning pipeline"""
    logger.info("=" * 60)
    logger.info("Qwen2.5-Coder-1.5B Fine-Tuning - M2 Mac")
    logger.info("=" * 60 + "\n")
    
    # Setup
    device = setup_device()
    
    # Load dataset
    raw_data = load_dataset_from_jsonl(DATASET_PATH)
    examples = prepare_training_examples(raw_data)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(device)
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(examples, tokenizer)
    
    # Train
    trainer = train_model(model, tokenizer, train_dataset, val_dataset)
    
    # Save
    merged_model = save_model(trainer, model, tokenizer)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Fine-tuning Pipeline Complete!")
    logger.info("=" * 60)
    logger.info("\n📝 Next steps:")
    logger.info("   1. Run: python scripts/convert_to_gguf.py")
    logger.info("   2. Deploy with Ollama: ollama create micro-ai-coder-1-5b:latest")
    logger.info("   3. Test: python scripts/inference_finetuned.py")


if __name__ == "__main__":
    main()
