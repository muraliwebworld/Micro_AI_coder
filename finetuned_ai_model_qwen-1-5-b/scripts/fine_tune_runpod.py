#!/usr/bin/env python3
"""
Fine-tune Qwen 1.5B on RunPod GPU (RTX 4090, 24GB VRAM)
Training time: ~2-3 hours for 2 epochs
Batch size: 4 (4x larger than M2 Mac)
Max sequence: 1024 tokens (full, not reduced)
Cost: ~$1-3 total (~$0.44/hr on RTX 4090)

Usage:
  python scripts/fine_tune_runpod.py
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
from transformers.trainer_callback import TrainerCallback
import psutil

# ============ CONFIGURATION FOR RUNPOD (24GB VRAM + RTX 4090) ============
BATCH_SIZE = 4  # 4x larger than M2 (was 1)
GRADIENT_ACCUMULATION = 1  # Can use 1 (was 8 on M2 for memory)
MAX_LENGTH = 1024  # Full length (was 256 on M2 to save memory)
LEARNING_RATE = 2e-4  # Standard for 1.5B on GPU
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPUMemoryMonitorCallback(TrainerCallback):
    """Monitor GPU VRAM usage during training"""
    def __init__(self):
        self.max_vram_gb = 0
        self.total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            self.max_vram_gb = max(self.max_vram_gb, allocated_gb)
            percent = (allocated_gb / self.total_vram_gb) * 100
            
            logger.info(f"   Step {state.global_step}: VRAM {allocated_gb:.1f}GB / {self.total_vram_gb:.1f}GB ({percent:.1f}%) | "
                       f"Peak: {self.max_vram_gb:.1f}GB")


def setup_device():
    """Detect and configure GPU device"""
    logger.info("🔧 Setting up GPU device...")
    
    if not torch.cuda.is_available():
        logger.error("   ❌ No CUDA GPU found! RunPod requires GPU pod.")
        sys.exit(1)
    
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"   ✅ GPU: {device_name}")
    logger.info(f"   ✅ Total VRAM: {total_vram:.1f}GB")
    logger.info(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    
    return device


def load_dataset_from_jsonl(path):
    """Load training data from JSONL file"""
    logger.info(f"📂 Loading dataset from {path}...")
    
    if not os.path.exists(path):
        logger.error(f"   ❌ Dataset not found at {path}")
        sys.exit(1)
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    examples = []
    errors = 0
    for line in lines:
        try:
            data = json.loads(line.strip())
            examples.append(data)
        except json.JSONDecodeError:
            errors += 1
            continue
    
    logger.info(f"   ✅ Loaded {len(examples)} examples ({errors} errors skipped)")
    
    if len(examples) == 0:
        logger.error("   ❌ No valid examples found!")
        sys.exit(1)
    
    return examples


def prepare_training_examples(raw_data):
    """Format raw data into training text"""
    logger.info("   Formatting examples...")
    
    examples = []
    skipped = 0
    
    for item in raw_data:
        prompt = item.get("prompt", "")
        code = item.get("code", "")
        
        if prompt and code:
            text = f"Prompt: {prompt}\n\nCode:\n{code}"
            examples.append({"text": text})
        else:
            skipped += 1
    
    logger.info(f"   ✅ Prepared {len(examples)} formatted examples ({skipped} skipped)")
    
    return examples


def load_model_and_tokenizer(device):
    """Load Qwen 1.5B with 8-bit quantization"""
    logger.info("🤖 Loading model and tokenizer...")
    
    # 8-bit quantization config (reduces VRAM usage)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"   ✅ Model loaded (1.5B parameters)")
    logger.info(f"   ✅ Tokenizer loaded (vocab: {len(tokenizer)})")
    
    return model, tokenizer


def setup_lora(model):
    """Apply LoRA adapters to model"""
    logger.info("⚙️  Setting up LoRA...")
    
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=8,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Attention projections
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Calculate parameter efficiency
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / total_params
    
    logger.info(f"   ✅ LoRA applied to q_proj and v_proj")
    logger.info(f"   ✅ Trainable: {trainable_params:,} / {total_params:,} ({pct:.3f}%)")
    
    return model


def tokenize_function(examples, tokenizer):
    """Tokenize text examples"""
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
    
    # Create dataset
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
    """Fine-tune the model on GPU"""
    logger.info("🚀 Starting fine-tuning on RunPod...")
    logger.info(f"   Batch size: {BATCH_SIZE} (4x larger than M2)")
    logger.info(f"   Gradient accumulation: {GRADIENT_ACCUMULATION}")
    logger.info(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    logger.info(f"   Max sequence length: {MAX_LENGTH} tokens")
    logger.info(f"   Learning rate: {LEARNING_RATE}")
    logger.info(f"   Epochs: {EPOCHS}")
    logger.info(f"   Expected time: 2-3 hours on RTX 4090\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Training arguments optimized for GPU
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
        fp16=True,  # Mixed precision training
        gradient_checkpointing=True,  # Save memory
        logging_dir="./logs",
        logging_first_step=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=True,  # Faster data loading
        dataloader_num_workers=4,  # Parallel data loading
        optim="adamw_torch",  # Standard AdamW optimizer
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Trainer with memory monitoring
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[GPUMemoryMonitorCallback()],
    )
    
    # Start training
    logger.info("   Starting training loop...\n")
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
    logger.info(f"   ✅ Model size: ~3.2GB (fp16)")
    
    return merged_model


def main():
    """Main training pipeline"""
    logger.info("="*70)
    logger.info("     Qwen 1.5B Fine-tuning on RunPod GPU")
    logger.info("="*70 + "\n")
    
    try:
        # Setup
        device = setup_device()
        
        # Load and prepare data
        raw_data = load_dataset_from_jsonl(DATASET_PATH)
        examples = prepare_training_examples(raw_data)
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(device)
        model = setup_lora(model)
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = prepare_datasets(examples, tokenizer)
        
        # Train
        trainer = train_model(model, tokenizer, train_dataset, val_dataset)
        
        # Save
        save_model(trainer, model, tokenizer)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("                    ✅ TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info(f"  1. Download model: scp -r root@{{POD_IP}}:/workspace/.../models/qwen_reactjs_merged_runpod ./")
        logger.info(f"  2. Verify locally: du -sh models/qwen_reactjs_merged_runpod")
        logger.info(f"  3. Deploy to Ollama: ollama create micro-ai-coder-runpod:latest -f Modelfile")
        logger.info(f"  4. Test inference: python scripts/inference_finetuned.py")
        logger.info(f"\n💰 Cost: ~$1-3 for full training on RTX 4090!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
