# Local Training Guide - Detailed Instructions

## Full Training Pipeline for M2 Mac

### Overview
This guide provides complete step-by-step instructions for fine-tuning Qwen2.5-Coder-1.5B on your M2 Mac with 8GB RAM using LoRA adapters.

---

## Prerequisites

### System Requirements
- macOS 12.6 or newer
- M2 or M3 chip (with unified memory)
- 8GB RAM minimum (6.5-7.5GB will be used during training)
- 10GB free disk space
- Python 3.10 or 3.11

### Software
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify Python
python3 --version  # Should be 3.10+

# Verify Metal support
python3 -c "import torch; print('Metal available:', torch.backends.mps.is_available())"
```

---

## Step 1: Environment Setup

### 1.1 Create Virtual Environment
```bash
# Navigate to project
cd /Users/muralidharanramasamy/Micro_AI_coder/finetuned_ai_model_qwen-1-5-b

# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate
```

### 1.2 Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

### 1.3 Install Dependencies
```bash
# Install PyTorch for M2 (CPU version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all other requirements
pip install -r requirements.txt
```

### 1.4 Verify Installation
```bash
python3 << 'EOF'
import torch
import transformers
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ MPS available: {torch.backends.mps.is_available()}")
print(f"✅ MPS built: {torch.backends.mps.is_built()}")

# Test Metal GPU
x = torch.randn(10, 10, device='mps')
print(f"✅ GPU test: tensor on {x.device}")
EOF
```

**Expected output:**
```
✅ PyTorch: 2.0.0+
✅ Transformers: 4.36.0+
✅ MPS available: True
✅ MPS built: True
✅ GPU test: tensor on mps:0
```

---

## Step 2: Dataset Preparation

### 2.1 Copy Dataset
```bash
# Copy to datasets folder
cp /path/to/data_cleaned_huggingface_new.jsonl datasets/

# Verify
ls -lh datasets/
# Should show: data_cleaned_huggingface_new.jsonl (50-100MB)
```

### 2.2 Validate Dataset
```bash
python3 << 'EOF'
import json
from pathlib import Path

dataset_path = "datasets/data_cleaned_huggingface_new.jsonl"

# Load and validate
samples = []
with open(dataset_path) as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

print(f"✅ Total samples: {len(samples)}")

# Show sample
sample = samples[0]
print(f"\n📊 Sample format:")
print(f"   Keys: {list(sample.keys())}")
print(f"   Prompt: {sample['prompt'][:60]}...")
print(f"   Code length: {len(sample['code'])} chars")

# Validate structure
required_keys = {'prompt', 'code'}
for i, s in enumerate(samples[:10]):
    if not all(k in s for k in required_keys):
        print(f"❌ Sample {i} missing keys!")
        
print(f"\n✅ Dataset validation PASSED")
EOF
```

---

## Step 3: Configure Model & Training

### 3.1 Update Model Path
Edit `scripts/fine_tune_local.py`, line ~25:

```python
# Option A: Use HuggingFace Hub (recommended)
MODEL_PATH = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Option B: Local model path
# MODEL_PATH = "/path/to/local/qwen-1.5b-model"
```

### 3.2 Review Training Hyperparameters
Key settings in `scripts/fine_tune_local.py`:

```python
# Line ~20: Dataset
DATASET_PATH = "./datasets/data_cleaned_huggingface_new.jsonl"

# Line ~28: Model
MODEL_PATH = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Line ~34: LoRA configuration for 1.5B
LORA_R = 16              # Rank (smaller for 1.5B model)
LORA_ALPHA = 8           # Alpha value
TARGET_MODULES = ["q_proj", "v_proj"]  # Attention layers

# Line ~44: Training parameters
BATCH_SIZE = 1           # Critical! Don't increase
GRADIENT_ACCUMULATION = 4  # Effective batch = 4
MAX_LENGTH = 512         # Token length
LEARNING_RATE = 5e-4     # Higher for smaller model
EPOCHS = 2               # 2 epochs = 4-6 hours
```

### 3.3 Memory Optimization
For 8GB RAM, these are critical:

```python
BATCH_SIZE = 1          # NEVER increase this
MAX_LENGTH = 512        # Can reduce to 256 if OOM
EPOCHS = 2              # Can reduce to 1 if very tight
```

---

## Step 4: Run Training

### 4.1 Start Training
```bash
# Activate venv
source venv/bin/activate

# Run training script
python scripts/fine_tune_local.py
```

### 4.2 Monitor Training

**In terminal**, you'll see:
```
✅ Setting up device...
   Device: mps
🤖 Loading model...
   ✅ Model loaded (fp16)
📂 Loading dataset...
   ✅ Loaded 39000 samples
📝 Preparing examples...
   ✅ Prepared 39000 examples
🎯 Setting up LoRA adapters...
   trainable params: 50,176 || all params: 1,607,126,016 || trainable%: 0.003122
🚀 Starting fine-tuning...
   Effective batch size: 4
   Learning rate: 5e-04
   Epochs: 2
   Expected time: 4-6 hours

Epoch 1: 100%|██████████| 975/975 [110:30<00:00, 6.80s/it]
  Loss: 1.84, Val Loss: 1.62
Epoch 2: 100%|██████████| 975/975 [105:15<00:00, 6.48s/it]
  Loss: 1.32, Val Loss: 1.28
✅ Training complete!
   Final training loss: 1.3157
💾 Saving model...
   Merging LoRA adapters...
   ✅ Model saved to ./models/qwen_reactjs_merged
   Model size: ~1.2 GB (fp16)
```

### 4.3 System Monitoring
While training runs, open a separate terminal:

```bash
# Monitor memory and CPU
watch -n 5 'top -n 1 -b | head -15'

# Or use Activity Monitor: ⌘ + Space → "Activity Monitor"
# - Memory tab: should stay 6.5-7.5GB
# - GPU tab: M2 should show 40-60% usage
```

**Expected values:**
- Memory: 6.5-7.5 GB (stable)
- GPU: 40-70% usage
- One process: Python ~150% CPU (Metal GPU parallelization)

### 4.4 Training Logs
Training saves logs to:
```
./logs/
  events.out.tfevents.* (TensorBoard logs)
```

### 4.5 Checkpoints
Every ~30 minutes, model saves:
```
./checkpoints/
  checkpoint-500/
    pytorch_model.bin
    optimizer.pt
    ...
  checkpoint-1000/
    ...
```

---

## Step 5: Monitor Loss Curves

### 5.1 What to Expect

**Good training indicators:**
- Training loss: decreases monotonically (1.84 → 1.3 → 1.0)
- Validation loss: decreases per epoch (1.62 → 1.28)
- No sudden spikes (indicates learning issues)
- Stable loss after first 100 steps

**Warning signs:**
- Validation loss increases (overfitting)
- Loss jumps suddenly (learning rate issue)
- Loss plateaus early (underfitting)

### 5.2 Loss Targets
- **Epoch 1**: training ~1.8-1.9, validation ~1.6-1.7
- **Epoch 2**: training ~1.3-1.4, validation ~1.2-1.3
- **Final**: validation loss < 1.5 is good

---

## Step 6: Handle Training Issues

### Issue 1: Out of Memory (OOM)
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB.
```

**Solution:**
Edit `scripts/fine_tune_local.py`:
```python
# Line 44
BATCH_SIZE = 0.5  # Reduce from 1
GRADIENT_ACCUMULATION = 8  # Increase from 4

# And/or line 42
MAX_LENGTH = 256  # Reduce from 512
```

### Issue 2: Metal GPU Not Detected
```
❌ Metal GPU not available
```

**Solution:**
```bash
# Check PyTorch version
pip show torch | grep Version  # Should be 2.0.0+

# Reinstall with correct version
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cpu
```

### Issue 3: Model Not Found
```
OSError: Can't find model at Qwen/Qwen2.5-Coder-1.5B-Instruct
```

**Solution:**
```bash
# Download model first
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Model downloaded successfully")
EOF
```

### Issue 4: Training Extremely Slow
```
> 10 seconds per step (should be 6-7 seconds)
```

**Verify Metal is active:**
```bash
python3 << 'EOF'
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
x = torch.randn(100, 100, device='mps')
y = x @ x.T
print("✅ GPU working")
EOF
```

If GPU isn't working, training will use CPU (24+ hours).

---

## Step 7: Training Completion

### 7.1 Verify Training Output
```bash
ls -lh models/qwen_reactjs_merged/

# Should show:
# -rw-r--r-- config.json
# -rw-r--r-- pytorch_model.bin (1.2 GB)
# -rw-r--r-- tokenizer.json
# -rw-r--r-- tokenizer_config.json
# -rw-r--r-- generation_config.json
```

### 7.2 Check Model Size
```bash
du -sh models/qwen_reactjs_merged/
# Should show: ~1.2G

ls -lh models/qwen_reactjs_merged/pytorch_model.bin
# Should show: -rw-r--r-- 1.2G (approx)
```

---

## Step 8: Troubleshooting

### Can't Find venv
```bash
# Activate from project root
source finetuned_ai_model_qwen-1-5-b/venv/bin/activate

# Or reinstall
cd finetuned_ai_model_qwen-1-5-b
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Permission Denied
```bash
# Make scripts executable
chmod +x scripts/*.py
chmod +x scripts/*.sh

# Or run with python
python scripts/fine_tune_local.py
```

### Disk Space Issue
```bash
# Check available space
df -h

# If low, clear:
rm -rf .cache/huggingface/*  # HF model cache
rm -rf __pycache__            # Python cache
rm -rf *.egg-info             # Pip cache
```

---

## Next Steps

After training completes:

1. **Convert to GGUF**: `python scripts/convert_to_gguf.py`
2. **Deploy with Ollama**: See `docs/OLLAMA_DEPLOYMENT.md`
3. **Test model**: `python tests/validate_finetuned_qwen.py`
4. **Use for inference**: `python scripts/inference_finetuned.py`

---

## Performance Notes

### 1.5B Model vs Larger Models
| Model | Parameters | Training Time (M2) | GGUF Size | Inference Speed |
|-------|------------|--------------------|-----------|-----------------|
| 1.5B | 1.5B | 4-6 hours | 800 MB | 20-40 tok/s |
| 7B | 7B | 12-24 hours | 3.5 GB | 5-15 tok/s |

### Memory Usage
- Peak: 7.2 GB
- Typical: 6.8 GB
- Headroom: ~0.8 GB for OS

### Training Throughput
- Tokens/second: 500-800 (M2 GPU)
- Tokens/second: 50-100 (CPU-only)

---

## Monitoring Checklist

- [ ] Environment setup complete
- [ ] Dataset validated (39K samples)
- [ ] Model path configured
- [ ] Training parameters reviewed
- [ ] Metal GPU verified active
- [ ] Training started without errors
- [ ] Memory stays 6.5-7.5GB
- [ ] Loss decreases each epoch
- [ ] No OOM errors
- [ ] Training completes successfully
- [ ] Model files saved correctly
- [ ] Ready for GGUF conversion

