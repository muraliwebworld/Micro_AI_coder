# Local Fine-Tuning Setup Guide - M2 Mac

## System Requirements

### Hardware
- **Mac mini M2** with unified memory architecture
- **8GB RAM minimum** (tight but viable)
- **10GB free disk space** for model + datasets + checkpoints
- **M2 GPU** with Metal Performance Shaders support (automatic)

### Software
- **macOS 12.6+** (Monterey or newer)
- **Python 3.10 or 3.11** (3.12 may have compatibility issues)
- **Xcode Command Line Tools**

### Verify Setup
```bash
# Check Python
python3 --version  # Should be 3.10+

# Install Xcode tools if needed
xcode-select --install

# Verify Metal GPU support
python3 -c "import torch; print(f'Metal available: {torch.backends.mps.is_available()}')"
```

---

## Installation Steps

### 1. Create Virtual Environment
```bash
cd /Users/muralidharanramasamy/Micro_AI_coder/finetuned_ai_model_qwen-1-5-b
python3 -m venv venv
source venv/bin/activate
```

### 2. Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch for M2
```bash
# Install CPU first, then add MPS support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

### 5. Verify Installation
```bash
python3 -c "
import torch
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"
```

---

## Dataset Preparation

### 1. Copy Dataset
```bash
cp /path/to/data_cleaned_huggingface_new.jsonl datasets/
```

### 2. Verify Dataset
```bash
python3 << 'EOF'
import json

dataset_path = "datasets/data_cleaned_huggingface_new.jsonl"
with open(dataset_path) as f:
    samples = [json.loads(line) for line in f if line.strip()]

print(f"✅ Dataset: {len(samples)} samples")
sample = samples[0]
print(f"Keys: {list(sample.keys())}")
print(f"Prompt: {sample['prompt'][:60]}...")
print(f"Code length: {len(sample['code'])} chars")
EOF
```

---

## Model Path Configuration

### 1. Locate Your Qwen 1.5B Model
```bash
# If using Ollama model storage (common)
~/.ollama/models/manifests/registry.ollama.ai/library/qwen2.5-coder/

# Or if local file system
find ~ -name "*qwen*1.5*" -type f 2>/dev/null
```

### 2. Update Configuration
Edit `scripts/fine_tune_local.py`, line ~25:
```python
# Option A: Use Hugging Face hub
MODEL_PATH = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Option B: Local model directory
MODEL_PATH = "/path/to/local/qwen-1.5b"
```

---

## Memory Optimization for M2 with 8GB RAM

### Key Settings in `fine_tune_local.py`:

1. **Batch Size** (line ~60)
   ```python
   batch_size=1  # Maximum for 8GB RAM
   # Don't increase this - will cause OOM
   ```

2. **Gradient Accumulation** (line ~60)
   ```python
   gradient_accumulation_steps=4  # Effective batch = 4
   ```

3. **Max Token Length** (line ~90)
   ```python
   max_length=512  # React components fit comfortably
   ```

4. **LoRA Configuration** (line ~40)
   ```python
   r=16  # Rank for 1.5B model (smaller than 7B)
   lora_alpha=8
   ```

5. **Mixed Precision** (line ~75)
   ```python
   fp16=True  # Uses Metal GPU efficiently
   gradient_checkpointing=True  # Saves memory
   ```

---

## Expected Training Time

| Step | Duration | Notes |
|------|----------|-------|
| Model loading | 2-3 min | One-time |
| Dataset loading | 2-3 min | One-time |
| Tokenization | 5-8 min | Cached for epochs 2-3 |
| Training epoch 1 | 90-120 min | Slowest, M2 warming up |
| Training epoch 2 | 80-100 min | Slightly faster |
| Total | **4-6 hours** | Depending on load |

---

## Running Training

```bash
# Activate environment
source venv/bin/activate

# Run training
python scripts/fine_tune_local.py

# Monitor in Activity Monitor:
# - Memory should stay 6.5-7.5 GB
# - GPU should be active (check Metal activity)
```

### What to Expect
```
✅ GPU: Apple Metal (mps)
✅ Model loaded: 1.5B parameters
✅ Dataset: 31200 train, 3900 val, 3900 test
🚀 Starting training...
Epoch 1: 100%|████| 3900/3900 [110:30<00:00, 1.70s/it]
  Loss: 1.84, Val Loss: 1.62
Epoch 2: 100%|████| 3900/3900 [105:15<00:00, 1.62s/it]
  Loss: 1.32, Val Loss: 1.28
✅ Training complete!
```

---

## Troubleshooting

### Issue: Metal GPU Not Detected
```bash
# Check availability
python3 -c "import torch; print(torch.backends.mps.is_available())"

# Solution: Update to PyTorch 2.1+
pip install --upgrade torch
```

### Issue: Out of Memory During Training
```python
# In fine_tune_local.py:
# - Reduce batch_size from 1 to 0.5
# - Reduce max_length from 512 to 256
# - Reduce gradient_accumulation from 4 to 2
```

### Issue: Training Extremely Slow
```bash
# Check if GPU is being used
python3 << 'EOF'
import torch
x = torch.randn(10, 10, device='mps')
print("GPU working:", x.device)
EOF
```

### Issue: Model File Not Found
```bash
# Ensure model is downloaded first:
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
print(f"Downloading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Model downloaded")
EOF
```

---

## Next Steps

1. **Complete Training**: Wait for training to finish (~4-6 hours)
2. **Convert to GGUF**: See `docs/GGUF_CONVERSION.md`
3. **Deploy to Ollama**: See `docs/OLLAMA_DEPLOYMENT.md`
4. **Validate Model**: See `docs/VALIDATION_TESTING.md`

