# RunPod vs Local Training - Quick Comparison

## Performance & Cost Comparison

| Metric | M2 Mac (Local) | RunPod RTX 4090 |
|--------|---|---|
| **VRAM** | 8GB unified RAM | 24GB dedicated VRAM |
| **Batch Size** | 1 | 4 (4x larger!) |
| **Sequence Length** | 256 tokens | 1024 tokens (full!) |
| **Grad Accumulation** | 8 | 1 |
| **Time per Epoch** | ~3 hours | ~30 minutes |
| **Total 2 Epochs** | 6-7 hours | 2-3 hours |
| **Training Speed** | Baseline | **2-3x FASTER** |
| **Cost** | Electricity (~$2) | $1.32 (3 hrs × $0.44) |
| **Cost/Speed** | ✅ Free but slow | ✅ Super cheap & fast |

## Why RunPod is Better for This

1. **Much Faster**: 2-3 hours vs 6-7 hours on M2
2. **Better Quality**: Can use full 1024 token sequences (not reduced to 256)
3. **Cheaper**: ~$1.32 for full training (RTX 4090)
4. **No Risk**: No thermal throttling, no system slowdown
5. **Easier**: No memory optimization needed, just run the script

## Quick Start Checklist

### Step 1: Create RunPod Account
- [ ] Go to https://www.runpod.io
- [ ] Sign up and add payment method
- [ ] Go to Pods → GPU Cloud

### Step 2: Launch Pod
- [ ] Select RTX 4090 (or A100 if want faster)
- [ ] Container: pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
- [ ] Disk: 30GB
- [ ] Click Deploy

### Step 3: Connect & Setup (5 minutes)
```bash
# In RunPod Web Terminal:

cd /workspace
git clone https://github.com/muraliwebworld/Micro_AI_coder.git
cd Micro_AI_coder/finetuned_ai_model_qwen-1-5-b

# Install
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Test GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 4: Pre-download Model (2 minutes)
```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct', trust_remote_code=True)
print('✅ Model downloaded')
"
```

### Step 5: Run Training (2-3 hours)
```bash
python scripts/fine_tune_runpod.py
```

**Expected Output:**
```
🔧 Setting up GPU device...
   ✅ GPU: NVIDIA RTX 4090
   ✅ Total VRAM: 24.0GB
🤖 Loading model and tokenizer...
   ✅ Model loaded (1.5B parameters)
⚙️  Setting up LoRA...
📂 Loading dataset...
   ✅ Loaded 39000 examples
📊 Preparing datasets...
   Train: 31200 samples
   Val: 3900 samples
   Test: 3900 samples
🚀 Starting fine-tuning on RunPod...
   Batch size: 4 (4x larger than M2)
   Max sequence length: 1024 tokens
   Expected time: 2-3 hours on RTX 4090

Epoch 1/2: 100%|████| 7800/7800 [58:34<00:00]  loss=0.54
Epoch 2/2: 100%|████| 7800/7800 [57:22<00:00]  loss=0.42

✅ TRAINING COMPLETE!
```

### Step 6: Download Model (5 minutes)
```bash
# From your Mac:
scp -r root@{POD_IP}:/workspace/Micro_AI_coder/finetuned_ai_model_qwen-1-5-b/models/qwen_reactjs_merged_runpod ./models/
```

### Step 7: Stop Pod
- [ ] Go to RunPod dashboard
- [ ] Click your pod
- [ ] Click "Terminate"
- [ ] Pod stops, you stop being charged

### Step 8: Deploy Locally
```bash
# On your Mac:
python scripts/inference_finetuned.py
```

## Cost Breakdown

```
RTX 4090 Price: $0.44/hour
Training Time: 2.5 hours average
Total Cost: 2.5 × $0.44 = $1.10 ✅

Compare to:
- Google Colab (T4): Free but limited
- Local M2: Free electricity but 6+ hours = pain
- RunPod: $1.10 + done in 2.5 hours = best value 🚀
```

## File Locations

After training, files will be at:

**On RunPod:**
```
/workspace/Micro_AI_coder/finetuned_ai_model_qwen-1-5-b/
├── models/qwen_reactjs_merged_runpod/  (3.2GB - download this)
├── checkpoints_runpod/  (optional backup)
└── logs/  (training logs)
```

**On Your Mac (after download):**
```
./models/qwen_reactjs_merged_runpod/  (3.2GB)
├── config.json
├── model.safetensors
├── pytorch_model.bin
├── tokenizer.json
└── tokenizer_config.json
```

## Monitoring During Training

### Real-time VRAM Usage
```bash
# In RunPod, open another terminal:
watch -n 1 nvidia-smi
```

### Training Loss Graph
```bash
# Terminal 1: Start tensorboard
tensorboard --logdir ./logs --port 6006 --host 0.0.0.0

# Then open: http://{POD_IP}:6006 in browser
```

### Quick Status Check
```bash
# Check if training is running
nvidia-smi

# Check output logs
tail -f logs/run_*/1.log

# Check progress
ls -lh checkpoints_runpod/ | grep checkpoint
```

## Troubleshooting

### Problem: CUDA Out of Memory
```bash
# Solution: Reduce batch size in script
BATCH_SIZE = 2  # was 4
```

### Problem: Connection Timeout
```bash
# Solution: Use tmux to keep session alive
tmux new-session -d -s training
tmux send-keys -t training "python scripts/fine_tune_runpod.py" Enter

# Check later:
tmux attach -t training
```

### Problem: Model Download Timeout
```bash
# Solution: Increase timeout
HF_HUB_READ_TIMEOUT=120 python scripts/fine_tune_runpod.py
```

### Problem: Pod Disconnects
- RunPod keeps training running in background
- Reconnect to web terminal to check status
- Training continues even if you disconnect!

## Key Differences from Local Script

| Aspect | Local (M2) | RunPod |
|--------|-----------|--------|
| Batch size | 1 | 4 |
| Max length | 256 | 1024 |
| Grad accum | 8 | 1 |
| Optimizer | Adafactor | AdamW |
| Eval frequency | Less | More |
| Data workers | 0 | 4 |
| Memory target | <6.5GB | Use all 24GB |

## After Training

### Option 1: Convert to GGUF
```bash
# RunPod is also great for GGUF conversion (faster)
python scripts/convert_to_gguf_runpod.py
```

### Option 2: Quick Download & Test
```bash
# Fast enough to just download and test locally
scp -r root@{IP}:.../ ./models/qwen_reactjs_merged_runpod/
python scripts/inference_finetuned.py
```

## Recommended Workflow

1. **Use RunPod for training** (2-3 hours, $1.32)
   - Faster training
   - Better quality (full sequences)
   - Cheaper than continued M2 time
   
2. **Download to Mac** (5 min, free)
   - Inference ready
   - Can iterate quickly
   
3. **Deploy with Ollama** (on Mac, free)
   - Use locally
   - Great for testing
   
4. **Fine-tune iterations** (use RunPod again if needed)
   - Collect more data
   - Improve quality
   - Repeat

## Quick Reference Commands

```bash
# Create pod
# 1. RunPod.io → GPU Cloud → RTX 4090 → Deploy

# SSH in
ssh root@{POD_IP} -p {PORT}

# Setup (run once)
cd /workspace && git clone <repo> && cd Micro_AI_coder/finetuned_ai_model_qwen-1-5-b
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Train (main command)
python scripts/fine_tune_runpod.py

# Monitor
nvidia-smi
tail -f logs/run_*/1.log

# Download (from Mac)
scp -r root@{IP}:...models/qwen_reactjs_merged_runpod ~/models/

# Delete pod
# RunPod dashboard → Terminate
```

---

**Recommendation**: Use RunPod! 3x faster, $1 cheaper, and you can keep your M2 Mac running normally. 🚀
