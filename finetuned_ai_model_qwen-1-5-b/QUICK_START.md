# Qwen2.5-Coder-1.5B Local Fine-Tuning on M2 Mac - Quick Start

## ⚡ 5-Minute Setup

### Prerequisites
- Qwen2.5-Coder-1.5B model already on your machine
- Python 3.10+
- 8GB RAM (M2 Mac)
- ~10GB free disk space

### Step 1: Install Dependencies
```bash
cd finetuned_ai_model_qwen-1-5-b
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
1. Copy `data_cleaned_huggingface_new.jsonl` to `datasets/` folder
2. File should contain 39K React component examples

### Step 3: Update Configuration
Edit `scripts/fine_tune_local.py`, line ~25:
```python
MODEL_PATH = "/path/to/your/qwen-1.5b-model"  # Update this path
DATASET_PATH = "./datasets/data_cleaned_huggingface_new.jsonl"
```

### Step 4: Start Training
```bash
python scripts/fine_tune_local.py
```

**Expected:**
- Training time: 4-6 hours (M2 GPU-accelerated)
- Initial loss: ~2.0
- Final validation loss: <1.5
- Memory usage: 6-7GB RAM

### Step 5: Convert to GGUF
```bash
python scripts/convert_to_gguf.py
```

**Output:** `models/qwen_reactjs_merged.Q4_K_M.gguf` (~800MB)

### Step 6: Deploy with Ollama
```bash
ollama create micro-ai-coder-1-5b:latest -f models/Modelfile
ollama run micro-ai-coder-1-5b:latest "Write a React component"
```

### Step 7: Test Integration
```bash
python scripts/inference_finetuned.py
```

---

## 📊 Performance Expectations

| Metric | Value |
|--------|-------|
| Model Size | 1.5B parameters |
| GGUF Size | ~800MB (Q4_K_M) |
| Inference Speed | 20-40 tokens/sec |
| Success Rate | ≥85% valid React code |
| RAM Required (inference) | 2-3GB |

---

## 🛠️ Troubleshooting

### Out of Memory During Training?
→ Reduce `batch_size` from 1 to 0.5 in `fine_tune_local.py`

### Training Too Slow?
→ Ensure Metal GPU is being used: check logs for "device: mps"

### Model Not Generating Good Code?
→ Increase training epochs from 2 to 3 in `fine_tune_local.py`

---

## 📚 Full Documentation
See [LOCAL_TRAINING_GUIDE.md](docs/LOCAL_TRAINING_GUIDE.md) for detailed setup.
