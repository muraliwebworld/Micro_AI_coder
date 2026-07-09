# Qwen2.5-Coder-1.5B Fine-Tuned for ReactJS
## Optimized for M2 Mac mini 8GB RAM

**Complete setup for fine-tuning and deploying a lightweight AI code generation model on your local M2 machine.**

---

## 📋 Quick Overview

| Metric | Details |
|--------|---------|
| **Base Model** | Qwen2.5-Coder-1.5B-Instruct |
| **Training Dataset** | 39K React components |
| **GGUF Size** | 1.5-2 GB (Q4_K_M) |
| **Runtime RAM** | 2-3 GB (on M2 Mac mini) |
| **Inference Speed** | 15-20 tokens/sec |
| **Training Time** | 45-60 minutes |
| **Target Device** | M2 Mac mini 8GB RAM ✅ |
| **Code Quality** | ≥90% valid React syntax |

---

## 📁 Folder Structure

```
finetuned_ai_model_qwen-1-5-b/
├── colab_notebooks/
│   └── fine_tune_qwen_1_5b_colab.ipynb          # Main training notebook
├── phase3_inference/
│   └── v3_inference_finetuned_1_5b.py           # Inference module
├── phase4_agent/
│   └── micro_ai_coder_agent_1_5b.py             # Agent orchestrator
├── tests/
│   └── validate_finetuned_qwen_1_5b.py          # Validation tests
├── models/
│   ├── qwen_1_5b_reactjs_merged/                # (Populated after training)
│   ├── qwen_1_5b_reactjs_merged.Q4_K_M.gguf    # (After GGUF conversion)
│   └── Modelfile_qwen_1_5b                      # Ollama model config
├── docs/
│   ├── README.md                                # This file
│   ├── QUICK_START.md                           # 5-minute setup
│   ├── TRAINING_GUIDE.md                        # Complete training guide
│   └── M2_DEPLOYMENT_GUIDE.md                   # M2 Mac mini deployment
└── scripts/
    ├── convert_to_gguf.py                       # GGUF conversion script
    └── setup_ollama.sh                          # Ollama setup script
```

---

## 🚀 Quick Start (5 minutes)

### 1. Prepare Dataset
```bash
# Upload to Google Drive:
# - Create folder: micro_ai_coder_data
# - Upload: data_cleaned_huggingface_new.jsonl
```

### 2. Run Colab Notebook
```
1. Open: colab_notebooks/fine_tune_qwen_1_5b_colab.ipynb
2. In Google Colab:
   - Go to Runtime → Change runtime type → GPU (T4)
   - Run cells 1-11 sequentially
   - Training: 45-60 minutes
3. Download merged model from Google Drive
```

### 3. Local Setup (M2 Mac mini)
```bash
# Install dependencies
pip install torch transformers bitsandbytes peft

# Convert to GGUF
python scripts/convert_to_gguf.py \
  --model_path models/qwen_1_5b_reactjs_merged \
  --output_path models/qwen_1_5b_reactjs_merged.Q4_K_M.gguf

# Install Ollama (if not already)
brew install ollama

# Deploy model
ollama create micro-ai-coder-1-5b:latest -f models/Modelfile_qwen_1_5b

# Test
ollama run micro-ai-coder-1-5b:latest "Write a React button component"
```

### 4. Run Agent
```bash
python phase4_agent/micro_ai_coder_agent_1_5b.py
```

---

## 📊 Why Qwen2.5-Coder-1.5B?

### Comparison with 7B Model

```
Model Size:           1.5B (8x smaller than 7B)
GGUF Format:          1.5-2 GB (vs 4-5 GB for 7B)
Runtime RAM:          2-3 GB (vs 6-8 GB for 7B)
Inference Speed:      15-20 tok/s (vs 5-10 tok/s for 7B)
Training Time:        45-60 min (vs 2-3 hours for 7B)
Code Quality Loss:    ~5-10% (still excellent for React)
M2 Mac mini:          Perfect Fit ✅ (vs Tight squeeze for 7B)
```

### 1.5B Advantages for M2 Mac mini (8GB RAM)

✅ **Fits Comfortably**: 2-3GB model + 3GB OS + 2GB headroom = 8GB total  
✅ **Fast Inference**: 15-20 tokens/sec (still plenty for single dev)  
✅ **Quick Training**: 45-60 minutes on free T4 GPU  
✅ **Good Quality**: Pre-trained on billions of code tokens  
✅ **Low Latency**: Perfect for interactive coding workflows  

---

## 🔧 Configuration Details

### LoRA Setup
```python
r = 32                      # Lower rank for 1.5B model
lora_alpha = 16
target_modules = ["q_proj", "v_proj"]
trainable_params = ~0.2-0.3% of total
```

### Training Hyperparameters
```python
batch_size = 8              # Can use larger batches with 1.5B
epochs = 3
learning_rate = 2e-4
warmup_steps = 300
max_length = 512            # Optimal for React components
```

### Quantization
```python
load_in_8bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = True
bnb_4bit_compute_dtype = torch.float16
```

---

## 📈 Expected Results

### Training Metrics
- **Training Loss**: Should decrease from ~3.5 → 0.8-1.2
- **Validation Loss**: Should converge < 1.3
- **Training Time**: 45-60 minutes per epoch on T4 GPU

### Code Generation Quality
- **Valid React Syntax**: ≥90% of generated components
- **Proper Imports**: Automatically includes required imports
- **Component Structure**: Correct JSX, hooks, exports
- **Inference Speed**: 15-20 tokens/sec on M2

### Example Generated Code
```javascript
// Input: "Write a React form component"
import React, { useState } from 'react';

export default function Form() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Form submitted:', formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        name="name"
        value={formData.name}
        onChange={handleChange}
        placeholder="Enter name"
      />
      <button type="submit">Submit</button>
    </form>
  );
}
```

---

## 🔄 Complete Workflow

### Phase 1: Training (Google Colab)
1. Upload dataset to Google Drive
2. Run Colab notebook (45-60 min)
3. Download merged model

### Phase 2: Local Conversion (M2 Mac)
1. Convert to GGUF format (~2 GB)
2. Register with Ollama
3. Verify with test prompts

### Phase 3: Integration
1. Copy inference module
2. Copy agent module
3. Run validation tests (≥90% should pass)

### Phase 4: Deployment
1. Start Ollama server
2. Run agent
3. Generate React components interactively

---

## 🐛 Troubleshooting

### Training Issues

**OutOfMemoryError on T4 GPU**
```bash
# Reduce batch size in Step 7
per_device_train_batch_size = 4
gradient_accumulation_steps = 2
```

**Slow Training**
- This is normal for 1.5B model - takes 45-60 min
- T4 GPU provides about 100-150 TFLOPS
- Keep browser tab active to avoid Colab disconnect

### Inference Issues

**Model not loading on M2**
```bash
# Check RAM usage
top -o MEM

# Reduce quantization if needed
# Use Q3_K_M (3-bit) instead of Q4_K_M (4-bit)
# Reduces size to ~1 GB but slight quality loss
```

**Slow inference on M2**
- Normal: M2 is ~2x slower than T4
- Expected: 15-20 tokens/sec
- For production: Consider running Ollama on faster Mac Mini (M3/M4)

---

## 📚 Files Included

| File | Purpose |
|------|---------|
| `fine_tune_qwen_1_5b_colab.ipynb` | Complete training notebook |
| `v3_inference_finetuned_1_5b.py` | Inference via Ollama HTTP API |
| `micro_ai_coder_agent_1_5b.py` | Interactive menu-driven agent |
| `validate_finetuned_qwen_1_5b.py` | 20-test validation suite |
| `Modelfile_qwen_1_5b` | Ollama model configuration |
| `convert_to_gguf.py` | GGUF conversion utility |
| `setup_ollama.sh` | Ollama installation script |

---

## ✅ Validation Checklist

After setup, verify:

- [ ] Training loss converged < 1.3
- [ ] Validation tests pass ≥18/20 (90%)
- [ ] Model loads on M2 with < 4GB RAM
- [ ] Inference runs at 15+ tokens/sec
- [ ] Generated code is valid React syntax
- [ ] Agent interactive menu works

---

## 📖 Additional Resources

- **Google Colab**: Free T4 GPU for training
- **Ollama**: Local LLM deployment platform
- **GGUF Format**: Efficient model quantization
- **LoRA**: Parameter-efficient fine-tuning
- **Qwen2.5-Coder**: Pre-trained code generation model

---

## 📝 License & Attribution

- **Model**: Qwen2.5-Coder-1.5B by Alibaba Cloud  
- **Dataset**: 39K React components (from micro_ai_coder)
- **Fine-tuning Framework**: HuggingFace Transformers, PEFT
- **Quantization**: llama.cpp

---

**Ready to train and deploy?** Start with [QUICK_START.md](QUICK_START.md)!
