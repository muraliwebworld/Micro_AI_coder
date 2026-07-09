# Implementation Plan - Qwen2.5-Coder-1.5B on M2 Mac

## Project Overview

Fine-tune Qwen2.5-Coder-1.5B (1.5 billion parameter model) on 39K React components using LoRA adapters, optimized for local training on M2 Mac mini with 8GB RAM.

**Target Specifications:**
- Model: Qwen2.5-Coder-1.5B-Instruct
- Fine-tuning approach: LoRA (Low-Rank Adaptation)
- Training data: 39K React component examples
- Output format: GGUF Q4_K_M (~800MB)
- Inference: Ollama on M2 Mac
- Expected success rate: ≥85% valid React code

---

## Phase 1: Preparation (30 minutes)

### 1.1 Environment Setup
- [ ] Create Python 3.10+ virtual environment
- [ ] Install dependencies from `requirements.txt`
- [ ] Verify Metal GPU support (`torch.backends.mps`)
- [ ] Confirm model path configuration

**Files involved:**
- `requirements.txt` - all dependencies
- `docs/SETUP_GUIDE.md` - detailed instructions
- `scripts/fine_tune_local.py` - uses venv

**Expected outcome:**
```
✅ Metal available: True
✅ All packages installed
✅ Ready for training
```

### 1.2 Dataset Preparation
- [ ] Copy `data_cleaned_huggingface_new.jsonl` to `datasets/`
- [ ] Verify 39,000 samples loaded
- [ ] Confirm JSON format: `{"prompt": "...", "code": "...", "Complex_CoT": "..."}`

**Files involved:**
- `datasets/data_cleaned_huggingface_new.jsonl` - input

**Expected outcome:**
```
✅ Dataset: 39000 samples
   Prompt: "Write a React button..."
   Code length: 150-400 chars
```

### 1.3 Model Verification
- [ ] Confirm Qwen2.5-Coder-1.5B is accessible (HuggingFace hub or local)
- [ ] Model path set correctly in `scripts/fine_tune_local.py`
- [ ] 1.5B parameter count verified

**Command:**
```bash
python3 -c "from transformers import AutoModelForCausalLM; print(AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct'))"
```

---

## Phase 2: Fine-Tuning (4-6 hours)

### 2.1 Local Training Execution
- [ ] Run training script: `python scripts/fine_tune_local.py`
- [ ] Monitor training loss convergence
- [ ] Verify no out-of-memory errors

**Files involved:**
- `scripts/fine_tune_local.py` - training script
- `checkpoints/` - checkpoint directory (auto-created)

**Expected output:**
```
✅ GPU: Apple Metal (mps)
✅ Model loaded: 1.5B parameters
✅ Dataset split:
   Train: 31200 samples
   Val: 3900 samples
   Test: 3900 samples
🚀 Starting training...
Epoch 1: 100%|████| 3900/3900 [110m:30s]
  Loss: 1.84, Val Loss: 1.62
Epoch 2: 100%|████| 3900/3900 [105m:15s]
  Loss: 1.32, Val Loss: 1.28
✅ Training complete!
```

### 2.2 Training Monitoring
- [ ] Check memory usage stays 6.5-7.5GB
- [ ] Verify loss curves
  - Training loss: should decrease monotonically
  - Validation loss: should converge < 1.5
- [ ] Model saves every ~30 minutes to `checkpoints/`

**Key metrics:**
- Initial val_loss: ~2.0
- Final val_loss target: <1.5
- Training time per epoch: 90-120 min
- GPU memory: 6.5-7.5GB (consistent)

### 2.3 Training Completion
- [ ] Training completes successfully (no interruptions)
- [ ] Model merged and saved to `models/qwen_reactjs_merged/`
- [ ] Files created: `pytorch_model.bin`, `config.json`, `tokenizer.json`

**Files created:**
- `models/qwen_reactjs_merged/` - fine-tuned model

---

## Phase 3: Model Conversion (10-30 minutes)

### 3.1 Convert to GGUF Format
- [ ] Run: `python scripts/convert_to_gguf.py`
- [ ] Select quantization: Q4_K_M (4-bit, ~800MB)
- [ ] Verify output file created

**Files involved:**
- `scripts/convert_to_gguf.py` - conversion script

**Expected output:**
```
✅ GGUF file created successfully!
   File: models/qwen_reactjs_merged.Q4_K_M.gguf
   Size: 850 MB
```

### 3.2 Verify GGUF File
- [ ] File size: ~800-900MB
- [ ] File location: `models/qwen_reactjs_merged.Q4_K_M.gguf`
- [ ] Modelfile present: `models/Modelfile`

**Command:**
```bash
ls -lh models/qwen_reactjs_merged.Q4_K_M.gguf
# Should show: -rw-r--r--  850M  Jul  9 14:30  qwen_reactjs_merged.Q4_K_M.gguf
```

---

## Phase 4: Ollama Deployment (15 minutes)

### 4.1 Install/Verify Ollama
- [ ] Ollama installed on M2 Mac
- [ ] Ollama service running: `ollama serve`

**Verification:**
```bash
ollama --version
# Should show: version X.X.X
```

### 4.2 Create Ollama Model
- [ ] Run: `ollama create micro-ai-coder-1-5b:latest -f models/Modelfile`
- [ ] Wait for model to load (~2-3 minutes)

**Expected output:**
```
writing model
✅ success
```

### 4.3 Verify Model is Available
- [ ] Run: `ollama list`
- [ ] Output shows: `micro-ai-coder-1-5b:latest`

**Command:**
```bash
ollama list | grep micro-ai-coder-1-5b
```

---

## Phase 5: Testing & Validation (30 minutes)

### 5.1 Quick Manual Test
- [ ] Run: `ollama run micro-ai-coder-1-5b:latest "Write a React button"`
- [ ] Verify output is valid React code

**Expected:**
```jsx
export const Button = ({ label, onClick }) => {
  return <button onClick={onClick}>{label}</button>;
};
```

### 5.2 Comprehensive Validation
- [ ] Run: `python tests/validate_finetuned_qwen.py`
- [ ] Expected: ≥17/20 tests pass (≥85% success rate)

**Files involved:**
- `tests/validate_finetuned_qwen.py` - test suite

**Expected output:**
```
📊 Test Results Summary
Total tests: 20
Passed: 17 (85.0%)
Failed: 3 (15.0%)
🎉 85.0% - EXCELLENT! Model is production-ready!
```

### 5.3 Interactive Inference
- [ ] Run: `python scripts/inference_finetuned.py`
- [ ] Generate 5-10 sample components
- [ ] Verify all code is valid React

**Expected:**
```
React Component Generator
1. Single generation
2. Batch generation (test prompts)
3. Exit

Select option (1-3): 1
📝 Enter prompt: Write a React counter component
✅ Valid React Code:
   [Generated code preview]
```

---

## Phase 6: Deployment & Integration (Optional)

### 6.1 Integration with Micro AI Coder
- [ ] Copy inference script to main project
- [ ] Update phase3_inference module
- [ ] Test with main application

### 6.2 Documentation
- [ ] Document final model performance
- [ ] Create deployment runbook
- [ ] Update README with results

---

## Troubleshooting Checklist

### Memory Issues (OOM)
**If:** `RuntimeError: CUDA out of memory`
**Solution:**
- Edit `scripts/fine_tune_local.py`:
  - `BATCH_SIZE = 1` → `0.5` (gradient accumulation = 8)
  - `MAX_LENGTH = 512` → `256`

### Training Too Slow
**If:** >2 minutes per training step
**Solution:**
- Verify Metal GPU is active: check Activity Monitor GPU%
- Ensure no other heavy processes running
- If CPU-only, training will take 24+ hours

### Model Not Generating Good Code
**If:** <70% success rate on validation
**Solution:**
- Increase training epochs: `EPOCHS = 2` → `3` (add 3-4 hours)
- Increase training data if available
- Lower learning rate: `LEARNING_RATE = 5e-4` → `3e-4`

### Ollama Model Not Found
**If:** `error: model 'micro-ai-coder-1-5b:latest' not found`
**Solution:**
```bash
# Verify GGUF file exists
ls -lh models/qwen_reactjs_merged.Q4_K_M.gguf

# Recreate Ollama model
ollama create micro-ai-coder-1-5b:latest -f models/Modelfile
```

### GGUF Conversion Failed
**If:** `error during conversion`
**Solution:**
- Verify fine-tuned model files:
  ```bash
  ls models/qwen_reactjs_merged/
  # Should show: pytorch_model.bin, config.json, tokenizer.json, ...
  ```
- Try alternative quantization: `Q3_K_M` (smaller, faster)

---

## Timeline Estimates

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Preparation | 30 min |
| 2 | Fine-tuning | 4-6 hours |
| 3 | GGUF conversion | 10-30 min |
| 4 | Ollama deployment | 15 min |
| 5 | Validation testing | 30 min |
| **Total** | **All phases** | **5-7 hours** |

---

## Success Criteria

- [x] Setup complete with no errors
- [x] Training converges with val_loss < 1.5
- [x] GGUF file created (~800MB)
- [x] Ollama model deployed and accessible
- [x] Validation tests pass ≥85%
- [x] Generates syntactically valid React code consistently

---

## Deliverables

1. **Fine-tuned Model**: `models/qwen_reactjs_merged.Q4_K_M.gguf` (800MB)
2. **Ollama Model**: `micro-ai-coder-1-5b:latest` (registered)
3. **Inference Script**: `scripts/inference_finetuned.py` (ready to use)
4. **Test Results**: `outputs/test_results_*.md` (≥85% pass rate)
5. **Documentation**: All guides and setup instructions

---

## Post-Deployment

### Monitor Model Performance
- Track inference latency
- Monitor generation quality
- Collect failed generations for retraining

### Potential Improvements
- Fine-tune with additional tech stacks (Node, PHP, etc.)
- Create ensemble of multiple models
- Add prompt engineering layer
- Implement fallback generation strategies

