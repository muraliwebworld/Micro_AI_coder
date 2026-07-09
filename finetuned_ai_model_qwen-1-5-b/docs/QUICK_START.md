# Quick Start Guide - 5 Minutes

## Prerequisites Checklist
- [ ] Google account (for Colab)
- [ ] M2 Mac mini with 8GB RAM
- [ ] Homebrew installed (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- [ ] Dataset file: `data_cleaned_huggingface_new.jsonl` (39K React components)

---

## Step 1: Prepare Data for Colab (2 min)

```bash
# 1. Go to Google Drive: https://drive.google.com
# 2. Create folder: micro_ai_coder_data
# 3. Upload: data_cleaned_huggingface_new.jsonl
# 4. Note: Take 2-3 minutes for 39K samples (~25MB file)
```

---

## Step 2: Run Colab Notebook (50 min)

```bash
# 1. Open notebook in Google Colab:
#    - Copy: finetuned_ai_model_qwen-1-5-b/colab_notebooks/fine_tune_qwen_1_5b_colab.ipynb
#    - Paste into new Google Colab tab

# 2. Setup Colab:
#    - Runtime → Change runtime type → GPU (T4)
#    - Click "Save" to confirm

# 3. Run cells sequentially:
#    - Cell 1: GPU check (~30 sec)
#    - Cell 2: Mount Drive (~30 sec)
#    - Cell 3: Install packages (~2 min)
#    - Cell 4: Load dataset (~1 min)
#    - Cell 5: Load model (~2 min)
#    - Cell 6: Setup LoRA (~30 sec)
#    - Cell 7: Prepare datasets (~2 min)
#    - Cell 8: Configure training (~30 sec)
#    - Cell 9: START TRAINING (~45-60 min) ← Keep browser tab active!
#    - Cell 10: Save model (~2 min)
#    - Cell 11: Merge adapters (~3 min)

# Total: ~50-60 minutes
# Keep browser tab active (Colab times out after 12 hours idle)
```

**Expected Output**:
```
✅ GPU: Tesla T4 (14.56 GB VRAM)
✅ Tokenizer loaded
✅ Model loaded (8-bit quantized)
✅ LoRA adapters applied
✅ Dataset split: Train: 31200, Val: 3900, Test: 3900
✅ Training configured
🚀 Starting fine-tuning...
    Epoch 1: [=====>] Loss: 1.245
    Epoch 2: [=====>] Loss: 0.892
    Epoch 3: [=====>] Loss: 0.756
✅ Training complete!
✅ LoRA adapters saved
✅ Merged model saved to qwen_1_5b_reactjs_checkpoints/qwen_1_5b_reactjs_merged/
```

---

## Step 3: Download Model (5 min)

```bash
# On your M2 Mac:
# 1. Go to Google Drive
# 2. Navigate to: qwen_1_5b_reactjs_checkpoints/qwen_1_5b_reactjs_merged/
# 3. Download all files (should be ~3-4GB)
# 4. Extract and place in:
#    finetuned_ai_model_qwen-1-5-b/models/qwen_1_5b_reactjs_merged/

# Files needed:
#   - config.json
#   - pytorch_model.bin (largest)
#   - tokenizer.json
#   - tokenizer_config.json
#   - special_tokens_map.json
```

---

## Step 4: Convert to GGUF (15 min)

```bash
# Install conversion tool
pip install llama-cpp-python

# Run conversion (takes ~10 minutes)
python scripts/convert_to_gguf.py \
  --model_path models/qwen_1_5b_reactjs_merged \
  --output_path models/qwen_1_5b_reactjs_merged.Q4_K_M.gguf

# Expected output: qwen_1_5b_reactjs_merged.Q4_K_M.gguf (~1.5-2 GB)
```

---

## Step 5: Setup Ollama (5 min)

```bash
# Install Ollama
brew install ollama

# Start Ollama server (background)
ollama serve &

# Register model
ollama create micro-ai-coder-1-5b:latest -f models/Modelfile_qwen_1_5b

# Verify
ollama list | grep micro-ai-coder-1-5b

# Test generation
ollama run micro-ai-coder-1-5b:latest "Write a React button component"
```

**Expected**: Model generates valid React code in ~30-60 seconds

---

## Step 6: Run Agent (2 min)

```bash
# Make sure Ollama server is running
ollama serve &

# Run agent
python phase4_agent/micro_ai_coder_agent_1_5b.py

# Menu options:
#   1. Generate React component (single)
#   2. Batch generate (multiple)
#   3. View generation logs
#   4. Exit
```

---

## ✅ Validation (Optional - 5 min)

```bash
# Run validation tests
python tests/validate_finetuned_qwen_1_5b.py

# Expected output:
# Test 1: Button component ... PASS ✅
# Test 2: Counter component ... PASS ✅
# ...
# Results: 18/20 passed (90%) ✅
```

---

## 🎯 Total Time Breakdown

| Step | Time | Status |
|------|------|--------|
| Prepare data | 2 min | One-time |
| Colab training | 50 min | Active |
| Download model | 5 min | One-time |
| Convert GGUF | 15 min | One-time |
| Setup Ollama | 5 min | One-time |
| **Total** | **77 min** | **Complete setup** |

---

## 🔗 Important Links

- **Google Colab**: https://colab.research.google.com
- **Google Drive**: https://drive.google.com
- **Ollama**: https://ollama.ai
- **Qwen Model**: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct

---

## ❓ FAQ

**Q: Can I run this on non-free Colab?**  
A: Yes! Pro/Pro+ tiers have better GPUs (A40/H100), but free T4 is sufficient.

**Q: What if training gets interrupted?**  
A: Colab saves checkpoints to Drive. Re-run Step 9 with `trainer.train(resume_from_checkpoint=True)`

**Q: Can I use a different Mac?**  
A: Yes! Works on any Apple Silicon (M1/M2/M3/M4). Adjust expectations:
   - M1: Similar performance to M2
   - M3/M4: 1.5-2x faster inference

**Q: How much storage do I need?**  
A: ~6GB temporary (model download) + 2GB final (GGUF format)

**Q: Can I generate other languages besides React?**  
A: This model is fine-tuned only for React. For multi-language, retrain with diverse dataset.

---

## 🚀 Next Steps

1. ✅ Complete steps 1-6 above
2. ✅ Run validation tests
3. ✅ Customize prompts in `phase4_agent/micro_ai_coder_agent_1_5b.py`
4. ✅ Integrate into your development workflow

**Happy coding! 🎉**
