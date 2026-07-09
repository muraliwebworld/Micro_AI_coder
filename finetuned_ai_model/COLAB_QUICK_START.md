# Quick Start: Google Colab Fine-Tuning

Use this guide to fine-tune the model directly in Google Colab (browser-based, no local setup needed).

## ⏱️ Timeline
- **Setup**: 10 minutes
- **Training**: 2-3 hours (on T4 GPU)
- **Model Download**: 10 minutes
- **Total**: 3-4 hours

## 🚀 Quick Steps

### 1. Open Colab Notebook (2 minutes)

**Option A: Direct Upload**
1. Go to [Google Colab](https://colab.research.google.com)
2. Click "File" → "Upload notebook"
3. Select `fine_tune_qwen_colab.ipynb` from `finetuned_ai_model/colab_notebooks/`
4. Click "Open"

**Option B: From GitHub** (if you push to GitHub)
1. Go to [Google Colab](https://colab.research.google.com)
2. Click "GitHub" tab
3. Paste your repo URL
4. Select the notebook file

### 2. Set GPU (1 minute)

1. Click "Runtime" menu
2. Select "Change runtime type"
3. Choose:
   - **Runtime type**: Python 3
   - **Hardware accelerator**: **GPU** (T4 is fine, free tier)
4. Click "Save"

### 3. Prepare Dataset (5 minutes)

1. Go to [Google Drive](https://drive.google.com)
2. Create a new folder: `micro_ai_coder_data`
3. Upload `data_cleaned_huggingface_new.jsonl` to this folder
4. Come back to Colab

### 4. Run Cells in Order (3-4 hours total)

In the Colab notebook, run each cell sequentially:

#### Cell 1: Check GPU
```python
# Verify T4 GPU is available
```
**Expected**: `CUDA available: True`, `GPU: Tesla T4`, `VRAM: 16.0 GB`

#### Cell 2: Mount Google Drive
```python
# Authorize access to Drive
```
**Expected**: "Google Drive mounted"

#### Cell 3: Install Dependencies
```python
# Download packages
```
**Expected**: "Dependencies installed" (takes 3-5 min)

#### Cell 4: Load Dataset
```python
# Load 39K React components
```
**Expected**: "✅ Loaded 39000 samples"

#### Cell 5-6: Load Base Model
```python
# Download Qwen2.5-Coder-7B-Instruct
```
**Expected**: "✅ Model loaded (8-bit quantized)" (takes 2-3 min)

#### Cell 7: Setup LoRA
```python
# Configure LoRA adapters
```
**Expected**: Shows trainable parameters (~20M)

#### Cell 8: Prepare Data
```python
# Tokenize and split dataset
```
**Expected**: "✅ Dataset split" with train/val/test counts

#### Cell 9: Configure Training
```python
# Setup training parameters
```
**Expected**: "✅ Training configured"

#### Cell 10: Start Training ⏱️ (2-3 hours)
```python
# Fine-tune model
trainer.train()
```

**Monitor Progress**:
- Watch the loss curves
- Training loss should decrease
- Validation loss should converge < 1.2
- This is the **longest step** (~2-3 hours)

**Expected Output**:
```
Epoch 1/3: train_loss=1.82, val_loss=1.22
Epoch 2/3: train_loss=0.91, val_loss=0.99
Epoch 3/3: train_loss=0.62, val_loss=0.88
✅ Training complete!
```

#### Cell 11: Save Model (10 minutes)
```python
# Save LoRA adapters and tokenizer
```
**Expected**: "✅ LoRA adapters saved"

#### Cell 12: Merge Adapters (5-10 minutes)
```python
# Merge LoRA into base model
```
**Expected**: "✅ Merged model saved" (~7.6 GB to Google Drive)

#### Cell 13: Validation (Optional)
```python
# Test generation on sample prompts
```
**Expected**: Generated React code samples

---

## 📥 After Training: Download Model

### Step 1: Find Model in Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Navigate to: `qwen_reactjs_checkpoints/qwen_reactjs_merged/`
3. You should see:
   - `pytorch_model.bin` (large file, ~7.6 GB)
   - `config.json`
   - `tokenizer.json`
   - `special_tokens_map.json`
   - `tokenizer_config.json`

### Step 2: Download to Local Machine
1. Right-click on `qwen_reactjs_merged` folder
2. Select "Download"
3. Wait for download (~30 min on typical internet)

### Step 3: Place in Project
```bash
# On your local machine
mv qwen_reactjs_merged ~/Micro_AI_coder/finetuned_ai_model/models/
```

---

## 🔄 Next Steps (Local Machine)

Once model is downloaded, follow [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md) steps 16-25 to:
1. Convert model to GGUF format
2. Create Ollama modelfile
3. Register with Ollama
4. Update Phase 3/4 and test

---

## ⚠️ Troubleshooting

### Issue: "Disconnected" Message
**Solution**: 
- Colab has 12-hour timeout per session
- Keep browser tab open during training
- Checkpoints auto-save to Google Drive every epoch
- Can resume from checkpoint if needed

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size
```python
# In Cell 9, change:
per_device_train_batch_size=2,  # was 4
gradient_accumulation_steps=4,   # was 2
```

### Issue: Training Loss Not Decreasing
**Solution**: Lower learning rate
```python
# In Cell 9, change:
learning_rate=1e-4,  # was 2e-4
```

### Issue: Model Converges Too Early
**Solution**: This is normal
- If validation loss plateaus after epoch 2, training is complete
- Can stop early without issue
- Model is already good enough

### Issue: Dataset Not Found
**Solution**:
1. Check Google Drive path: `/content/drive/My Drive/micro_ai_coder_data/data_cleaned_huggingface_new.jsonl`
2. Ensure exact filename and capitalization
3. Re-run Cell 4 to verify

---

## ✅ Success Checklist

- [ ] T4 GPU detected in Colab
- [ ] Google Drive mounted
- [ ] Dependencies installed
- [ ] Dataset loaded (39K samples)
- [ ] Base model downloaded (~7.6 GB)
- [ ] LoRA adapters configured
- [ ] Training starts without errors
- [ ] Validation loss < 1.2 after training
- [ ] Model saved to Google Drive
- [ ] Model downloaded to local machine
- [ ] Ready for GGUF conversion

---

## 📊 Expected Results

| Metric | Current Model | Fine-Tuned | Improvement |
|--------|--------------|-----------|------------|
| Validation Loss | 1.88 | <0.9 | ✅ Better |
| Success Rate | ~1% | ≥95% | ✅ Huge |
| Valid Code Gen | Rare | Common | ✅ Expected |
| Fallback Usage | Frequent | Rare | ✅ Good |

---

## 💾 Files Created

```
Google Drive/
├── micro_ai_coder_data/
│   └── data_cleaned_huggingface_new.jsonl  (uploaded)
└── qwen_reactjs_checkpoints/              (created by training)
    ├── checkpoint-xxxx/                   (intermediate)
    └── qwen_reactjs_merged/               (final model)
        ├── pytorch_model.bin              (7.6 GB)
        ├── config.json
        ├── tokenizer.json
        └── ...
```

---

## 🎯 Next: Post-Training Steps

After downloading the merged model, return to [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md) for:
- **Step 16-19**: Convert to GGUF and deploy
- **Step 20-21**: Validate locally
- **Step 22-25**: Integrate into main project

---

**Status**: Ready to start  
**Estimated Time**: 3-4 hours total  
**Next Action**: Open Colab notebook and run cells in order
