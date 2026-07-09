# Fine-Tuned Qwen2.5-Coder for Micro AI Coder

This folder contains everything needed to fine-tune `Qwen2.5-Coder-7B-Instruct` on your React dataset and deploy it as a custom Ollama model for the Micro AI Coder project.

## 📋 Quick Start

### Step 1: Run Fine-Tuning on Google Colab
1. Open `colab_notebooks/fine_tune_qwen_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Follow the step-by-step cells to fine-tune the model on your data
3. Expected time: 2-3 hours on T4 GPU
4. Model will be saved to Google Drive

### Step 2: Download & Convert Model
1. Download the merged model from Google Drive
2. Place in `finetuned_ai_model/models/qwen_reactjs_merged/`
3. Run conversion script to create GGUF format
4. Output: `models/qwen_reactjs_merged.Q4_K_M.gguf`

### Step 3: Deploy as Ollama Model
1. Register model with Ollama: `ollama create micro-ai-coder-v2:latest -f models/Modelfile_qwen_reactjs`
2. Test: `ollama run micro-ai-coder-v2:latest "Write a React login form"`

### Step 4: Update Phase 3/4
1. Copy `phase3_inference/v3_inference_finetuned.py` to main project
2. Copy `phase4_agent/micro_ai_coder_agent_v2.py` to main project
3. Update imports to use new inference module
4. Run validation tests: `python tests/validate_finetuned_qwen.py`

## 📁 Folder Structure

```
finetuned_ai_model/
├── colab_notebooks/
│   └── fine_tune_qwen_colab.ipynb     ← Run this on Google Colab
├── phase3_inference/
│   └── v3_inference_finetuned.py      ← Copy to main project
├── phase4_agent/
│   └── micro_ai_coder_agent_v2.py     ← Copy to main project
├── tests/
│   └── validate_finetuned_qwen.py     ← Run local validation
├── models/
│   ├── Modelfile_qwen_reactjs         ← Ollama model definition
│   └── qwen_reactjs_merged/           ← Store merged model here
├── FINE_TUNING_GUIDE.md               ← Detailed step-by-step guide
└── README.md                          ← This file
```

## 🎯 Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base Model | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B params, better quality |
| Training Data | `data_cleaned_huggingface_new.jsonl` | 39K React components |
| LoRA Rank | 64 | Efficient fine-tuning |
| Batch Size | 4 | Per device (T4 GPU) |
| Learning Rate | 2e-4 | Conservative for pre-trained |
| Epochs | 3 | Prevent overfitting |
| Expected Val Loss | < 1.0 | vs. 1.88 currently |
| Training Time | 2-3 hours | On T4 GPU |

## 📊 Expected Results

✅ **After Step 1 (Training)**:
- Validation loss < 1.2
- No divergence in loss curves
- Model saved to Google Drive

✅ **After Step 3 (Ollama)**:
- Model registered: `ollama list | grep micro-ai-coder-v2`
- Response time < 10 seconds on CPU
- Generates valid React code

✅ **After Step 4 (Validation)**:
- 20/20 test samples pass syntax validation
- ≥95% success rate (vs. ~1% with old model)
- No fallback templates needed

## ⚠️ Prerequisites

- Google Colab account (free T4 GPU access)
- Stable internet (for 3+ hour training session)
- ~5 GB local storage (for model download)
- Ollama running locally (`ollama serve`)

## 🔗 Related Files

- Training data: `datasets/data_cleaned_huggingface_new.jsonl` (39K samples)
- Current model: `phase2_training/v2_train_model.py` (old approach)
- Main inference: `phase3_inference/v2_inference.py` (old model)
- Main agent: `phase4_agent/micro_ai_coder_agent.py` (old model)

## 📝 Notes

- The new model is a **drop-in replacement** for the current system
- All API calls remain the same (Ollama HTTP endpoint)
- Fallback templates are kept as safety net
- Training is non-destructive (doesn't modify source data)

## 🚀 Next Steps

1. **Start fine-tuning now**: Open Colab notebook
2. **Monitor training**: Watch loss curves for convergence
3. **Download model**: Save to local machine when complete
4. **Convert to GGUF**: Quantize for efficient deployment
5. **Test locally**: Validate with 20 sample prompts
6. **Deploy**: Replace old model in production

---

**Last Updated**: 2026-07-09  
**Status**: Ready for implementation  
**Estimated Success Rate**: ≥95% valid code generation (vs. ~1% currently)
