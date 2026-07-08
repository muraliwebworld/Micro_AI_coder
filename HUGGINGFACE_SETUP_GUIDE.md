# Hugging Face Dataset Integration Guide

## Overview

This is a complete setup for training and inferencing using Hugging Face datasets. It includes:

1. **Data Analysis Script** - Understand your HF dataset
2. **Training Script (v2_train_model_huggingface.py)** - Train on HF data
3. **Inference Agent (micro_ai_coder_agent_huggingface.py)** - Generate components with HF model

---

## Step 1: Analyze Your Dataset

First, understand the structure of your Hugging Face dataset:

```bash
cd /Users/muralidharanramasamy/Micro_AI_coder
python3 scripts/analyze_hf_dataset.py
```

**Output will show:**
- Total entries in dataset
- Field names and frequency
- Code length statistics
- Sample entries

**Example Output:**
```
✅ Total entries: 15,000

📋 Field names and frequencies:
  code                : 15,000 (100%)
  prompt              :  8,500 ( 57%)
  response            :  7,200 ( 48%)
  ...

📏 Code length statistics:
  Min: 100 chars
  Max: 8,500 chars
  Avg: 2,100 chars
```

---

## Step 2: Train Model on HF Data

The training script is optimized for larger datasets:

```bash
python3 phase2_training/v2_train_model_huggingface.py
```

**Configuration:**
- **BATCH_SIZE**: 16 (larger for more data)
- **MAX_ITERS**: 10,000 (more iterations)
- **WARMUP_STEPS**: 500 (learning rate ramp-up)
- **EVAL_INTERVAL**: 200 (validate every 200 steps)
- **PATIENCE**: 15 (early stopping)

**Model Architecture:**
- Embedding: 256 dimensions
- Heads: 8
- Layers: 12
- Total params: ~5M

**Outputs:**
- `models/tiny_code_model_huggingface.pt` - Trained model
- `models/model_config_huggingface.json` - Configuration
- `logs/training_log_huggingface.jsonl` - Per-iteration metrics

**Training Output Example:**
```
Iter    100 | train_loss: 2.5432 | val_loss: 2.4102 ↓ | lr: 2.00e-05
Iter    200 | train_loss: 2.3421 | val_loss: 2.2891 ↓ | lr: 1.00e-04
           ✅ New best model saved (val_loss: 2.2891)
...
Iter  10000 | train_loss: 1.2345 | val_loss: 1.4567 ↑ | lr: 1.50e-06

✅ TRAINING COMPLETE!
📊 Final metrics:
   Best validation loss: 1.4123
   Model saved to: models/tiny_code_model_huggingface.pt
   Config saved to: models/model_config_huggingface.json
   Training log: logs/training_log_huggingface.jsonl
   Total parameters: 5.12M
```

**Monitor Training:**
```bash
# Watch training progress in real-time
tail -f logs/training_log_huggingface.jsonl | python3 -m json.tool
```

---

## Step 3: Generate Components with HF Model

Run the interactive inference agent:

```bash
python3 phase4_agent/micro_ai_coder_agent_huggingface.py
```

**Menu Options:**

```
====================================================================
REACT COMPONENT INFERENCE - HUGGING FACE MODEL
====================================================================

1. Generate a React component
2. Generate multiple components
3. View error logs
4. Exit

➤ Choice (1-4):
```

### Option 1: Generate Single Component

```
💬 Describe a React component:
➤ Prompt: Create a shopping cart with add/remove items

🔄 Generating component...

✅ COMPONENT GENERATED
Location: /Micro_AI_coder/outputs/20260708_120530_hf/
  - component.jsx (1,245 chars)
  - explanation.md (380 chars)
```

### Option 2: Batch Generation

```
📋 Generate multiple components
➤ How many? (1-10): 5

Enter 5 prompts:
  1. Login form with validation
  2. Dark mode toggle
  3. Search bar with autocomplete
  4. User profile card
  5. Navigation menu

[1/5] ✅ Generated
[2/5] ✅ Generated
[3/5] ✅ Generated
[4/5] ✅ Generated
[5/5] ✅ Generated
```

### Option 3: View Error Logs

```
====================================================================
ERROR LOGS
====================================================================

Total errors: 3

  Prompt: Shopping cart component
  Errors: Unbalanced parentheses; Missing export
  
  Prompt: Payment form
  Errors: Corrupted import
```

---

## File Structure

```
Micro_AI_coder/
├── datasets/
│   ├── data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl    # Your HF dataset
│   ├── generated_projects_final.jsonl                      # Original training data
│   └── ...
│
├── phase2_training/
│   ├── v2_train_model.py                                  # Original trainer
│   └── v2_train_model_huggingface.py                       # HF trainer (NEW)
│
├── phase4_agent/
│   ├── micro_ai_coder_agent.py                            # Original agent
│   └── micro_ai_coder_agent_huggingface.py                 # HF agent (NEW)
│
├── scripts/
│   └── analyze_hf_dataset.py                              # Dataset analyzer (NEW)
│
├── models/
│   ├── tiny_code_model.pt                                 # Original model
│   ├── model_config.json                                  # Original config
│   ├── tiny_code_model_huggingface.pt                      # HF model (NEW)
│   └── model_config_huggingface.json                       # HF config (NEW)
│
├── logs/
│   ├── training_log.jsonl                                 # Original training log
│   ├── training_log_huggingface.jsonl                      # HF training log (NEW)
│   ├── inference_errors.jsonl                             # Original error log
│   └── inference_errors_huggingface.jsonl                  # HF error log (NEW)
│
└── outputs/
    ├── 20260707_183640/                                   # Original outputs
    ├── 20260708_120530_hf/                                # HF outputs (NEW)
    └── ...
```

---

## Comparison: Original vs. Hugging Face

| Feature | Original | Hugging Face |
|---------|----------|--------------|
| **Training Data** | 290 entries | 1,000+ entries |
| **Batch Size** | 8 | 16 |
| **Max Iterations** | 6,000 | 10,000 |
| **Model File** | `tiny_code_model.pt` | `tiny_code_model_huggingface.pt` |
| **Error Log** | `inference_errors.jsonl` | `inference_errors_huggingface.jsonl` |
| **Agent Script** | `micro_ai_coder_agent.py` | `micro_ai_coder_agent_huggingface.py` |

---

## Features

### ✅ Data Analysis
- Automatic field detection (code, response, solution, etc.)
- Statistics: count, code length, field frequency
- Sample inspection

### ✅ Training Improvements
- **Flexible field detection** - Works with different HF dataset schemas
- **Learning rate warmup** - Smoother training start
- **Validation monitoring** - Every 200 iterations
- **Best model checkpointing** - Saves best model automatically
- **Early stopping** - Stops if validation loss doesn't improve
- **Gradient clipping** - Prevents exploding gradients
- **Comprehensive logging** - Per-iteration metrics to JSONL

### ✅ Error Handling
- **Code validation** - Checks for balanced parentheses, proper imports/exports
- **Corruption detection** - Identifies malformed code patterns
- **Error logging** - Logs invalid code with context for future analysis
- **Fallback templates** - Always produces valid React code
- **4 prompt-aware templates** - Login forms, counters, todo lists, generic

### ✅ Inference Features
- **Temperature control** - Adjust output diversity (default: 0.6)
- **Top-K sampling** - Limits vocabulary (default: 40)
- **Max tokens** - Control output length (default: 800)
- **Interactive menu** - Single/batch generation, error inspection
- **Timestamped outputs** - Organized by generation time

---

## Troubleshooting

### Issue: Dataset not found
**Solution:** Make sure you've placed the Hugging Face dataset at:
```
datasets/data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl
```

### Issue: "No valid code samples found"
**Solution:** Check dataset format. Run analysis script:
```bash
python3 scripts/analyze_hf_dataset.py
```
The script looks for: `code`, `response`, `solution`, `content`, `body` fields.

### Issue: CUDA out of memory
**Solution:** Reduce BATCH_SIZE in training script (currently 16, try 8)

### Issue: Training is too slow
**Solution:** Training should take 10-30 minutes on GPU. CPU will be slower.

### Issue: Generated code has errors
**Solution:** The system automatically logs errors and uses fallback templates. Check:
```bash
tail logs/inference_errors_huggingface.jsonl | python3 -m json.tool
```

---

## Quick Start Script

```bash
#!/bin/bash
cd /Users/muralidharanramasamy/Micro_AI_coder

echo "🔍 Analyzing dataset..."
python3 scripts/analyze_hf_dataset.py

echo -e "\n🚀 Training model..."
python3 phase2_training/v2_train_model_huggingface.py

echo -e "\n🧪 Testing inference..."
python3 phase4_agent/micro_ai_coder_agent_huggingface.py
```

Save as `run_hf_pipeline.sh` and run:
```bash
chmod +x run_hf_pipeline.sh
./run_hf_pipeline.sh
```

---

## Next Steps

1. **Run analysis:** `python3 scripts/analyze_hf_dataset.py`
2. **Train model:** `python3 phase2_training/v2_train_model_huggingface.py`
3. **Test inference:** `python3 phase4_agent/micro_ai_coder_agent_huggingface.py`
4. **Monitor training:** Watch logs in real-time or check JSON output
5. **Compare models:** Test both original and HF models on same prompts

---

## Performance Expectations

- **Training time (GPU):** 10-30 minutes (depends on dataset size)
- **Training time (CPU):** 1-3 hours
- **Inference time:** 2-5 seconds per component
- **Model size:** ~20 MB
- **Memory usage (training):** 4-6 GB GPU / 8-12 GB CPU

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review script output and error logs
3. Run analysis script to verify dataset format
4. Check logs for detailed error context

Happy coding! 🚀
