# Quick Start Guide - Updated Pipeline

## What Changed? 🔄

Your code generation pipeline is now **focused on React components** instead of multi-file projects.

**Before**: Generate full-stack projects (React + Express + MySQL + etc.)  
**After**: Generate individual React components with explanations

---

## The 4 Phases

### Phase 1: Generate Training Data
```bash
python phase1_dataset_creator/v2_dataset_creator.py
```
**Creates**: React component dataset in format matching `reactjs_projects_dataset.jsonl`  
**Output**: `datasets/generated_projects_final.jsonl`  
**Data**: `{prompt, code, Complex_CoT}`

---

### Phase 2: Train Model
```bash
python phase2_training/v2_train_model.py
```
**Trains**: PyTorch transformer on React components  
**Output**: `models/tiny_code_model.pt` + `models/model_config.json`  
**Time**: ~5-10 minutes (depending on dataset size)

---

### Phase 3: Test Inference (Optional)
```bash
python phase3_inference/v2_inference.py
```
**Tests**: Component generation directly  
**Interactive Menu**:
- 1: Generate one component
- 2: Generate multiple components
- 3: Exit

---

### Phase 4: Main Agent (Production)
```bash
python phase4_agent/micro_ai_coder_agent.py
```
**Main Interface**: Interactive component generation  
**Interactive Menu**:
- 1️⃣ Generate a React component
- 2️⃣ Generate multiple components (batch)
- 3️⃣ Get prompt format help
- 4️⃣ View output directory
- 5️⃣ Exit

---

## Output Structure

Each generated component is saved in: `outputs/YYYYMMDD_HHMMSS/`

```
component.jsx       ← Your React component code
explanation.md      ← How it works (Chain-of-Thought)
metadata.json       ← Timestamp and prompt
```

---

## Example Prompts

✅ **Good Prompts**:
- "Generate a React login form with email and password fields"
- "Create a React counter component with increment and decrement buttons"
- "Write a React todo list that can add and remove items"
- "Build a React product card with image, title, price, and rating"
- "Create a React form with validation for email and phone number"

❌ **Avoid**:
- "Create a component" (too vague)
- "Make an app" (not specific enough)
- Just one word like "form" or "button"

---

## Typical Workflow

### 1. Generate Dataset (One-time setup)
```bash
# Generate 10-20 sample components for training
python phase1_dataset_creator/v2_dataset_creator.py

# Follow the prompts:
# - Describe a React component
# - System generates code and explanation
# - Repeat 10-20 times
```

### 2. Retrain (Optional, if new components)
```bash
# Train model on your new dataset
python phase2_training/v2_train_model.py
```

### 3. Generate Components (Main use)
```bash
# Use the agent to generate new components
python phase4_agent/micro_ai_coder_agent.py

# Choose option 1 or 2 from menu
# Describe your component
# Get component.jsx + explanation.md
```

---

## Key Files Modified

| File | Change | Impact |
|------|--------|--------|
| `v2_dataset_creator.py` | Complete rewrite | Now generates single components with CoT |
| `v2_train_model.py` | Minor update | Better dataset loading |
| `v2_inference.py` | Complete rewrite | Simpler component generation |
| `micro_ai_coder_agent.py` | Complete rewrite | Cleaner UI and batch mode |

---

## Dataset Format

### New Format (what you should create)
```json
{
  "prompt": "Generate a React component for a login form",
  "code": "import React, { useState } from 'react';\n...",
  "Complex_CoT": "The prompt asks for a login form. I import React..."
}
```

### Matches Your Reference Dataset
Your existing `datasets/reactjs_projects_dataset.jsonl` already has this format!

---

## Troubleshooting

### Phase 1: Can't connect to Ollama
```bash
# Make sure Ollama is running
ollama serve
# In another terminal, ensure qwen2.5-coder:3b is available
ollama pull qwen2.5-coder:3b
```

### Phase 2: Not enough data
- Phase 1 requires at least 5-10 components to train
- More data (20-50) produces better results

### Phase 3/4: Model file not found
```bash
# Make sure you ran Phase 2 first
python phase2_training/v2_train_model.py
```

---

## Performance Tips

1. **Better Components**: Use specific prompts
   - ✅ "React form with email validation and submit button"
   - ❌ "React form"

2. **Batch Generation**: Generate 10 at once instead of 1 each time
   - Use Phase 4 option 2️⃣

3. **Review Output**: Check generated components and reuse good patterns
   - Look at `outputs/*/component.jsx`
   - Copy patterns you like

4. **Retrain Regularly**: Add new components to dataset and retrain
   - Better dataset = better components

---

## File Locations

```
your-project/
├── phase1_dataset_creator/
│   └── v2_dataset_creator.py          ← Dataset generator
├── phase2_training/
│   └── v2_train_model.py              ← Model training
├── phase3_inference/
│   └── v2_inference.py                ← Testing/inference
├── phase4_agent/
│   └── micro_ai_coder_agent.py        ← Main agent (use this!)
├── datasets/
│   └── generated_projects_final.jsonl  ← Your training data
├── models/
│   ├── tiny_code_model.pt              ← Trained model
│   └── model_config.json               ← Model config
└── outputs/
    ├── 20260708_120000/
    │   ├── component.jsx               ← Generated component
    │   ├── explanation.md              ← How it works
    │   └── metadata.json               ← Info
    └── ...more outputs...
```

---

## Next Steps

1. ✅ **All code is updated** - No more multi-file project generation
2. 📊 **Test with Phase 1** - Generate 5-10 sample components
3. 🚀 **Use Phase 4** - Main interface for component generation
4. 📈 **Optional: Retrain** - With your new dataset for better results

---

## Questions?

Refer to: [CODE_UPDATE_SUMMARY.md](CODE_UPDATE_SUMMARY.md) for detailed information on all changes.

**Status**: ✅ Ready to use!
