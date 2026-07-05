# Micro AI Coder Agent 🤖

A complete 4-phase system for training and deploying a specialized AI code generator for React + full-stack applications.

## Overview

This system generates high-quality training datasets, trains a compact PyTorch model with token-based tokenization (tiktoken), and provides streaming inference for multi-file React + Express/Node.js/MySQL/WordPress code generation.

```
User Prompt
    ↓
[Phase 1: Dataset Generator] → Generate training data from Ollama
    ↓
[Phase 2: Model Training] → Train PyTorch model with tiktoken
    ↓
[Phase 3: Inference Engine] → Stream tokens in real-time
    ↓
[Phase 4: Agent Orchestrator] → Multi-file project generation
    ↓
Generated Code Structure
```

## Quick Start

### Prerequisites
- Python 3.10+
- Local Ollama server running on `http://localhost:11434`
- Model: `qwen2.5-coder:3b` loaded in Ollama
- 8+ GB RAM (16+ recommended)
- GPU optional (CPU supported)

### Installation

```bash
# Clone/navigate to workspace
cd /Users/muralidharanramasamy/Micro_AI_coder

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import tiktoken; print('✅ Ready to go!')"
```

### Phase 1: Generate Dataset

```bash
python phase1_dataset_creator/v2_dataset_creator.py
```

**Interactive prompts:**
- Enter project description (e.g., "Social media feed app")
- Select technologies (e.g., "react, node, express, mysql")
- Repeat 10–20 times to build dataset

**Output:**
- `datasets/generated_projects.jsonl` (training format)
- `datasets/generated_projects.txt` (human-readable)

### Phase 2: Train Model

```bash
python phase2_training/v2_train_model.py
```

**Expected output:**
- `models/tiny_code_model.pt` (~100–150 MB)
- `models/model_config.json` (hyperparameters)
- `logs/training_log.jsonl` (metrics)

**Training time:** 30–60 mins on M-series Mac (CPU), 5–10 mins on GPU

### Phase 3: Test Inference

```bash
python phase3_inference/v2_inference.py
```

**Interactive menu:**
1. Single file generation
2. Multi-file project generation
3. Streaming demo
4. Prompt parsing test
5. Exit

**Test prompts:**
- "create homepage and contact page"
- "create node.js express api"
- "create database schema for user registration"

### Phase 4: Run Agent

```bash
python phase4_agent/micro_ai_coder_agent.py
```

**Example prompt:**
```
Create a React + Express full-stack app with homepage, about, contact, 
register, and login pages. Use MySQL backend.
```

**Output:** Complete project structure with streaming code generation

## Architecture

### Phase 1: Dataset Creator (`v2_dataset_creator.py`)
- Interactive CLI for project description + tech stack selection
- Multi-file generation (React, Express, SQL schemas)
- Ollama integration with retry logic
- Dual output format (JSONL for training, TXT for review)
- Summary statistics (project count, token estimate)

### Phase 2: Model Training (`v2_train_model.py`)
- PyTorch transformer with tiktoken (cl100k_base) tokenization
- Architecture: n_embd=256, n_head=8, n_layer=8, block_size=512
- Training: AdamW optimizer, cosine annealing, gradient clipping
- Validation: Every 500 iterations with best model checkpoint
- Output: Model weights + config + training metrics

### Phase 3: Inference Engine (`v2_inference.py`)
- `CodeGenerator` class with streaming token generation
- Prompt parsing for multi-file extraction (page names, backend type)
- Token-by-token streaming for real-time display
- Interactive CLI for testing

### Phase 4: Agent Orchestrator (`micro_ai_coder_agent.py`)
- `MicroAICoderAgent` class wrapping inference engine
- Backend classification (Express/WordPress/MySQL detection)
- Organized output: frontend/ + backend/ + config/
- Metadata tracking (timestamp, prompt, file count)

## File Structure After Execution

```
outputs/
├── {timestamp}/
│   ├── frontend/
│   │   ├── HomePage.jsx
│   │   ├── AboutPage.jsx
│   │   ├── ContactPage.jsx
│   │   ├── RegisterPage.jsx
│   │   ├── LoginPage.jsx
│   │   └── App.jsx
│   ├── backend/
│   │   ├── server.js
│   │   ├── routes.js
│   │   ├── auth.js
│   │   ├── middleware.js
│   │   └── schema.sql
│   ├── config/
│   │   └── .env.example
│   └── metadata.json
└── manifest.json
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Ollama not found** | Check Ollama running: `curl http://localhost:11434/api/tags` |
| **CUDA out of memory** | Reduce batch_size or use CPU |
| **Empty dataset** | Run Phase 1 multiple times (10+ projects) |
| **Model not found** | Verify Phase 2 completed; check `models/` directory |
| **Truncated code** | Increase max_tokens in inference config |

## Performance Notes

### M-series Mac (CPU)
- Phase 1: ~3–5 mins per project (depends on Ollama)
- Phase 2: ~45 mins for 20 projects (100k tokens)
- Phase 3: ~1–2 secs per file generation
- Phase 4: Real-time streaming

### GPU (NVIDIA CUDA)
- Phase 2: ~5–10 mins for 20 projects
- Phase 3–4: Near-instant generation

## Model Architecture Details

**Parameters:**
- Embedding dim: 256
- Attention heads: 8
- Layers: 8
- Context window: 512 tokens (~180 lines of code)
- Vocab size: 100,257 (tiktoken cl100k_base)
- Total params: ~50M

**Tokenization:**
- Library: tiktoken
- Encoding: cl100k_base
- Special tokens: `<|file_end|>` for file boundaries

**Training:**
- Optimizer: AdamW (lr=5e-4)
- Schedule: CosineAnnealingLR
- Batch size: 16
- Max iterations: 5000
- Gradient clipping: norm=1.0

## Expected Output Quality

### React Components
✅ Valid JSX syntax  
✅ React hooks (useState, useEffect, useContext)  
✅ Error handling & comments  
✅ 50–100 lines per component  

### Express.js Routes
✅ Proper async/await  
✅ Middleware integration  
✅ Route definitions  
✅ Error handling  

### SQL Schemas
✅ Valid DDL with tables  
✅ Foreign keys & indexes  
✅ Constraints & sample data  
✅ Comments explaining structure  

## Next Steps

1. ✅ Install dependencies
2. ⏳ Generate dataset (Phase 1)
3. ⏳ Train model (Phase 2)
4. ⏳ Test inference (Phase 3)
5. ⏳ Run agent (Phase 4)

## Support

For issues or questions:
1. Check PLAN.md for detailed architecture
2. Review logs in `logs/training_log.jsonl`
3. Test components individually (Phase 3 has interactive menu)

---

**Happy coding with Micro AI Coder! 🚀**
