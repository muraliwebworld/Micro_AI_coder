# MICRO AI CODER AGENT - COMPLETE IMPLEMENTATION PLAN

**Version:** 1.0  
**Created:** 2026-07-05  
**Status:** Implementation In Progress  
**Workspace:** `/Users/muralidharanramasamy/Micro_AI_coder/`

---

## 📋 EXECUTIVE SUMMARY

Build a **4-phase AI code generation system** that:
- **Phase 1:** Generates diverse full-stack training datasets from user prompts via local Ollama (qwen2.5-coder:3b)
- **Phase 2:** Trains a compact token-based PyTorch model using tiktoken tokenization
- **Phase 3:** Provides streaming inference to generate multi-file React + Express/Node.js/MySQL/WordPress projects
- **Phase 4:** Integrates all pieces into a unified agent accepting user prompts and generating separate component/file code with real-time token streaming

**Total Effort:** ~5–7 hours | **Timeline:** Sequential phases with parallel discovery

---

## 🎯 KEY DECISIONS

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Scope** | Build from scratch, clean implementations | Leverage lessons from existing code; avoid technical debt |
| **Backend Stack** | Express.js + Node.js, MySQL, WordPress (user selects) | Maximum flexibility; covers most use cases |
| **Tokenization** | Token-based (tiktoken cl100k_base) | 100k+ vocab; better code precision than character-level |
| **Streaming** | Token-by-token streaming | Real-time UX; user sees generation progress |
| **Multi-file** | Parse prompts → generate separate component files | Realistic multi-page React + backend structure |
| **Model Size** | n_layer=8, n_embd=256, block_size=512 | ~50M params; balance speed/quality on M-series Mac |

---

## 📁 WORKSPACE STRUCTURE

```
/Users/muralidharanramasamy/Micro_AI_coder/
│
├── PLAN.md                                    # Master reference
├── README.md                                  # Quick start guide
├── requirements.txt                           # Python dependencies
│
├── phase1_dataset_creator/
│   └── v2_dataset_creator.py                  # Dataset generation (interactive CLI)
│
├── phase2_training/
│   └── v2_train_model.py                      # PyTorch training with tiktoken
│
├── phase3_inference/
│   ├── v2_inference.py                        # Streaming inference + testing
│   └── test_inference.py                      # Test suite for Phase 3
│
├── phase4_agent/
│   └── micro_ai_coder_agent.py                # Main agent orchestrator
│
├── datasets/
│   ├── generated_projects.jsonl               # Auto-created by Phase 1 (training data)
│   └── generated_projects.txt                 # Auto-created by Phase 1 (human-readable)
│
├── outputs/
│   └── {timestamp}/                           # Auto-created by Phase 3 (generated code)
│
├── models/
│   ├── tiny_code_model.pt                     # Auto-created by Phase 2 (trained weights)
│   └── model_config.json                      # Auto-created by Phase 2 (hyperparams)
│
└── logs/
    └── training_log.jsonl                     # Auto-created by Phase 2 (metrics)
```

---

## 🔄 IMPLEMENTATION PHASES

### Phase 1: Dataset Creator
- **Input:** User project description, tech stack selection
- **Output:** datasets/generated_projects.jsonl, datasets/generated_projects.txt
- **Time:** ~1–2 hours (depends on Ollama generation speed)
- **Status:** [In Progress]

### Phase 2: Model Training
- **Input:** datasets/generated_projects.jsonl
- **Output:** models/tiny_code_model.pt, models/model_config.json
- **Time:** ~30–60 mins (depends on dataset size + hardware)
- **Status:** [Pending Phase 1]

### Phase 3: Inference & Testing
- **Input:** Trained model + tokenizer
- **Output:** Generated code files + streaming output
- **Time:** ~1–2 hours (implementation + testing)
- **Status:** [Pending Phase 2]

### Phase 4: Agent Orchestration
- **Input:** User natural language prompt
- **Output:** Complete multi-file project structure with streaming
- **Time:** ~1 hour (integration)
- **Status:** [Pending Phase 3]

---

## 🚀 QUICK START

### 1. Install Dependencies
```bash
cd /Users/muralidharanramasamy/Micro_AI_coder
pip install -r requirements.txt
```

### 2. Phase 1: Generate Dataset
```bash
python phase1_dataset_creator/v2_dataset_creator.py

# Follow prompts:
# - Project description: "Social media feed app"
# - Technologies: "react, node, express, mysql"
# - Repeat 10–20 times for good dataset
```

### 3. Phase 2: Train Model
```bash
python phase2_training/v2_train_model.py
# Wait for training to complete (~30–60 mins)
```

### 4. Phase 3: Test Inference
```bash
python phase3_inference/v2_inference.py
# Interactive menu to test code generation
```

### 5. Phase 4: Run Agent
```bash
python phase4_agent/micro_ai_coder_agent.py
# Natural language prompt → multi-file code generation
```

---

## ✅ VERIFICATION CHECKLIST

- [ ] Phase 1: datasets/generated_projects.jsonl created with ≥10 valid entries
- [ ] Phase 2: models/tiny_code_model.pt created (~100–150 MB)
- [ ] Phase 3: Code generation produces valid React/Node.js/SQL syntax
- [ ] Phase 4: Agent accepts prompts and generates organized output structure

---

## 📝 DEPENDENCIES

- Python 3.10+
- Local Ollama server with qwen2.5-coder:3b
- PyTorch 2.2.0
- tiktoken 0.5.2
- requests 2.31.0
- numpy 1.24.3

---

**Ready to proceed with implementation?**
