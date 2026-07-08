# Update Plan: Align Code Generation to React Component Dataset Format

## Current State Analysis

### Dataset Format (reactjs_projects_dataset.jsonl)
The dataset contains simple React components with this structure:
```json
{
  "prompt": "Generate a React component for a login form",
  "code": "import React, { useState } from 'react';\n...",
  "Complex_CoT": "The prompt asks for a login form... [explanation]"
}
```

**Key Characteristics:**
- Single React component per entry (not full projects)
- Self-contained, functional components
- Includes reasoning/Chain-of-Thought explanation
- ~10 lines to 30+ lines of code per component

### Current System Structure (Problem)
The existing system is designed for **full-stack projects**:
- Generates multiple files (backend, frontend, config)
- Creates Express servers, SQL schemas, PHP pages
- Complex directory hierarchy (outputs/timestamp/backend|frontend|config)
- Output format doesn't match the simple component dataset

**Result:** The model is trained on complex multi-file projects but the reference dataset shows simple components, causing format mismatch.

---

## Solution Overview

Adapt the pipeline to work with **simple React components** instead of full-stack projects:

| Phase | Current | Target | Key Changes |
|-------|---------|--------|------------|
| **Phase 1 (Dataset)** | Multi-file projects | Single components | Generate one component per prompt; capture CoT |
| **Phase 2 (Training)** | Train on all file types | Train on components only | Simpler tokenization; focus on React syntax |
| **Phase 3 (Inference)** | Generate multi-file projects | Generate single components | Simpler prompt parsing; return component + CoT |
| **Phase 4 (Agent)** | Orchestrate projects | Generate components on demand | Simpler output structure |

---

## Detailed Update Plan

### **Phase 1: v2_dataset_creator.py**
**Goal:** Generate React components matching reactjs_projects_dataset.jsonl format

#### Changes:
1. **Simplify component generation** instead of project generation
   - Remove file_specs logic (no backend/config/database files)
   - Focus only on React components
   - Input: Single prompt (e.g., "Generate a React counter component")
   - Output: Single component code + CoT explanation

2. **Capture Chain-of-Thought reasoning**
   - Modify Ollama prompt to include reasoning
   - Extract explanation: "Explain how this component works"
   - Store in dataset: `{"prompt": "...", "code": "...", "Complex_CoT": "..."}`

3. **Dataset format adjustment**
   - Change from multi-file structure to simple JSONL
   - Exactly match reactjs_projects_dataset.jsonl schema
   - Remove project names, technologies, descriptions

4. **Update validation**
   - Check for React import/export
   - Verify component structure (function or class)
   - Validate JSX syntax

#### Sample Changes:
```python
# Before: Multi-file generation
files = generate_multi_file_project(description, technologies)  # -> 5+ files

# After: Single component generation
component = generate_single_react_component(prompt)  # -> 1 component + CoT
```

**Expected Dataset Entry:**
```json
{
  "prompt": "Create a React form with validation",
  "code": "import React, { useState } from 'react';\n...",
  "Complex_CoT": "I start by creating a component with state for form data..."
}
```

---

### **Phase 2: v2_train_model.py**
**Goal:** Train on React component dataset (simpler training data)

#### Changes:
1. **Adapt dataset loading**
   - Read: `{"prompt": "...", "code": "...", "Complex_CoT": "..."}`
   - Only use `code` field for training (same as before)
   - Dataset is much smaller and simpler

2. **No other model changes needed**
   - Architecture stays the same (TinyCodeModel)
   - Tokenization stays the same (cl100k_base)
   - Training loop stays the same
   - But dataset is cleaner (only React, not mixed languages)

3. **Expected improvements**
   - Faster convergence (simpler, consistent code)
   - Better React component generation
   - Smaller model footprint (no need to handle backend code)

#### Code Impact:
```python
# In load_and_tokenize()
# Currently reads: {"filename": "...", "code": "..."}
# Change to:      {"prompt": "...", "code": "...", "Complex_CoT": "..."}
# Extract only the 'code' field (same as before)
```

---

### **Phase 3: v2_inference.py**
**Goal:** Generate single React components with explanations

#### Changes:
1. **Simplify prompt parsing**
   - Remove complex backend/page/database detection
   - No need to parse for multi-file structure
   - Just accept: "Generate a React [component type]"

2. **Update code generation**
   - Generate single component (not multi-file)
   - Limit tokens (components are 200-400 tokens typically)
   - Add temperature/top_k tuning for React code

3. **Add CoT generation**
   - After generating component code, generate explanation
   - Prompt: "Explain how this React component works"
   - Store both code + explanation

4. **Simplify streaming**
   - Still support token-by-token streaming (optional)
   - Return: `{"code": "...", "cot": "...", "tokens_used": N}`

#### Code Pattern:
```python
# Before:
spec = generator.parse_user_prompt("Create a site with...")
component = generator.generate_component("react", "App.jsx", spec)  # Complex

# After:
code = generator.generate_component("Generate a React login form")
cot = generator.generate_cot(code)  # Explain the component
return {"code": code, "complex_cot": cot}
```

---

### **Phase 4: micro_ai_coder_agent.py**
**Goal:** Simplify to single-component generation

#### Changes:
1. **Simplify agent logic**
   - Remove `classify_backend_and_pages()` (no longer needed)
   - Remove directory structure creation (no multi-file)
   - Direct component generation

2. **Update output structure**
   - Instead of: `outputs/timestamp/backend|frontend|config/`
   - Use: `outputs/timestamp/component.jsx` + `component.cot.md`
   - Or: `outputs/timestamp/metadata.json` with embedded code/cot

3. **Streamline the main loop**
   - Input: Natural language prompt for a React component
   - Process: Send to inference engine
   - Output: Single component file + explanation file

#### Output Example:
```
outputs/
  20260708_120000/
    ├── metadata.json          # {"prompt": "...", "timestamp": "..."}
    ├── component.jsx          # The generated React code
    └── explanation.md         # The Chain-of-Thought reasoning
```

---

## Implementation Order (Recommended)

1. **Start with Phase 1** (v2_dataset_creator.py)
   - Make it generate single components matching the reactjs dataset schema
   - Verify output matches reactjs_projects_dataset.jsonl format
   - Test with 5-10 manual prompts

2. **Then Phase 3** (v2_inference.py)
   - Update to work with simpler component generation
   - Don't need Phase 2 yet (can reuse existing trained model)
   - Test component generation on sample prompts

3. **Then Phase 4** (micro_ai_coder_agent.py)
   - Restructure agent to use simplified Phase 3
   - Test end-to-end component generation

4. **Finally Phase 2** (v2_train_model.py)
   - Retrain on new dataset format from Phase 1
   - Optional: Only if you want better performance

---

## Key Differences Summary

| Aspect | Current | Target |
|--------|---------|--------|
| **Component Type** | Multi-file projects | Single React components |
| **Files per generation** | 5-7 files | 1 component |
| **Output folder** | backend/frontend/config | Flat: component.jsx + .md |
| **Dataset schema** | Multi-field complex | Simple: {prompt, code, Complex_CoT} |
| **Prompt parsing** | Detect backend/pages | Simple component request |
| **Model reuse** | Needs retraining | Can reuse (if training on consistent data) |
| **Inference complexity** | High (orchestrate multi-file) | Low (single generation) |

---

## Questions for Clarification

Before implementation, please clarify:

1. **Data Source**: Should Phase 1 generate new training data, or should it convert existing data?
   - Generate from scratch using Ollama? 
   - Or process/enhance the existing reactjs_projects_dataset.jsonl?

2. **Scope**: Do you want to support only React, or also other frameworks?
   - React-only (as shown in dataset)?
   - Other: Vue, Angular, Svelte?

3. **Component Types**: Should it support specific component types?
   - Free-form (any React component)?
   - Specific categories (Form, List, Counter, etc.)?

4. **CoT Generation**: How should explanations be generated?
   - Via Ollama (separate prompt after code)?
   - Extracted from code via regex/heuristics?
   - Predefined templates?

5. **Output Format**: Final component + CoT storage preference?
   - Separate files (.jsx + .md)?
   - Single JSONL entry (like training data)?
   - Both options?

