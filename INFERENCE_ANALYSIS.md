# Inference Analysis & Improvement Plan

## Current Status

### ✅ Working
- **Data Quality**: 100% valid React components in training dataset
- **Model Training**: Completed with early stopping at 3,100 iterations (best val_loss: 2.4990)
- **Inference Framework**: Properly set up with fallback templates
- **Generated Output**: Always produces valid React components (via fallback)

### ❌ Issues Found

1. **Model Generation Produces Corrupted Code**
   - Generated code has unbalanced parentheses/braces/brackets
   - Falls back to template instead of using generated code
   - Example errors: `import React.FC = () => {` (invalid syntax)

2. **Root Cause Analysis**
   ```
   ✅ Data: Valid (100% balanced, proper imports/exports)
   ✅ Training: Completed (val_loss: 2.50)
   ❌ Generation: Produces tokens that don't form valid code
   ```

3. **Why Validation Loss (2.5) is Still Too High**
   - For valid code generation, target val_loss should be < 1.5
   - Current loss suggests model doesn't understand code structure well
   - Model generates plausible token sequences but syntactically invalid

## Current System Architecture

```
Phase 2: Training
  Input: 6,529 React components (data_cleaned_huggingface.jsonl)
  Model: TinyCodeModel (27.27M params, 8 layers, 128 embedding)
  Output: tiny_code_model.pt (val_loss: 2.50)

Phase 3: Inference
  Input: User prompt
  Step 1: Generate code using model
  Step 2: Validate generated code
  Step 3: If invalid → Use fallback template
  Output: LoginForm template (not AI-generated)
```

## Improvements Needed

### High Priority
1. **Increase Model Capacity**
   - Current: 27.27M params
   - Recommended: 100M+ params (larger embeddings, more layers)
   - This allows better code understanding

2. **Longer Training**
   - Current: 3,100 iterations (early stopped)
   - Recommended: 10,000-20,000 iterations with validation monitoring
   - Watch for divergence and use checkpoints

3. **Better Hyperparameters**
   - Lower learning rate: 1e-4 (instead of 3e-4)
   - Higher dropout: 0.25 (instead of 0.2)
   - Longer warmup: 2000 steps (instead of 500)

### Medium Priority
4. **Improved Prompt Engineering**
   - Current: `import React, { useState } from 'react'; const Component = () => {`
   - Add examples/few-shot learning in prompt
   - Include component type hints

5. **Better Generation Strategy**
   - Use beam search instead of sampling
   - Implement constrained decoding (force balanced brackets)
   - Add post-processing cleanup

6. **Data Augmentation**
   - Current: 6,529 samples
   - Recommended: 20,000+ samples
   - Add more component types (hooks, context, etc.)

### Low Priority
7. **Model Architecture**
   - Add position-wise FFN improvements
   - Use Flash Attention for better efficiency
   - Add layer normalization variants

## Quick Wins (Implement First)

### 1. Fix Inference Temperature
Current: `temperature=0.4, top_k=30`
Try: `temperature=0.1, top_k=15` (more conservative)

### 2. Add Constrained Decoding
```python
def generate_with_constraints(prompt):
    # Force balanced brackets by tracking state
    open_braces = 0
    while open_braces > 0 or generation_incomplete:
        sample_next_token()
        track_braces()
```

### 3. Improve Fallback Selection
- Map prompts to specific component types
- Keep library of common patterns
- Use semantic search for similar examples

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Validation Loss | 2.50 | < 1.5 |
| % Valid Generation (no fallback) | 0% | > 80% |
| Model Parameters | 27M | 100M+ |
| Training Iterations | 3,100 | 15,000 |
| Longest Code Generated | 600 tokens | 1,500 tokens |

## Recommended Next Steps

### Week 1
1. ✅ Train larger model (100M params) for 15,000 iterations
2. ✅ Implement constrained decoding in inference
3. ✅ Create component pattern library (50+ examples)

### Week 2
1. ✅ Evaluate generation quality on 100 test prompts
2. ✅ Fine-tune generation hyperparameters
3. ✅ Add semantic similarity to fallback selection

### Week 3
1. ✅ Add few-shot learning to prompts
2. ✅ Implement beam search generation
3. ✅ Create comprehensive evaluation metrics

## Testing Strategy

```python
# Test generation quality
test_prompts = [
    "Create a login form",
    "Build a todo list component",
    "Make a navbar with dropdown menu",
    "Create a card component",
    "Build a pagination component"
]

for prompt in test_prompts:
    code = generate(prompt)
    if validate(code):
        print(f"✅ {prompt}: Generated valid code")
    else:
        print(f"❌ {prompt}: Used fallback template")
```

## Files Modified
- `phase2_training/v2_train_model.py`: Added early stopping, removed final model overwrite
- `phase3_inference/v2_inference.py`: Improved generation parameters, better prompt engineering

## Conclusion

The system currently **works** (produces valid components via fallback), but **doesn't truly generate code** yet. 
The model understands patterns but isn't yet capable of generating syntactically valid React code.

**Priority**: Increase model capacity and training duration to achieve > 80% direct generation success rate.
