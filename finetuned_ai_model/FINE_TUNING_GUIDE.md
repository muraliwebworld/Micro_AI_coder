# Fine-Tuning Guide: Step-by-Step Instructions

This guide walks you through the complete process of fine-tuning Qwen2.5-Coder on your React dataset.

## Phase 1: Setup (Google Colab) - 30 minutes

### Step 1: Open Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com)
2. Click "File" → "Open Notebook"
3. Upload the notebook from `colab_notebooks/fine_tune_qwen_colab.ipynb`
   - Or create a new notebook and copy-paste the cells

### Step 2: Enable GPU
1. Click "Runtime" → "Change runtime type"
2. Select **GPU** as Hardware accelerator
3. Choose **T4** (free tier, 16GB VRAM)
4. Click "Save"

### Step 3: Mount Google Drive
Run the first cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```
- Authorize access to Google Drive
- Your Drive will be accessible at `/content/drive`

### Step 4: Install Dependencies
Run the second cell to install required packages:
```python
!pip install transformers peft bitsandbytes accelerate torch datasets
```
This takes 3-5 minutes.

### Step 5: Verify GPU
Run this to check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```
Expected output: `CUDA available: True`, `GPU: Tesla T4`, `VRAM: 16.0 GB`

---

## Phase 2: Prepare Data (5 minutes)

### Step 6: Upload Dataset
Upload `data_cleaned_huggingface_new.jsonl` to Google Drive:
1. Open [Google Drive](https://drive.google.com)
2. Create folder: `micro_ai_coder_data`
3. Upload the JSONL file there

### Step 7: Load & Verify Dataset
In Colab, run:
```python
import json
dataset_path = "/content/drive/My Drive/micro_ai_coder_data/data_cleaned_huggingface_new.jsonl"

# Count samples
with open(dataset_path) as f:
    samples = [json.loads(line) for line in f if line.strip()]
print(f"✅ Loaded {len(samples)} samples")

# Check format
sample = samples[0]
print(f"Keys: {sample.keys()}")
print(f"Prompt: {sample['prompt'][:50]}...")
print(f"Code length: {len(sample['code'])} chars")
```

Expected output:
```
✅ Loaded 39000 samples
Keys: dict_keys(['prompt', 'code', 'Complex_CoT'])
Prompt: Generate a React component for a login form...
Code length: 1250 chars
```

---

## Phase 3: Fine-Tuning (2-3 hours)

### Step 8: Load Base Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import bitsandbytes

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load in 8-bit to save memory
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
```
This takes 2-3 minutes to download (7.6 GB).

### Step 9: Setup LoRA Adapters
```python
# Configure LoRA
lora_config = LoraConfig(
    r=64,                          # LoRA rank
    lora_alpha=16,                 # LoRA alpha
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())
```

Expected output:
```
trainable params: 20,971,520 || all params: 7,090,909,184 || trainable%: 0.30
```

### Step 10: Prepare Datasets
```python
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

# Load data
with open(dataset_path) as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

# Create training examples
examples = []
for item in raw_data:
    prompt = item['prompt']
    code = item['code']
    # Format: "Prompt: {prompt}\n\nCode:\n{code}"
    text = f"Prompt: {prompt}\n\nCode:\n{code}"
    examples.append({"text": text})

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

dataset = Dataset.from_dict({"text": [e["text"] for e in examples]})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split: 80% train, 10% val, 10% test
train_size = int(0.8 * len(tokenized_dataset))
val_size = int(0.1 * len(tokenized_dataset))

train_dataset = tokenized_dataset[:train_size]
val_dataset = tokenized_dataset[train_size:train_size + val_size]
test_dataset = tokenized_dataset[train_size + val_size:]

print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
```

### Step 11: Configure Training
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/content/drive/My Drive/qwen_reactjs_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    logging_dir="/content/logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-4,
    bf16=False,  # Use fp16 instead
    fp16=True,
    gradient_checkpointing=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
```

### Step 12: Start Training
```python
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete!")
```

**Monitoring**:
- Watch the loss curves in Colab's output
- Training loss should decrease monotonically
- Validation loss should converge < 1.2
- Expected duration: 2-3 hours

**Expected Output**:
```
[Epoch 1/3] | train_loss: 1.8234 | val_loss: 1.2156
[Epoch 2/3] | train_loss: 0.9123 | val_loss: 0.9876
[Epoch 3/3] | train_loss: 0.6234 | val_loss: 0.8765
✅ Training complete!
```

---

## Phase 4: Save Model (10 minutes)

### Step 13: Merge LoRA Adapters
```python
# Load best model from checkpoint
model = trainer.model

# Merge LoRA weights into base model
if hasattr(model, 'merge_and_unload'):
    merged_model = model.merge_and_unload()
else:
    merged_model = model

print("✅ LoRA adapters merged")
```

### Step 14: Save to Google Drive
```python
output_dir = "/content/drive/My Drive/qwen_reactjs_merged"
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Merged model saved to {output_dir}")
print(f"   Model: ~7.6 GB (fp16)")
print(f"   Download to local machine when complete")
```

---

## Phase 5: Local Setup (45 minutes)

### Step 15: Download from Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Find `qwen_reactjs_merged/` folder
3. Download to: `finetuned_ai_model/models/qwen_reactjs_merged/`

### Step 16: Install llama-cpp-python
```bash
pip install llama-cpp-python
```

### Step 17: Convert to GGUF
On your local machine, run:
```bash
python -m llama_cpp.server \
  --model_path models/qwen_reactjs_merged/pytorch_model.bin \
  --n_ctx 1024
```

Or use conversion script (see below).

### Step 18: Create Ollama Modelfile
Create `models/Modelfile_qwen_reactjs`:
```
FROM models/qwen_reactjs_merged.Q4_K_M.gguf

TEMPLATE """{{ .Prompt }}"""

SYSTEM "You are a helpful React code generation assistant. Generate complete, syntactically valid React components based on user requirements."

PARAMETER temperature 0.3
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_predict 512
```

### Step 19: Register with Ollama
```bash
ollama create micro-ai-coder-v2:latest -f models/Modelfile_qwen_reactjs
```

Verify:
```bash
ollama list | grep micro-ai-coder-v2
```

Expected output:
```
micro-ai-coder-v2:latest     7e9d1a2f1b4c     7.3 GB     0 minutes ago
```

---

## Phase 6: Testing (30 minutes)

### Step 20: Run Validation Tests
```bash
python tests/validate_finetuned_qwen.py
```

Expected output:
```
✅ Test 1/20: Login form component... PASS
✅ Test 2/20: Counter component... PASS
✅ Test 3/20: Todo list... PASS
...
✅ All 20 tests PASSED (100% success rate)
```

### Step 21: Manual Testing
```bash
ollama run micro-ai-coder-v2:latest "Write a React button component with onClick handler"
```

Should generate valid React code like:
```javascript
import React from 'react';

export default function Button() {
  const handleClick = () => {
    console.log('Button clicked');
  };

  return (
    <button onClick={handleClick}>
      Click me
    </button>
  );
}
```

---

## Phase 7: Deploy (1 hour)

### Step 22: Update Phase 3 Inference
Copy `phase3_inference/v3_inference_finetuned.py` to main project:
```bash
cp phase3_inference/v3_inference_finetuned.py ../phase3_inference/
```

### Step 23: Update Phase 4 Agent
Copy `phase4_agent/micro_ai_coder_agent_v2.py` to main project:
```bash
cp phase4_agent/micro_ai_coder_agent_v2.py ../phase4_agent/
```

### Step 24: Update Model Name
In both files, verify MODEL_NAME is set to:
```python
MODEL_NAME = "micro-ai-coder-v2:latest"
```

### Step 25: Run Full Pipeline Test
```bash
python ../phase4_agent/micro_ai_coder_agent_v2.py
```

Select option to generate a React component and verify output is valid.

---

## 📊 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size
```python
per_device_train_batch_size=2,
gradient_accumulation_steps=4,  # Increase to maintain effective batch size
```

### Issue: Loss not decreasing
**Solution**: Lower learning rate
```python
learning_rate=1e-4,  # Instead of 2e-4
```

### Issue: Validation loss plateaus early
**Solution**: This is normal, model has converged. Can stop after epoch 2.

### Issue: Model generates nonsense after conversion
**Solution**: Check GGUF quantization level. Try Q5_K_M instead of Q4_K_M:
```
FROM models/qwen_reactjs_merged.Q5_K_M.gguf
```

### Issue: Ollama can't find model
**Solution**: Verify model file exists and Ollama modelfile is correct:
```bash
ls -lh models/qwen_reactjs_merged.Q4_K_M.gguf
ollama list
```

---

## ✅ Success Checklist

- [ ] Google Colab notebook runs without errors
- [ ] GPU detected (T4, 16GB VRAM)
- [ ] Dataset loaded (39K samples)
- [ ] Model downloaded (~7.6 GB)
- [ ] Training starts (loss decreasing)
- [ ] Training completes (val_loss < 1.2)
- [ ] Model saved to Google Drive
- [ ] Model downloaded locally (~4 GB)
- [ ] GGUF conversion succeeds
- [ ] Ollama model registered
- [ ] 20/20 validation tests pass
- [ ] Manual tests generate valid code
- [ ] Phase 3/4 updated and tested
- [ ] Full pipeline works end-to-end

---

**Estimated Total Time**: 4-5 hours  
**Status**: Ready to implement  
**Last Updated**: 2026-07-09
