# Ollama Deployment Guide - M2 Mac

## Deploy Fine-Tuned Model to Ollama

### Prerequisites
- Fine-tuned model merged: `models/qwen_reactjs_merged/`
- GGUF file created: `models/qwen_reactjs_merged.Q4_K_M.gguf`
- Ollama installed on M2 Mac
- Modelfile created: `models/Modelfile`

---

## Step 1: Install Ollama

### 1.1 Download Ollama
```bash
# Download from https://ollama.ai
# Or via Homebrew:
brew install ollama

# Verify installation
ollama --version
# Should show: version 0.X.X
```

### 1.2 Start Ollama Service
```bash
# In background
ollama serve &

# Or as a service (if installed via Homebrew)
brew services start ollama

# Verify running
curl http://localhost:11434/api/tags
# Should return JSON with available models
```

---

## Step 2: Prepare Modelfile

The Modelfile tells Ollama how to run your fine-tuned model.

### 2.1 Verify Modelfile
Located at: `models/Modelfile`

**Content should be:**
```
FROM ./models/qwen_reactjs_merged.Q4_K_M.gguf

SYSTEM """You are an expert React component generator. 
When given a prompt about a React component, generate clean, 
valid React code with proper imports, exports, and JSX syntax.
Focus on functionality and best practices.
Keep components concise and reusable."""

PARAMETER temperature 0.3
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_predict 256
```

### 2.2 Update Modelfile Path (if needed)
If your GGUF file is in a different location, update line 1:

```
FROM /path/to/qwen_reactjs_merged.Q4_K_M.gguf
```

---

## Step 3: Register Model with Ollama

### 3.1 Create Model
```bash
# From the project root directory
cd finetuned_ai_model_qwen-1-5-b

# Register model
ollama create micro-ai-coder-1-5b:latest -f models/Modelfile

# You should see:
# writing model
# removing any old model files
# ✅ success
```

**Note:** First time takes 2-3 minutes as Ollama processes the GGUF file.

### 3.2 Verify Model is Registered
```bash
# List all models
ollama list

# Should show:
# NAME                            ID              SIZE      MODIFIED
# micro-ai-coder-1-5b:latest     abc123...       850 MB    2 minutes ago
# llama2:latest                  xyz789...       3.8 GB    1 week ago
```

---

## Step 4: Test Model

### 4.1 Quick Test via CLI
```bash
# Single generation
ollama run micro-ai-coder-1-5b:latest "Write a React button component"

# Should output:
# Loading model...
# export const Button = ({ label, onClick }) => {
#   return <button onClick={onClick}>{label}</button>;
# };
```

### 4.2 Interactive Mode
```bash
# Start interactive chat
ollama run micro-ai-coder-1-5b:latest

# Type prompts:
>>> Write a React counter with state
>>> Create a React form component
>>> /exit to quit
```

### 4.3 Test via HTTP API
```bash
# Test API endpoint
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "micro-ai-coder-1-5b:latest",
    "prompt": "Write a React button",
    "stream": false,
    "temperature": 0.3
  }'

# Should return JSON with generated code
```

---

## Step 5: Performance Tuning

### 5.1 Memory Usage
```bash
# Monitor during inference
# In separate terminal:
watch -n 1 'top -n 1 -b | grep ollama'

# Expected on M2 with 8GB RAM:
# - Inference: 2-3 GB RAM
# - Headroom: 5-6 GB free (safe)
```

### 5.2 Inference Speed
```bash
# Time a single request
time curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "micro-ai-coder-1-5b:latest",
    "prompt": "Write a React component",
    "stream": false
  }' > /dev/null

# Expected: 10-30 seconds for 256 tokens (~20-40 tokens/sec)
```

### 5.3 Adjust Generation Parameters

Edit `models/Modelfile` to tune:

```
# Temperature: 0.0 = deterministic, 1.0 = random
PARAMETER temperature 0.3  # Lower = more consistent code

# Top-k: number of top-scoring tokens to consider
PARAMETER top_k 40  # Lower = more focused

# Top-p: nucleus sampling threshold
PARAMETER top_p 0.9  # Lower = more coherent

# Max tokens to generate
PARAMETER num_predict 256  # Can increase for longer components
```

After editing, recreate model:
```bash
ollama create micro-ai-coder-1-5b:latest -f models/Modelfile
```

---

## Step 6: Automated Inference

### 6.1 Run Inference Script
```bash
# From project root
python scripts/inference_finetuned.py

# Menu:
# 1. Single generation - generate one component
# 2. Batch generation - test prompts
# 3. Exit

# Select option and follow prompts
```

### 6.2 Batch Testing
```bash
# Run test suite (20 diverse prompts)
python tests/validate_finetuned_qwen.py

# Expected output:
# ✅ 17/20 tests passed (85%)
# Results saved to: outputs/test_results_*.md
```

---

## Step 7: Troubleshooting

### Issue 1: Model Not Found
```
Error: model 'micro-ai-coder-1-5b:latest' not found
```

**Solution:**
```bash
# Verify GGUF file exists
ls -lh models/qwen_reactjs_merged.Q4_K_M.gguf

# Recreate model
ollama create micro-ai-coder-1-5b:latest -f models/Modelfile

# Verify registration
ollama list | grep micro-ai-coder
```

### Issue 2: Ollama Connection Error
```
Error: Cannot connect to http://localhost:11434
```

**Solution:**
```bash
# Start Ollama service
ollama serve &

# Or check if already running
lsof -i :11434

# If stuck, kill and restart
killall ollama
sleep 2
ollama serve &
```

### Issue 3: Out of Memory During Inference
```
Error: not enough memory
```

**Solution:**
- Close other applications
- Reduce `num_predict` in Modelfile (256 → 128)
- Use smaller model or different quantization

### Issue 4: Very Slow Generation
```
> 60 seconds for 256 tokens
```

**Verify GPU is used:**
```bash
# Check if Metal is active
ps aux | grep ollama

# Should show GPU usage in Activity Monitor
# If CPU-only, will be slow on M2 (still viable at 5-10 tok/s)
```

### Issue 5: Ollama Won't Start
```
Error: bind: address already in use
```

**Solution:**
```bash
# Find and kill existing process
lsof -i :11434
kill -9 <PID>

# Or change port in Ollama (advanced)
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

---

## Step 8: Integration

### 8.1 Use in Python Code
```python
import requests

MODEL = "micro-ai-coder-1-5b:latest"
BASE_URL = "http://localhost:11434"

def generate_react(prompt):
    response = requests.post(
        f"{BASE_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": f"Prompt: {prompt}\n\nCode:\n",
            "stream": False,
            "temperature": 0.3,
        }
    )
    return response.json()["response"]

# Test
code = generate_react("Write a React button")
print(code)
```

### 8.2 Run in Background
```bash
# Start as background service (macOS)
launchctl load ~/Library/LaunchAgents/com.ollama.plist

# Or use nohup
nohup ollama serve > ollama.log 2>&1 &
```

### 8.3 Startup Script
Create `scripts/start_ollama.sh`:
```bash
#!/bin/bash
# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Verify model is available
ollama list | grep micro-ai-coder-1-5b
```

---

## Step 9: Monitoring & Maintenance

### 9.1 Check Model Status
```bash
# List all models and their status
ollama list

# Show model details
ollama show micro-ai-coder-1-5b:latest

# Check model file
du -sh ~/.ollama/models/manifests/registry.ollama.ai/library/micro-ai-coder-1-5b/
```

### 9.2 Update Model
If you fine-tune again:
```bash
# Remove old model
ollama rm micro-ai-coder-1-5b:latest

# Create new version
ollama create micro-ai-coder-1-5b:latest -f models/Modelfile
```

### 9.3 Logs
```bash
# Check Ollama logs (if using Homebrew service)
tail -f ~/.ollama/logs/*.log

# Or check console output if running in foreground
# Ctrl+C to stop
```

---

## Performance Benchmarks

### M2 Mac mini (8GB RAM)

| Metric | Value |
|--------|-------|
| Model size | 800 MB GGUF |
| Inference latency | 15-30 seconds/prompt |
| Generation speed | 20-40 tokens/sec |
| Memory (peak) | 3-4 GB |
| CPU usage | 40-60% (GPU accelerated) |
| Success rate | ≥85% valid React code |

### Comparison: 1.5B vs 7B Model
| Model | GGUF Size | Inference Speed | RAM | Success Rate |
|-------|-----------|-----------------|-----|--------------|
| 1.5B | 800 MB | 20-40 tok/s | 3-4 GB | ≥85% |
| 7B | 3.5 GB | 5-15 tok/s | 6-8 GB | ≥95% |

---

## Next Steps

1. **Inference**: Run `python scripts/inference_finetuned.py`
2. **Validation**: Run `python tests/validate_finetuned_qwen.py`
3. **Integration**: Use in your applications
4. **Monitoring**: Track performance and generation quality
5. **Improvement**: Plan retraining with more/better data

---

## Support

### Common Questions

**Q: Can I run this on a Mac with less than 8GB RAM?**
A: Possibly with 6GB using aggressive quantization (Q3_K_M), but not guaranteed.

**Q: How do I use this with my existing Micro AI Coder?**
A: Copy `scripts/inference_finetuned.py` to your phase3_inference/ module.

**Q: Can I fine-tune on other tech stacks (Node, PHP, etc.)?**
A: Yes! Change the dataset to focus on that language and retrain.

**Q: Is the model running on GPU or CPU?**
A: Check Activity Monitor GPU tab. Metal GPU (mps) will show M2 GPU usage.

