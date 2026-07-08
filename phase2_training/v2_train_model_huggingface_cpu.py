#!/usr/bin/env python3
"""
PHASE 2: MODEL TRAINING - CPU-OPTIMIZED FOR MAC MINI M2 (8GB RAM)
Lightweight training specifically designed for CPU-only machines with limited memory.

Optimizations for 8GB RAM:
- Batch size: 4 (instead of 16)
- Model size: 128 embeddings, 8 layers (instead of 256, 12)
- Block size: 96 (instead of 128)
- Reduced iterations: 4,000 (instead of 10,000)
- Memory-efficient data loading
- CPU-specific optimizations
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import tiktoken
from pathlib import Path
import numpy as np
from datetime import datetime
import gc
import os

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR 8GB RAM CPU
# ============================================================================

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent / "logs"
INPUT_JSONL = DATASETS_DIR / "data_uniques.jsonl"  # Using LLaMA-3.1-405B dataset (39,860 high-quality entries)
OUTPUT_MODEL = MODELS_DIR / "tiny_code_model_llama_cpu.pt"
OUTPUT_CONFIG = MODELS_DIR / "model_config_llama_cpu.json"
TRAINING_LOG = LOGS_DIR / "training_log_llama_cpu.jsonl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# CPU-OPTIMIZED HYPERPARAMETERS (AGGRESSIVE ANTI-OVERFITTING)
BATCH_SIZE = 4              # ⬇️  Even smaller (3 instead of 4) - tighter gradient updates
BLOCK_SIZE = 128             # ⬇️  Shorter sequences (was 96) - less context corruption
MAX_ITERS = 6000            # ⬇️  Half iterations (was 6000) - stop before overfitting
LEARNING_RATE = 5e-4        # ⬇️  Lower LR (was 1e-3) - slower, more stable learning
N_EMBD = 128                 # ⬇️  Smaller (was 128) - 25% smaller model
N_HEAD = 4                  # ⬇️  Fewer heads (was 4) - less capacity
N_LAYER = 8                 # ⬇️  Fewer layers (was 8) - lighter model
DROPOUT = 0.15              # ⬆️  Much higher dropout (was 0.1) - 50% more regularization
GRAD_CLIP = 0.3             # ⬇️  Tighter gradient clipping (was 0.5)
TOKENIZER_NAME = "cl100k_base"
DEVICE = 'cpu'

# CPU Training improvements (MAXIMUM ANTI-OVERFITTING)
EVAL_INTERVAL = 150         # ⬆️  Evaluate very frequently (every 250 steps)
WARMUP_STEPS = 400          # ⬆️  Longer warmup (was 300) - better stability
WEIGHT_DECAY = 0.02         # ⬆️  Double L2 regularization (was 0.01) - strong penalty
SAVE_INTERVAL = 500         # ⬇️  Save even less frequently
PATIENCE = 8                # ⬇️  Very early stopping (was 8) - stop immediately if plateau

# CPU-specific settings
NUM_THREADS = os.cpu_count() or 4  # Use all CPU cores
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(1)

print_header = lambda text: print("\n" + "="*80 + "\n" + text + "\n" + "="*80)
print_step = lambda text: print(f"\n➤ {text}")
print_success = lambda text: print(f"✅ {text}")
print_warning = lambda text: print(f"⚠️  {text}")

# ============================================================================
# MODEL ARCHITECTURE (LIGHTWEIGHT VERSION)
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),  # ⬇️  Smaller FFN (2x instead of 4x)
            nn.GELU(),
            nn.Linear(2 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

class TinyCodeModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.block_size, f"Sequence {T} exceeds block size {self.block_size}"
        
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = self.emb_dropout(tok_emb + pos_emb)

        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        return self.lm_head(x)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============================================================================
# DATA LOADING (MEMORY EFFICIENT)
# ============================================================================

def extract_code_field(entry):
    """Extract code from entry - handles both original and cleaned dataset formats"""
    
    # Cleaned dataset format: direct 'code' field
    if 'code' in entry and isinstance(entry['code'], str) and len(entry['code']) > 50:
        return entry['code']
    
    # Original dataset format: messages with code blocks
    if 'messages' in entry and isinstance(entry['messages'], list):
        code_blocks = []
        for msg in entry['messages']:
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if '```' in content:
                    parts = content.split('```')
                    for i in range(1, len(parts), 2):
                        code = parts[i].strip()
                        lines = code.split('\n')
                        if lines and lines[0].lower() in ['tsx', 'jsx', 'javascript', 'typescript', 'python', 'java', 'cpp', 'c', 'go', 'rust', 'php', 'html', 'css']:
                            code = '\n'.join(lines[1:])
                        if len(code) > 100:
                            code_blocks.append(code.strip())
        if code_blocks:
            return '\n'.join(code_blocks)
    
    # Fallback to other field names
    for field in ['response', 'solution', 'content', 'body', 'text', 'output']:
        if field in entry and isinstance(entry[field], str) and len(entry[field]) > 50:
            return entry[field]
    
    return None

def load_and_tokenize():
    print_step("Loading Hugging Face dataset (CPU-optimized)...")
    
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_JSONL}")
    
    code_samples = []
    skipped = 0
    line_num = 0
    
    # Progress tracking
    max_samples = 18000  # ⬇️ Focus on first 3000 (highest quality from cleaned dataset)
    
    try:
        with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                if line.strip():
                    try:
                        entry = json.loads(line)
                        code = extract_code_field(entry)
                        if code:
                            # Limit code length to 2000 chars for memory
                            if len(code) > 2000:
                                code = code[:2000]
                            code_samples.append(code)
                        else:
                            skipped += 1
                    except (json.JSONDecodeError, Exception):
                        skipped += 1
                
                # Stop after max_samples for memory efficiency
                if len(code_samples) >= max_samples:
                    print_warning(f"Reached {max_samples} samples limit (memory efficiency)")
                    break
                
                if line_num % 2000 == 0:
                    print(f"  Processed {line_num:,} lines... ({len(code_samples)} valid)")
    except KeyboardInterrupt:
        print_warning("Loading interrupted")
    
    print_success(f"Loaded {len(code_samples)} code samples")
    print_warning(f"Skipped {skipped} entries")
    
    if not code_samples:
        raise ValueError("No valid code samples found!")
    
    # Tokenize
    print_step("Tokenizing...")
    text = "\n<|file_end|>\n".join(code_samples)
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    
    print_success(f"Tokenized to {len(data):,} tokens")
    
    # Clear memory
    del text
    del code_samples
    gc.collect()
    
    return data, enc, enc.n_vocab

def get_batch(split, train_data, val_data):
    data_split = train_data if split == 'train' else val_data
    max_ix = max(1, len(data_split) - BLOCK_SIZE)
    ix = torch.randint(max_ix, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=3):  # ⬇️ Fewer eval iterations
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split, train_data, val_data)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out

def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step / WARMUP_STEPS)
    else:
        # Linear decay for CPU stability
        return LEARNING_RATE * (1.0 - step / MAX_ITERS)

# ============================================================================
# TRAINING (CPU-OPTIMIZED)
# ============================================================================

def train():
    print_header("🚀 TRAINING (CPU-OPTIMIZED FOR M2 MAC 8GB RAM)")
    
    print(f"CPU Threads: {NUM_THREADS}")
    print(f"Device: {DEVICE}")
    print(f"\n⚠️  INFO:")
    print(f"   - This is optimized for M2 Mac with 8GB RAM")
    print(f"   - Training will take 15-30 minutes (aggressive regularization)")
    print(f"   - Your Mac may use swap - this is normal")
    print(f"   - You can interrupt with Ctrl+C (checkpoints will be saved)")
    print(f"   - AGGRESSIVE anti-overfitting: Smaller model, higher dropout, lower LR, early stopping")
    
    data, enc, vocab_size = load_and_tokenize()
    
    # Split (80/20)
    n = int(0.8 * len(data))
    train_data, val_data = data[:n], data[n:]
    print_success(f"Train set: {len(train_data):,} tokens")
    print_success(f"Val set: {len(val_data):,} tokens")
    
    # Create model
    model = TinyCodeModel(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print_success(f"Model parameters: {total_params/1e3:.1f}K")
    
    # Optimizer (CPU-friendly)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_ITERS)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print_header("TRAINING PROGRESS")
    print(f"Total iterations: {MAX_ITERS}")
    print(f"Evaluation every {EVAL_INTERVAL} iterations\n")
    
    start_time = datetime.now()
    
    try:
        for iter in range(MAX_ITERS):
            # Update learning rate
            lr = get_lr(iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            x, y = get_batch('train', train_data, val_data)
            logits = model(x)
            train_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            # Periodic evaluation
            if (iter + 1) % EVAL_INTERVAL == 0:
                losses = estimate_loss(model, train_data, val_data, eval_iters=3)
                val_loss = losses['val']
                
                # Log
                log_entry = {
                    'iteration': iter + 1,
                    'train_loss': round(losses['train'], 4),
                    'val_loss': round(val_loss, 4),
                    'learning_rate': f"{lr:.2e}",
                    'timestamp': datetime.now().isoformat()
                }
                with open(TRAINING_LOG, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')
                
                # Display
                improvement = "↓" if val_loss < best_val_loss else "↑"
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                print(f"Iter {iter+1:5d}/{MAX_ITERS} | train: {losses['train']:.4f} | val: {val_loss:.4f} {improvement} | time: {elapsed:.1f}m")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), OUTPUT_MODEL)
                    print(f"           ✅ New best (val_loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"\n⚠️  Early stopping (patience: {PATIENCE})")
                        break
                
                # Memory cleanup
                gc.collect()
            
            # Periodic checkpoint (less frequent for CPU)
            if (iter + 1) % SAVE_INTERVAL == 0 and (iter + 1) % EVAL_INTERVAL != 0:
                checkpoint_path = MODELS_DIR / f"checkpoint_hf_cpu_iter_{iter+1}.pt"
                torch.save(model.state_dict(), checkpoint_path)
    
    except KeyboardInterrupt:
        print_warning("\n⚠️  Training interrupted by user")
        print_success("Saving checkpoint...")
        torch.save(model.state_dict(), OUTPUT_MODEL)
    
    # Save final model and config
    torch.save(model.state_dict(), OUTPUT_MODEL)
    with open(OUTPUT_CONFIG, 'w') as f:
        json.dump({
            "vocab_size": vocab_size,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "block_size": BLOCK_SIZE,
            "tokenizer": TOKENIZER_NAME,
            "total_params": total_params,
            "best_val_loss": float(best_val_loss),
            "training_iterations": iter + 1,
            "source": "huggingface_dataset_cpu",
            "device": DEVICE,
            "batch_size": BATCH_SIZE
        }, f, indent=2)
    
    # Final summary
    total_time = (datetime.now() - start_time).total_seconds() / 60
    print_header("✅ TRAINING COMPLETE!")
    print(f"📊 Final metrics:")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Total time: {total_time:.1f} minutes")
    print(f"   Iterations: {iter + 1}/{MAX_ITERS}")
    print(f"\n📁 Output files:")
    print(f"   Model:   {OUTPUT_MODEL}")
    print(f"   Config:  {OUTPUT_CONFIG}")
    print(f"   Log:     {TRAINING_LOG}")
    print(f"\n📊 Model info:")
    print(f"   Parameters: {total_params/1e3:.1f}K")
    print(f"   Size: ~{(total_params * 4 / 1024 / 1024):.1f}MB")
    print("="*80 + "\n")

if __name__ == "__main__":
    train()
