#!/usr/bin/env python3
"""
PHASE 2: MODEL TRAINING (HUGGING FACE DATASET VERSION)
Trains a PyTorch transformer model on Hugging Face dataset.
Features: Better data handling, flexible field detection, improved validation.
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

# Configuration
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent / "logs"
INPUT_JSONL = DATASETS_DIR / "data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl"
OUTPUT_MODEL = MODELS_DIR / "tiny_code_model_huggingface.pt"
OUTPUT_CONFIG = MODELS_DIR / "model_config_huggingface.json"
TRAINING_LOG = LOGS_DIR / "training_log_huggingface.jsonl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters (Tuned for larger dataset)
BATCH_SIZE = 16             # Larger batch for more data
BLOCK_SIZE = 128
MAX_ITERS = 10000           # More iterations with more data
LEARNING_RATE = 1e-4
N_EMBD = 256                # Medium-size model
N_HEAD = 8
N_LAYER = 12
DROPOUT = 0.1
GRAD_CLIP = 1.0
TOKENIZER_NAME = "cl100k_base"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training improvements
EVAL_INTERVAL = 200         # Evaluate validation every N iterations
WARMUP_STEPS = 500          # Warmup phase for learning rate
WEIGHT_DECAY = 0.01         # L2 regularization
SAVE_INTERVAL = 500         # Save checkpoint every N iterations
PATIENCE = 15               # Early stopping patience

print_header = lambda text: print("\n" + "="*80 + "\n" + text + "\n" + "="*80)
print_step = lambda text: print(f"\n➤ {text}")
print_success = lambda text: print(f"✅ {text}")
print_warning = lambda text: print(f"⚠️  {text}")

# ============================================================================
# MODEL ARCHITECTURE (Same as v2_train_model.py)
# ============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
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
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
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
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
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
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============================================================================
# DATA LOADING (FLEXIBLE FIELD DETECTION)
# ============================================================================

def extract_code_field(entry):
    """Extract code from entry - handles conversation format with messages"""
    
    # Check if entry has messages (conversation format)
    if 'messages' in entry and isinstance(entry['messages'], list):
        # Find assistant's response with code
        code_blocks = []
        
        for msg in entry['messages']:
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Extract code blocks (between ``` markers)
                if '```' in content:
                    # Split by ``` and get code blocks
                    parts = content.split('```')
                    for i in range(1, len(parts), 2):
                        code = parts[i].strip()
                        # Remove language specifier (e.g., tsx, python, etc.)
                        lines = code.split('\n')
                        if lines and lines[0].lower() in ['tsx', 'jsx', 'javascript', 'typescript', 'python', 'java', 'cpp', 'c', 'go', 'rust', 'php', 'html', 'css']:
                            code = '\n'.join(lines[1:])
                        if len(code) > 100:
                            code_blocks.append(code.strip())
        
        if code_blocks:
            return '\n'.join(code_blocks)
    
    # Fallback to direct field lookup
    for field in ['code', 'response', 'solution', 'content', 'body', 'text', 'output']:
        if field in entry and isinstance(entry[field], str) and len(entry[field]) > 50:
            return entry[field]
    
    return None

def load_and_tokenize():
    print_step("Loading Hugging Face dataset...")
    
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_JSONL}")
    
    code_samples = []
    skipped = 0
    line_num = 0
    
    try:
        with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                if line.strip():
                    try:
                        entry = json.loads(line)
                        code = extract_code_field(entry)
                        if code:
                            code_samples.append(code)
                        else:
                            skipped += 1
                    except (json.JSONDecodeError, Exception) as e:
                        skipped += 1
                
                if line_num % 10000 == 0:
                    print(f"  Processed {line_num:,} lines... ({len(code_samples)} valid)")
    except KeyboardInterrupt:
        print_warning("Loading interrupted by user")
    
    print_success(f"Loaded {len(code_samples)} code samples from {line_num:,} lines")
    print_warning(f"Skipped {skipped} entries")
    
    if not code_samples:
        raise ValueError("No valid code samples found in dataset!")
    
    # Tokenize
    print_step("Tokenizing...")
    text = "\n<|file_end|>\n".join(code_samples)
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    
    print_success(f"Tokenized to {len(data):,} tokens")
    return data, enc, enc.n_vocab

def get_batch(split, train_data, val_data):
    data_split = train_data if split == 'train' else val_data
    max_ix = max(1, len(data_split) - BLOCK_SIZE)
    ix = torch.randint(max_ix, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=10):
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
        return LEARNING_RATE

# ============================================================================
# TRAINING
# ============================================================================

def train():
    print_header("🚀 TRAINING WITH HUGGING FACE DATASET")
    
    print(f"Device: {DEVICE}")
    data, enc, vocab_size = load_and_tokenize()
    
    # Split (80/20)
    n = int(0.8 * len(data))
    train_data, val_data = data[:n], data[n:]
    print_success(f"Train set: {len(train_data):,} tokens")
    print_success(f"Val set: {len(val_data):,} tokens")
    
    model = TinyCodeModel(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print_success(f"Model parameters: {total_params/1e6:.2f}M")
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_ITERS)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print_header("TRAINING PROGRESS")
    
    for iter in range(MAX_ITERS):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        x, y = get_batch('train', train_data, val_data)
        logits = model(x)
        train_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        
        if (iter + 1) % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data, eval_iters=10)
            val_loss = losses['val']
            
            # Log
            log_entry = {
                'iteration': iter + 1,
                'train_loss': losses['train'],
                'val_loss': val_loss,
                'learning_rate': lr,
                'timestamp': datetime.now().isoformat()
            }
            with open(TRAINING_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Display
            improvement = "↓" if val_loss < best_val_loss else "↑"
            print(f"Iter {iter+1:6d} | train_loss: {losses['train']:.4f} | val_loss: {val_loss:.4f} {improvement} | lr: {lr:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), OUTPUT_MODEL)
                print(f"           ✅ New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n⚠️  Early stopping triggered (patience: {PATIENCE})")
                    break
        
        # Checkpoint
        if (iter + 1) % SAVE_INTERVAL == 0 and (iter + 1) % EVAL_INTERVAL != 0:
            checkpoint_path = MODELS_DIR / f"checkpoint_hf_iter_{iter+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
    
    # Save final config
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
            "training_iterations": MAX_ITERS,
            "source": "huggingface_dataset"
        }, f, indent=2)
    
    print_header("✅ TRAINING COMPLETE!")
    print(f"📊 Final metrics:")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {OUTPUT_MODEL}")
    print(f"   Config saved to: {OUTPUT_CONFIG}")
    print(f"   Training log: {TRAINING_LOG}")
    print(f"   Total parameters: {total_params/1e6:.2f}M")
    print("="*80 + "\n")

if __name__ == "__main__":
    train()
