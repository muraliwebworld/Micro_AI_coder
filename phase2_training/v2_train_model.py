#!/usr/bin/env python3
"""
PHASE 2: MODEL TRAINING
Trains a compact PyTorch transformer model on generated datasets using tiktoken tokenization.
Input: datasets/generated_projects.jsonl
Output: models/tiny_code_model.pt, models/model_config.json, logs/training_log.jsonl
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import tiktoken
from pathlib import Path
import sys
from datetime import datetime
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent / "logs"
INPUT_JSONL = DATASETS_DIR / "generated_projects.jsonl"
OUTPUT_MODEL = MODELS_DIR / "tiny_code_model.pt"
OUTPUT_CONFIG = MODELS_DIR / "model_config.json"
TRAINING_LOG = LOGS_DIR / "training_log.jsonl"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Model Hyperparameters
BATCH_SIZE = 16
BLOCK_SIZE = 512          # Context window (~180 lines of code)
MAX_ITERS = 5000
LEARNING_RATE = 5e-4
N_EMBD = 256              # Embedding dimension
N_HEAD = 8                # Number of attention heads
N_LAYER = 8               # Number of transformer layers
DROPOUT = 0.1
GRAD_CLIP = 1.0

# Tokenization
TOKENIZER_NAME = "cl100k_base"

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text)
    print("="*80)

def print_step(text):
    """Print step info"""
    print(f"\n➤ {text}")

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_log(text):
    """Print log info"""
    print(f"📊 {text}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / np.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Project and reshape for multi-head attention
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

class TinyCodeModel(nn.Module):
    """Tiny PyTorch transformer for code generation"""
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
        # Embeddings
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = self.emb_dropout(tok_emb + pos_emb)
        
        # Create causal mask (optional, for efficiency)
        # For now, rely on next-token prediction loss
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Output logits
        logits = self.lm_head(x)
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens"""
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
# DATA LOADING
# ============================================================================

def load_dataset():
    """Load dataset from JSONL"""
    print_step("Loading dataset...")
    
    if not INPUT_JSONL.exists():
        print_error(f"Dataset not found: {INPUT_JSONL}")
        print("Run Phase 1 first: python phase1_dataset_creator/v2_dataset_creator.py")
        sys.exit(1)
    
    code_samples = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    code_samples.append(entry['code'])
                except (json.JSONDecodeError, KeyError):
                    continue
    
    if not code_samples:
        print_error("No valid code samples found in dataset")
        sys.exit(1)
    
    # Concatenate all code with file separator
    text = "\n<|file_end|>\n".join(code_samples)
    print_success(f"Loaded {len(code_samples)} code samples")
    print_log(f"Total text length: {len(text):,} characters")
    
    return text

def tokenize_data(text):
    """Tokenize data with tiktoken"""
    print_step(f"Tokenizing with {TOKENIZER_NAME}...")
    
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    
    vocab_size = enc.n_vocab
    print_success(f"Tokenization complete")
    print_log(f"Vocab size: {vocab_size:,} tokens")
    print_log(f"Total tokens: {len(data):,}")
    
    return data, enc, vocab_size

def create_dataloaders(data):
    """Create train/val data loaders"""
    print_step("Splitting data (80/20 train/val)...")
    
    n = int(0.8 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    def get_batch(split):
        """Get a batch of data"""
        data_split = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
        x, y = x.to(DEVICE), y.to(DEVICE)
        return x, y
    
    print_success(f"Train tokens: {len(train_data):,}")
    print_success(f"Val tokens: {len(val_data):,}")
    
    return get_batch

# ============================================================================
# TRAINING
# ============================================================================

def estimate_loss(model, get_batch, eval_iters=100):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out

def train():
    """Main training loop"""
    print_header("PHASE 2: MODEL TRAINING")
    print(f"Device: {DEVICE}")
    print(f"Model: TinyCodeModel")
    print(f"  - Embedding: {N_EMBD}")
    print(f"  - Heads: {N_HEAD}")
    print(f"  - Layers: {N_LAYER}")
    print(f"  - Block size: {BLOCK_SIZE}")
    print(f"  - Vocab: 100,257 (tiktoken)")
    
    # Load data
    text = load_dataset()
    data, enc, vocab_size = tokenize_data(text)
    get_batch = create_dataloaders(data)
    
    # Initialize model
    print_step("Initializing model...")
    model = TinyCodeModel(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print_success(f"Model initialized")
    print_log(f"Total parameters: {n_params:,} (~{n_params/1e6:.1f}M)")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_ITERS)
    
    # Training loop
    print_header("TRAINING")
    best_val_loss = float('inf')
    
    TRAINING_LOG.write_text("", encoding="utf-8")  # Clear log file
    
    for iter in range(MAX_ITERS):
        # Sample batch
        x, y = get_batch('train')
        
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if (iter + 1) % 500 == 0 or iter == 0:
            losses = estimate_loss(model, get_batch, eval_iters=50)
            lr = optimizer.param_groups[0]['lr']
            
            log_line = {
                "iter": iter + 1,
                "train_loss": round(losses['train'], 4),
                "val_loss": round(losses['val'], 4),
                "lr": round(lr, 8)
            }
            
            # Save log
            with open(TRAINING_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_line) + "\n")
            
            print(f"Iter {iter+1:5d} | train_loss: {losses['train']:6.4f} | val_loss: {losses['val']:6.4f} | lr: {lr:.2e}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), OUTPUT_MODEL)
                print(f"           ✅ New best model saved (val_loss: {best_val_loss:.4f})")
    
    # Final save
    print_header("TRAINING COMPLETE")
    print_success(f"Model saved to {OUTPUT_MODEL}")
    print_log(f"Final validation loss: {best_val_loss:.4f}")
    
    # Save config
    config = {
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "block_size": BLOCK_SIZE,
        "vocab_size": vocab_size,
        "tokenizer": TOKENIZER_NAME,
        "dropout": DROPOUT,
        "trained_at": datetime.now().isoformat(),
        "max_iters": MAX_ITERS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "total_parameters": n_params
    }
    
    with open(OUTPUT_CONFIG, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print_success(f"Config saved to {OUTPUT_CONFIG}")
    
    # Sample generation
    print_header("VALIDATION SAMPLING")
    model.eval()
    with torch.no_grad():
        prompts = [
            "const [state, setState] = use",
            "const express = require",
            "CREATE TABLE"
        ]
        
        for prompt in prompts:
            tokens = enc.encode(prompt)
            x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
            generated = model.generate(x, max_new_tokens=50, temperature=0.7)
            completion = enc.decode(generated[0].tolist())
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {completion[:200]}...")

if __name__ == "__main__":
    train()
