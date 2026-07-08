#!/usr/bin/env python3
"""
PHASE 2: MODEL TRAINING
Trains a compact PyTorch transformer model on generated datasets.
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
INPUT_JSONL = DATASETS_DIR / "generated_projects_final.jsonl" # Matches Phase 1
OUTPUT_MODEL = MODELS_DIR / "tiny_code_model.pt"
OUTPUT_CONFIG = MODELS_DIR / "model_config.json"
TRAINING_LOG = LOGS_DIR / "training_log.jsonl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 4
BLOCK_SIZE = 128
MAX_ITERS = 6000
LEARNING_RATE = 5e-4
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
GRAD_CLIP = 1.0
TOKENIZER_NAME = "cl100k_base"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# MODEL ARCHITECTURE (FIXED: Added Causal Mask & Weight Init)
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
        
        # CRITICAL FIX: Apply Causal Mask to prevent looking into the future
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
        
        # CRITICAL FIX: Changed to ModuleList to pass 'mask' argument through blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights for better convergence
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

        # CRITICAL FIX: Generate Causal Mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        # Pass mask through each block
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
# DATA & TRAINING LOOP (Kept mostly identical, minor safety checks added)
# ============================================================================
def load_and_tokenize():
    print("Loading dataset...")
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Run Phase 1 first! Missing {INPUT_JSONL}")
    
    code_samples = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: code_samples.append(json.loads(line)['code'])
                except: continue
                
    text = "\n<|file_end|>\n".join(code_samples)
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    return data, enc, enc.n_vocab

def get_batch(split, train_data, val_data):
    data_split = train_data if split == 'train' else val_data
    # Safety check for small datasets
    max_ix = max(1, len(data_split) - BLOCK_SIZE)
    ix = torch.randint(max_ix, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def train():
    print(f"Device: {DEVICE}")
    data, enc, vocab_size = load_and_tokenize()
    
    n = int(0.8 * len(data))
    train_data, val_data = data[:n], data[n:]
    
    model = TinyCodeModel(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_ITERS)
    
    for iter in range(MAX_ITERS):
        x, y = get_batch('train', train_data, val_data)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        
        if (iter + 1) % 500 == 0:
            print(f"Iter {iter+1} | loss: {loss.item():.4f}")
            torch.save(model.state_dict(), OUTPUT_MODEL)

    print("✅ Training Complete!")
    with open(OUTPUT_CONFIG, 'w') as f:
        json.dump({
            "vocab_size": vocab_size, 
            "n_embd": N_EMBD, 
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "block_size": BLOCK_SIZE,
            "tokenizer": TOKENIZER_NAME
        }, f)

if __name__ == "__main__":
    train()