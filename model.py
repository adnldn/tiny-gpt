"""
Basic GPT2 Model
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """ Casual i.e., decoder """

    def __init__(self, config):
        super().__init__()

        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be a multiple of n_head")
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
     
        # combined k, q and v projections. Split later.
        self.cmb_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.cmb_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias) #  output projection
        self.attn_dropout = nn.Dropout(self.dropout) #  regularisation
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        q, k, v = self.query_key_value(x)
        y = self.scaled_dot_product_attention(q, k, v)
        y = self.reassemble_heads(y)

        # output projection
        y = self.resid_dropout(self.cmb_proj(y))
        return y

    def query_key_value(self, x):
        """Split the last dimension of x into (heads, depth)"""
        B, T, C = x.size() #  Batch, sequence length and embedding dimension
        q, k, v = self.cmb_attn(x).split(self.n_embd, dim=2)
        return [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v)]

    def scaled_dot_product_attention(self, q, k, v, attn_mask=None, is_causal=True):
        """Calculate the scaled dot product attention"""
        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            raise RuntimeError("PyTorch 2.0 or higher is required for scaled_dot_product_attention")
        
        dropout_p = self.dropout if self.training else 0
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
    
    def reassemble_heads(self, y):
        """Reassembles all head outputs into C"""
        B, H, T, C = y.size()
        return y.transpose(1, 2).contiguous().view(B, T, C * H)
    

class MLP(nn.Module):
    """ Multi-layer perceptron for computation in-between communication """
    def __init__(self, config, activation=nn.GELU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            activation,
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        return self.net(x)  


class Block(nn.Module):
    """ Transformer 'layers' """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias) 
        self.attn = CausalSelfAttention(config) # Takes in x~(B, T, C) and returns y~(B, T, C)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # add normalised input to passed through communication layer to the input
        x = x + self.mlp(self.ln_2(x)) # add normalised input to passed through computation layer to the input
        return x


@dataclass
class GPTConfig:
    """
    Produces the config which is passed to the model.
    Overriden by configuration dictionary if given kwargs.
    """

    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab size of 50257; padded to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    """ GPT model based of OpenAI's GPT-2 """
    def __init__(self, config):
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("config.vocab_size missing")
        if config.block_size is None:
            raise ValueError("config.block_size missing")
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._init_linear(m)
            elif isinstance(m, nn.Embedding):
                self._init_embedding(m)

        for pn, p in self.named_parameters():
            if pn.endswith('cmb_proj.weight'):
                self._init_cmb_proj(p)

    def _init_linear(self, m):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    def _init_embedding(self, m):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _init_cmb_proj(self, p):
        nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"context length {T} must be less than {self.config.block_size}")
        pos = torch.arange(T, dtype=torch.long, device=device)

        # forward the GPT model
        tok_emb = self.wte(idx) # token embedings ~ (B, T, n_embd)
        pos_emb = self.wpe(pos) # position embeddings ~ (T , n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # exclude targets that are -1 from the loss calculation
            # if device in ['cpu', 'cuda']:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) # list index used to preserve time dimention i.e., ~(B, 1, vocab_size)
            loss = None
        
        return logits, loss
    
    def configure_optimisers(self, weight_decay, learning_rate, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        weights = [p for p in params if p.dim() >= 2]
        biases = [p for p in params if p.dim() < 2]

        optim_groups = [
            {'params': weights, 'weight_decay': weight_decay},
            {'params': biases, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in weights)
        num_nodecay_params = sum(p.numel() for p in biases)

        print(f"No. of decayed parameter tensors: {len(weights)}, with {num_decay_params} parameters")
        print(f"No. not decayed parameter tensors: {len(biases)}, with {num_nodecay_params} parameters")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        fused_args = dict(fused=True) if use_fused else dict()
        optimiser = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **fused_args)

        return optimiser

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Only sample from top-k if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # sample and append the next token
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
