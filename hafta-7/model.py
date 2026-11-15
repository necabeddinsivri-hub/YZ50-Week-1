# model.py - tek head'li self-attention dil modeli
# YZ50 Hafta 7

import torch
import torch.nn as nn
from torch.nn import functional as F
from ortak import V, BLOCK

N_EMBD = 32
HEAD_SIZE = 32


class Head(nn.Module):
    """Tek bir self-attention head."""

    def __init__(self, head_size, block=BLOCK, maske=True):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block, block)))
        self.maske = maske
        self.head_size = head_size
        self.son_wei = None          # gorsellestirme icin saklariz

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5    # (B,T,T)
        if self.maske:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.son_wei = wei.detach()
        v = self.value(x)
        return wei @ v


class AttentionDilModeli(nn.Module):
    def __init__(self, block=BLOCK, maske=True):
        super().__init__()
        self.token_embedding_table = nn.Embedding(V, N_EMBD)
        self.position_embedding_table = nn.Embedding(block, N_EMBD)
        self.sa_head = Head(HEAD_SIZE, block=block, maske=maske)
        self.lm_head = nn.Linear(HEAD_SIZE, V)
        self.block = block

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)                       # (B,T,C)
        pos = self.position_embedding_table(torch.arange(T))        # (T,C)
        x = tok + pos                                               # sira bilgisi eklendi
        x = self.sa_head(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, g=None):
        for _ in range(max_new_tokens):
            # pozisyon tablosu sadece block kadar satir iceriyor:
            # girdiyi son block karaktere KIRPMAK zorundayiz
            idx_cond = idx[:, -self.block:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=g)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
