# model_h8.py - Hafta 8'de yazdigim GPT, token dizisiyle calisacak sekilde
# YZ50 Hafta 9, Gorev 6(b)
#
# Mimari Hafta 8 ile AYNI: multi-head attention + feedforward + residual
# + LayerNorm, 4 blok. Tek fark: girdi artik karakter id'leri degil,
# BPE token id'leri. Kodda degisen tek sey veri katmani.

import time

import torch
import torch.nn as nn
from torch.nn import functional as F


class TokenVeri:
    """Bir token id listesini alir, %90/%10 boler.

    Hafta 8'de Turkce korpusu SIRAYLA bolmustum ve val loss train'den
    dusuk cikmisti - val dilimi daha homojendi. Bu sefer dilimleri
    karistirarak boluyorum.
    """

    def __init__(self, ids, V, block, tohum=1337, karistir=True):
        self.V = V
        self.block = block
        d = torch.tensor(ids, dtype=torch.long)
        if karistir:
            # dizi butunlugu bozulmasin diye block*4 uzunlugunda
            # dilimler halinde karistiriyorum
            parca = block * 4
            n = len(d) // parca
            d = d[:n * parca].view(n, parca)
            g = torch.Generator().manual_seed(tohum)
            d = d[torch.randperm(n, generator=g)].reshape(-1)
        n = int(0.9 * len(d))
        self.train_data, self.val_data = d[:n], d[n:]

    def get_batch(self, bolum, block=None, batch=32, g=None):
        block = block or self.block
        d = self.train_data if bolum == 'train' else self.val_data
        ix = torch.randint(len(d) - block, (batch,), generator=g)
        x = torch.stack([d[i:i + block] for i in ix])
        y = torch.stack([d[i + 1:i + block + 1] for i in ix])
        return x, y


class Head(nn.Module):
    def __init__(self, head_size, n_embd, block, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block, block)))
        self.drop = nn.Dropout(dropout)
        self.hs = head_size

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * self.hs ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.drop(F.softmax(wei, dim=-1))
        return wei @ self.value(x)


class MultiHead(nn.Module):
    def __init__(self, n_head, n_embd, block, dropout):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(hs, n_embd, block, dropout)
                                    for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], -1)))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Blok(nn.Module):
    def __init__(self, n_embd, n_head, block, dropout):
        super().__init__()
        self.sa = MultiHead(n_head, n_embd, block, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, V, n_embd=128, n_head=4, n_blok=4, block=64,
                 dropout=0.1):
        super().__init__()
        self.block = block
        self.token_embedding_table = nn.Embedding(V, n_embd)
        self.position_embedding_table = nn.Embedding(block, n_embd)
        self.bloklar = nn.Sequential(*[Blok(n_embd, n_head, block, dropout)
                                       for _ in range(n_blok)])
        self.son_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, V)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + \
            self.position_embedding_table(torch.arange(T))
        logits = self.lm_head(self.son_ln(self.bloklar(x)))
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, g=None):
        self.eval()
        for _ in range(n):
            logits, _ = self(idx[:, -self.block:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1, generator=g)), 1)
        self.train()
        return idx


@torch.no_grad()
def loss_olc(model, veri, tekrar=200, batch=32):
    model.eval()
    s = {}
    g = torch.Generator().manual_seed(1234)
    for bolum in ['train', 'val']:
        k = torch.zeros(tekrar)
        for i in range(tekrar):
            X, Y = veri.get_batch(bolum, batch=batch, g=g)
            k[i] = model(X, Y)[1].item()
        s[bolum] = k.mean().item()
    model.train()
    return s


def egit(model, veri, adim=4000, lr=3e-4, batch=32, her=500):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1)
    t0 = time.time()
    for i in range(adim):
        xb, yb = veri.get_batch('train', batch=batch, g=g)
        loss = model(xb, yb)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % her == 0:
            s = loss_olc(model, veri, tekrar=50, batch=batch)
            print(f"    adim {i:>5}  train {s['train']:.4f}  val {s['val']:.4f}"
                  f"  ({time.time()-t0:.0f} sn)")
    return time.time() - t0


def param_sayisi(m):
    return sum(p.nelement() for p in m.parameters())
