# Gorev 1 - Veriyi hazirla, bigram tabanini kur
# YZ50 Hafta 7 (7:52 - 38:00)

import torch
import torch.nn as nn
from torch.nn import functional as F
from ortak import (metin, harfler, V, encode, decode, veri, train_data,
                   val_data, get_batch, loss_olc, egit, BLOCK, BATCH)

print("=" * 62)
print("1a) VERI")
print("=" * 62)
print(f"  toplam karakter : {len(metin):,}".replace(',', '.'))
print(f"  alfabe boyutu   : {V}")
print(f"  alfabe          : {''.join(harfler)!r}")
print()
print("  ilk 120 karakter:")
print("  " + repr(metin[:120]))
print()

print("--- tokenizer testi ---")
ornek = "Merhaba Shakespeare"
print(f"  encode('{ornek}') = {encode(ornek)[:8]} ...")
print(f"  decode(encode(x)) == x  ->  {decode(encode(ornek)) == ornek}")
print()
print("  NOT: bu bir KARAKTER tokenizer'i. Alfabe kucuk (65), diziler uzun.")
print("  GPT gibi modeller kelime parcasi (BPE) kullaniyor: alfabe buyuk")
print("  (~50.000), diziler kisa. Takas bu.")
print()

print(f"  train: {len(train_data):,} karakter".replace(',', '.'))
print(f"  val  : {len(val_data):,} karakter".replace(',', '.'))
print()

print("=" * 62)
print("1b) BLOCK VE BATCH")
print("=" * 62)
x = train_data[:BLOCK]
y = train_data[1:BLOCK + 1]
print(f"  block_size = {BLOCK}. Tek bir parcada {BLOCK} ayri ornek var:")
for t in range(BLOCK):
    print(f"    baglam {str(x[:t+1].tolist()):<30} -> hedef {y[t].item()}")
print()
print("  Yani 8 karakterlik bir parca, modele 1'den 8'e kadar her uzunlukta")
print("  baglam gormeyi ogretiyor. Uretim sirasinda ilk harften baslayacagi")
print("  icin kisa baglamlari da gormesi gerekiyor.")
print()

g = torch.Generator().manual_seed(1337)
xb, yb = get_batch('train', g=g)
print(f"  xb sekli: {tuple(xb.shape)}   yb sekli: {tuple(yb.shape)}")
print(f"  yani {BATCH} bagimsiz parca, her biri {BLOCK} karakter")
print()


# ==========================================================
# 1c) BIGRAM TABANI
# ==========================================================
class BigramDilModeli(nn.Module):
    def __init__(self):
        super().__init__()
        # her token dogrudan bir sonraki token'in skorlarini okuyor
        self.token_embedding_table = nn.Embedding(V, V)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)      # (B, T, V)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, g=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                  # son adimin skorlari
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=g)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


print("=" * 62)
print("1c) BIGRAM TABANI")
print("=" * 62)
model = BigramDilModeli()
print(f"  parametre sayisi: {sum(p.nelement() for p in model.parameters())}")
print()
import math
print(f"  egitimsiz beklenen loss = -log(1/{V}) = {math.log(V):.4f}")
_, loss0 = model(xb, yb)
print(f"  olculen ilk loss        = {loss0.item():.4f}")
print()

print("--- egitim (5000 adim, lr=1e-2) ---")
egit(model, adim=5000, lr=1e-2)
s = loss_olc(model)
print()
print(f"  TABAN:  train {s['train']:.4f}   val {s['val']:.4f}")
print()

gg = torch.Generator().manual_seed(42)
baslangic = torch.zeros((1, 1), dtype=torch.long)
print("--- bigram'in urettigi 300 karakter ---")
print(decode(model.generate(baslangic, 300, g=gg)[0].tolist()))
