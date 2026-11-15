# Gorev 4 - Head'i modele koy, egit, bigram tabanıyla karsilastir
# YZ50 Hafta 7 (1:19:11 - 1:21:59)

import torch
import torch.nn as nn
from torch.nn import functional as F
from ortak import V, decode, loss_olc, egit, BLOCK
from model import AttentionDilModeli, N_EMBD, HEAD_SIZE


class BigramDilModeli(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(V, V)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        return logits, F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, g=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1, generator=g)), 1)
        return idx


sonuc = {}

print("=" * 62)
print("TABAN: BIGRAM")
print("=" * 62)
torch.manual_seed(1337)
bigram = BigramDilModeli()
egit(bigram, adim=5000, lr=1e-2, sessiz=True)
s = loss_olc(bigram)
sonuc['bigram'] = (sum(p.nelement() for p in bigram.parameters()), s)
print(f"  parametre {sonuc['bigram'][0]}   train {s['train']:.4f}   val {s['val']:.4f}")
print()

print("=" * 62)
print("TEK HEAD'LI SELF-ATTENTION")
print("=" * 62)
torch.manual_seed(1337)
model = AttentionDilModeli()
print(f"  parametre sayisi: {sum(p.nelement() for p in model.parameters())}")
print("--- egitim (5000 adim, lr=1e-3) ---")
egit(model, adim=5000, lr=1e-3)
s2 = loss_olc(model)
sonuc['attention'] = (sum(p.nelement() for p in model.parameters()), s2)
print()
print(f"  train {s2['train']:.4f}   val {s2['val']:.4f}")
print()

print("=" * 62)
print("KARSILASTIRMA")
print("=" * 62)
print(f"{'model':<24}{'parametre':>12}{'train':>10}{'val':>10}")
for ad, (n, s) in sonuc.items():
    print(f"{ad:<24}{n:>12}{s['train']:>10.4f}{s['val']:>10.4f}")
fark = sonuc['bigram'][1]['val'] - sonuc['attention'][1]['val']
print(f"{'iyilesme':<24}{'':>12}{'':>10}{fark:>+10.4f}")
print()

print("--- tek head'in urettigi 400 karakter ---")
gg = torch.Generator().manual_seed(42)
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), 400, g=gg)[0].tolist()))
print()

print("=" * 62)
print("KENDI CUMLELERIMLE")
print("=" * 62)
print("""
  NEDEN DUSTU
  Bigram sadece BIR onceki harfe bakiyordu: 'q' gorunce 'u' demek
  gibi. Tek head'li model sekiz harfe birden bakabiliyor VE hangisine
  ne kadar bakacagini kendi secebiliyor.

  Hafta 6'daki WaveNet de sekiz harfe bakiyordu, ama birlestirme
  sekli SABITTI: hep ikiser ikiser, hep ayni agacla. Burada agirliklar
  veriye gore degisiyor. Bir sessiz harf kendinden onceki sesliyi
  arayabilir; satir basindaki bir karakter bir onceki satir sonunu
  arayabilir. Model bunu kimse soylemeden ogreniyor.

  Ikinci fark: pozisyon embedding'i. Model artik 'ben bu parcanin
  3. karakteriyim' bilgisini de tasiyor.

  NEDEN SON block_size HARFE KIRPMAK ZORUNDAYIZ
  position_embedding_table'in tam olarak block_size satiri var
  (burada 8). Modele 9 karakterlik bir girdi verirsem
  torch.arange(9) cagrisi 8. satiri aramaya calisir ve tablo o kadar
  uzun degildir: IndexError.

  Bu yuzden generate icinde idx[:, -block_size:] yaziyoruz. Model
  uretirken metin uzuyor ama her adimda sadece son 8 karakteri
  modele veriyoruz.

  Bunun bedeli var: model 8 karakterden eskisini unutuyor. GPT'nin
  'context window' dedigi sey tam olarak bu sinir.
""")
