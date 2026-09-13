# Gorev 2 - Gizli katman, cikis katmani ve loss
# YZ50 Hafta 4 (Part 2, 18:35 - 37:56)

import torch
import torch.nn.functional as F

kelimeler = open('names.txt', 'r').read().splitlines()
harfler = sorted(list(set(''.join(kelimeler))))
stoi = {h: i + 1 for i, h in enumerate(harfler)}
stoi['.'] = 0
itos = {i: h for h, i in stoi.items()}

BAGLAM = 3


def veri_kur(kelime_listesi):
    X, Y = [], []
    for k in kelime_listesi:
        pencere = [0] * BAGLAM
        for harf in k + '.':
            ix = stoi[harf]
            X.append(pencere)
            Y.append(ix)
            pencere = pencere[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


X, Y = veri_kur(kelimeler)

# --- MODEL PARAMETRELERI ---
g = torch.Generator().manual_seed(2147483647)
BOYUT_EMB = 2       # her harf kac sayiyla temsil ediliyor
GIZLI = 100         # gizli katmandaki neuron sayisi

C  = torch.randn((27, BOYUT_EMB), generator=g)
W1 = torch.randn((BAGLAM * BOYUT_EMB, GIZLI), generator=g)
b1 = torch.randn(GIZLI, generator=g)
W2 = torch.randn((GIZLI, 27), generator=g)
b2 = torch.randn(27, generator=g)
parametreler = [C, W1, b1, W2, b2]

print("--- model boyutlari ---")
for isim, p in zip(['C', 'W1', 'b1', 'W2', 'b2'], parametreler):
    print(f"  {isim:<3} {str(tuple(p.shape)):<12} {p.nelement():>6} parametre")
print(f"  TOPLAM: {sum(p.nelement() for p in parametreler)} parametre")
print()


# --- ILERI GECIS, adim adim ---
print("--- ileri gecis, sekil takibi ---")
emb = C[X]
print("1. emb = C[X]                    ->", tuple(emb.shape), " (ornek, 3 harf, 2 sayi)")

duz = emb.view(-1, BAGLAM * BOYUT_EMB)
print("2. duzlestir (view)              ->", tuple(duz.shape), "    (3 harf yan yana: 6 sayi)")

h = torch.tanh(duz @ W1 + b1)
print("3. gizli katman: tanh(x@W1+b1)   ->", tuple(h.shape), "  (100 neuron)")

logits = h @ W2 + b2
print("4. cikis: logits = h@W2+b2       ->", tuple(logits.shape), "   (27 harf icin ham skor)")
print()


# --- LOSS: once elle, sonra hazir fonksiyonla ---
sayimlar = logits.exp()
olasilik = sayimlar / sayimlar.sum(1, keepdim=True)
loss_elle = -olasilik[torch.arange(X.shape[0]), Y].log().mean()

loss_hazir = F.cross_entropy(logits, Y)

print("--- loss karsilastirmasi ---")
print(f"  elle hesap        : {loss_elle.item():.6f}")
print(f"  F.cross_entropy   : {loss_hazir.item():.6f}")
print(f"  fark              : {abs(loss_elle.item() - loss_hazir.item()):.2e}")
print()

print("Peki neden F.cross_entropy tercih ediliyor?")
print("  1) Daha az ara tensor uretiyor, bellek dostu.")
print("  2) Backward pass'i icin sadelesmis analitik formul kullaniyor, daha hizli.")
print("  3) SAYISAL OLARAK GUVENLI: buyuk logits'te exp() patlar.")
print()


# --- 3. maddeyi canli gorelim ---
print("--- sayisal guvenlik testi ---")
buyuk = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
hedef = torch.tensor([3])

s = buyuk.exp()
p = s / s.sum(1, keepdim=True)
elle = -p[0, 3].log()

print("  logits:", buyuk.tolist()[0])
print("  exp() sonucu:", s.tolist()[0], " <- son eleman patladi")
print(f"  elle loss        : {elle.item()}")
print(f"  F.cross_entropy  : {F.cross_entropy(buyuk, hedef).item():.6f}")
print()
print("  cross_entropy icten en buyuk logit'i cikarip normalize ediyor,")
print("  boylece exp() hicbir zaman tasmaz. Bizim elle versiyonumuz")
print("  daha buyuk sayilarda inf uretir.")
