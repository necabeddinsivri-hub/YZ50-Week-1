# Gorev 7 - Ayni MLP'yi Turkce isimlerle egitmek
# YZ50 Hafta 4

import math
import torch
import torch.nn.functional as F
from ortak import veri_yukle, veri_kur, bol

kelimeler, stoi, itos = veri_yukle('isimler.txt')
K = len(stoi)
BAGLAM, EMB, GIZLI = 3, 10, 200

print("--- Turkce veri ---")
print("toplam isim:", len(kelimeler))
print("alfabe:", ''.join(sorted(h for h in stoi if h != '.')), f"({K-1} harf + nokta = {K})")
print()

# hiz icin alt kume (tam liste 1.8M ornek uretiyor)
kelimeler = kelimeler[::4]
print("egitimde kullanilan isim sayisi:", len(kelimeler))

train, dev, test = bol(kelimeler)
Xtr, Ytr = veri_kur(train, stoi, BAGLAM)
Xdev, Ydev = veri_kur(dev, stoi, BAGLAM)
print(f"train {Xtr.shape[0]} ornek, dev {Xdev.shape[0]} ornek")
print()

kaiming = (5/3) / math.sqrt(BAGLAM * EMB)
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((K, EMB), generator=g)
W1 = torch.randn((BAGLAM * EMB, GIZLI), generator=g) * kaiming
b1 = torch.randn(GIZLI, generator=g) * 0.01
W2 = torch.randn((GIZLI, K), generator=g) * 0.01
b2 = torch.zeros(K)
ps = [C, W1, b1, W2, b2]
for p in ps: p.requires_grad = True
print("parametre sayisi:", sum(p.nelement() for p in ps))
print()


def ileri(Xb):
    emb = C[Xb]
    h = torch.tanh(emb.view(-1, BAGLAM * EMB) @ W1 + b1)
    return h @ W2 + b2


print("--- egitim (25.000 adim) ---")
gr = torch.Generator().manual_seed(1)
for adim in range(25000):
    ix = torch.randint(0, Xtr.shape[0], (64,), generator=gr)
    loss = F.cross_entropy(ileri(Xtr[ix]), Ytr[ix])
    for p in ps: p.grad = None
    loss.backward()
    lr = 0.1 if adim < 17500 else 0.01
    for p in ps: p.data += -lr * p.grad
    if adim % 5000 == 0:
        print(f"  adim {adim:>6}  minibatch loss = {loss.item():.4f}")

with torch.no_grad():
    tr = F.cross_entropy(ileri(Xtr), Ytr).item()
    dv = F.cross_entropy(ileri(Xdev), Ydev).item()
print()
print(f"train loss = {tr:.4f}")
print(f"dev   loss = {dv:.4f}")
print()

print("--- TURKCE KARSILASTIRMA ---")
print("  Hafta 3 bigram sayim modeli : 2.5830")
print("  Hafta 3 bigram sinir agi    : 2.6118")
print(f"  Hafta 4 MLP (3 harf baglam) : {dv:.4f}")
print()

print("--- MLP'nin urettigi 20 Turkce isim ---")
gs = torch.Generator().manual_seed(42)
for _ in range(20):
    cikti, pencere = [], [0] * BAGLAM
    while True:
        with torch.no_grad():
            p = F.softmax(ileri(torch.tensor([pencere])), dim=1)
        ix = torch.multinomial(p, num_samples=1, generator=gs).item()
        if ix == 0:
            break
        cikti.append(itos[ix])
        pencere = pencere[1:] + [ix]
    print("  ", ''.join(cikti))
print()
print("Hafta 3 bigram'in urettikleri: ar, zi, nurd, guym, zurancal, usulaiy")
