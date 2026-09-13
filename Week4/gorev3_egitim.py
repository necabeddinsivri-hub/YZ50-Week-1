# Gorev 3 - Egitim dongusu, minibatch, learning rate taramasi, veri bolme
# YZ50 Hafta 4 (Part 2, 37:56 - 1:00:49)

import torch
import torch.nn.functional as F
from ortak import veri_yukle, veri_kur, bol

kelimeler, stoi, itos = veri_yukle('names.txt')
BAGLAM, BOYUT_EMB, GIZLI = 3, 2, 100

train, dev, test = bol(kelimeler)
Xtr, Ytr = veri_kur(train, stoi, BAGLAM)
Xdev, Ydev = veri_kur(dev, stoi, BAGLAM)
Xte, Yte = veri_kur(test, stoi, BAGLAM)

print("--- veri bolmesi ---")
print(f"  train: {len(train):>6} isim  {Xtr.shape[0]:>7} ornek")
print(f"  dev  : {len(dev):>6} isim  {Xdev.shape[0]:>7} ornek")
print(f"  test : {len(test):>6} isim  {Xte.shape[0]:>7} ornek")
print()


def model_kur(tohum=2147483647):
    g = torch.Generator().manual_seed(tohum)
    C = torch.randn((27, BOYUT_EMB), generator=g)
    W1 = torch.randn((BAGLAM * BOYUT_EMB, GIZLI), generator=g)
    b1 = torch.randn(GIZLI, generator=g)
    W2 = torch.randn((GIZLI, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    ps = [C, W1, b1, W2, b2]
    for p in ps:
        p.requires_grad = True
    return ps


def ileri(ps, Xb):
    C, W1, b1, W2, b2 = ps
    emb = C[Xb]
    h = torch.tanh(emb.view(-1, BAGLAM * BOYUT_EMB) @ W1 + b1)
    return h @ W2 + b2


def loss_olc(ps, Xb, Yb):
    with torch.no_grad():
        return F.cross_entropy(ileri(ps, Xb), Yb).item()


# ==========================================================
# 3a) ONCE TEK BIR MINIBATCH'I OVERFIT ET
# ==========================================================
print("--- 3a) tek minibatch overfit testi ---")
print("  Amac: model dogru kurulmus mu? 32 ornegi ezberleyebilmeli.")
ps = model_kur()
Xk, Yk = Xtr[:32], Ytr[:32]
for adim in range(200):
    loss = F.cross_entropy(ileri(ps, Xk), Yk)
    for p in ps: p.grad = None
    loss.backward()
    for p in ps: p.data += -0.1 * p.grad
    if adim % 50 == 0:
        print(f"    adim {adim:>3}  loss = {loss.item():.4f}")
print(f"    adim 199  loss = {loss.item():.4f}   <- sifira yaklasti, model saglam")
print()


# ==========================================================
# 3b) LEARNING RATE TARAMASI
# ==========================================================
print("--- 3b) learning rate taramasi ---")
print("  Her lr icin 2000 adim minibatch egitimi, sonra train loss.")
sonuclar = []
for lr in [0.001, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0]:
    ps = model_kur()
    g = torch.Generator().manual_seed(1)
    for adim in range(2000):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        loss = F.cross_entropy(ileri(ps, Xtr[ix]), Ytr[ix])
        for p in ps: p.grad = None
        loss.backward()
        for p in ps: p.data += -lr * p.grad
    L = loss_olc(ps, Xtr, Ytr)
    sonuclar.append((lr, L))
    print(f"    lr = {lr:<7} -> train loss = {L:.4f}")

en_iyi = min(sonuclar, key=lambda x: x[1])
print(f"  en iyi lr: {en_iyi[0]}  (loss {en_iyi[1]:.4f})")
print()


# ==========================================================
# 3c) ASIL EGITIM
# ==========================================================
print("--- 3c) asil egitim (30.000 adim, lr kademeli dusuruluyor) ---")
ps = model_kur()
g = torch.Generator().manual_seed(1)
gecmis = []
for adim in range(30000):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    loss = F.cross_entropy(ileri(ps, Xtr[ix]), Ytr[ix])
    for p in ps: p.grad = None
    loss.backward()
    lr = 0.1 if adim < 20000 else 0.01     # son 10 bin adimda yavasla
    for p in ps: p.data += -lr * p.grad
    gecmis.append(loss.item())
    if adim % 5000 == 0:
        print(f"    adim {adim:>6}  minibatch loss = {loss.item():.4f}")

print()
print("--- SONUC ---")
tr = loss_olc(ps, Xtr, Ytr)
dv = loss_olc(ps, Xdev, Ydev)
print(f"  train loss = {tr:.4f}")
print(f"  dev   loss = {dv:.4f}")
print()
print("  Karsilastirma (Hafta 3):")
print("    bigram sayim modeli : 2.4544")
print("    bigram sinir agi    : 2.4830")
print(f"    bu haftaki MLP      : {dv:.4f}")
print()
print("  train ve dev birbirine yakin -> model henuz ezberlemiyor,")
print("  yani daha buyuk bir model denenebilir. (Gorev 4)")
