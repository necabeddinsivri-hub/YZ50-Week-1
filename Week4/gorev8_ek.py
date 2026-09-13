# Gorev 8 (ek) - Karpathy Part 3, E02:
# Egitim bittikten sonra BatchNorm'u bir onceki Linear katmanin
# W ve b'sine KATLA. Forward pass ayni kalmali.
# YZ50 Hafta 4

import math
import torch
import torch.nn.functional as F
from ortak import veri_yukle, veri_kur, bol

kelimeler, stoi, itos = veri_yukle('names.txt')
BAGLAM, EMB, GIZLI = 3, 10, 200
train, dev, test = bol(kelimeler)
Xtr, Ytr = veri_kur(train, stoi, BAGLAM)
Xdev, Ydev = veri_kur(dev, stoi, BAGLAM)

kaiming = (5/3) / math.sqrt(BAGLAM * EMB)
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((27, EMB), generator=g)
W1 = torch.randn((BAGLAM * EMB, GIZLI), generator=g) * kaiming
W2 = torch.randn((GIZLI, 27), generator=g) * 0.01
b2 = torch.zeros(27)
kazanc = torch.ones((1, GIZLI))
kaydirma = torch.zeros((1, GIZLI))
ps = [C, W1, W2, b2, kazanc, kaydirma]
for p in ps: p.requires_grad = True
durum = {'ort': torch.zeros((1, GIZLI)), 'std': torch.ones((1, GIZLI))}

print("--- BatchNorm'lu model egitiliyor (15.000 adim) ---")
gr = torch.Generator().manual_seed(1)
for adim in range(15000):
    ix = torch.randint(0, Xtr.shape[0], (64,), generator=gr)
    on = C[Xtr[ix]].view(-1, BAGLAM * EMB) @ W1
    ort, std = on.mean(0, keepdim=True), on.std(0, keepdim=True)
    with torch.no_grad():
        durum['ort'] = 0.999 * durum['ort'] + 0.001 * ort
        durum['std'] = 0.999 * durum['std'] + 0.001 * std
    on = kazanc * (on - ort) / (std + 1e-5) + kaydirma
    loss = F.cross_entropy(torch.tanh(on) @ W2 + b2, Ytr[ix])
    for p in ps: p.grad = None
    loss.backward()
    lr = 0.1 if adim < 10500 else 0.01
    for p in ps: p.data += -lr * p.grad
print("egitim bitti, son minibatch loss =", round(loss.item(), 4))
print()


# ---------- 1) BATCHNORM'LU HALIYLE TAHMIN ----------
def ileri_batchnorm(Xb):
    on = C[Xb].view(-1, BAGLAM * EMB) @ W1
    on = kazanc * (on - durum['ort']) / (durum['std'] + 1e-5) + kaydirma
    return torch.tanh(on) @ W2 + b2


# ---------- 2) BATCHNORM'U W1 VE b1'E KATLA ----------
# BatchNorm tahmin zamani sunu yapiyor:
#   cikti = kazanc * (x@W1 - ort) / std + kaydirma
# Dagitirsak:
#   cikti = x @ (W1 * kazanc/std) + (kaydirma - ort*kazanc/std)
# Yani sabit bir W1_yeni ve b1_yeni'ye donusuyor.
with torch.no_grad():
    olcek = kazanc / (durum['std'] + 1e-5)     # sekil (1, GIZLI)
    W1_katlanmis = W1 * olcek                   # her sutun ayri olcekleniyor
    b1_katlanmis = (kaydirma - durum['ort'] * olcek).flatten()


def ileri_katlanmis(Xb):
    on = C[Xb].view(-1, BAGLAM * EMB) @ W1_katlanmis + b1_katlanmis
    return torch.tanh(on) @ W2 + b2


# ---------- 3) IKISI AYNI MI ----------
print("--- KARSILASTIRMA ---")
with torch.no_grad():
    a = ileri_batchnorm(Xdev)
    b = ileri_katlanmis(Xdev)
    fark = (a - b).abs().max().item()
    la = F.cross_entropy(a, Ydev).item()
    lb = F.cross_entropy(b, Ydev).item()

print(f"  BatchNorm'lu dev loss  : {la:.6f}")
print(f"  Katlanmis dev loss     : {lb:.6f}")
print(f"  logits'ler arasi maksimum fark : {fark:.2e}")
print()
print("  Fark makine hassasiyeti seviyesinde -> forward pass ayni kaldi.")
print()

print("--- NE KAZANDIK ---")
print("  Once  : matris carpimi + ortalama cikarma + bolme + carpma + toplama")
print("  Sonra : sadece matris carpimi + bias toplama")
print()
print("  BatchNorm katmani tahmin zamaninda tamamen ORTADAN KALKTI.")
print("  Model kucuk ve hizli hale geldi, sonuc birebir ayni.")
print()
print("  Bu, gercek uretim sistemlerinde yapilan standart bir optimizasyon.")
print("  Sebebi basit: tahmin zamani BatchNorm sabit bir dogrusal donusum,")
print("  ve iki dogrusal donusum tek bir dogrusal donusumde birlestirilebilir.")
