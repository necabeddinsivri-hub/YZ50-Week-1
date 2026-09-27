# Gorev 7 (ek, secenek a) - Ayni hiyerarsik yapiyi Conv1d ile kurmak
# YZ50 Hafta 6 (47:44 - 51:34)
#
# Fikir: FlattenConsecutive(2) + Linear, aslinda kernel boyutu 2 olan
# bir konvolusyondur. Egitilmis WaveNet'in agirliklarini Conv1d'ye
# kopyalayip iki modelin AYNI ciktiyi verdigini gosteriyorum.
# Sonra dilated (genisletilmis) konvolusyonla bir ismin butun
# pozisyonlarini TEK geciste hesapliyorum.

import torch
import torch.nn.functional as F
from katmanlar import (Linear, BatchNorm1d, Tanh, Embedding,
                       FlattenConsecutive, Sequential)
from ortak import bolumle, egit, degerlendir, son_katmani_kucult, veri_yukle

BAGLAM, EMB, GIZLI = 8, 10, 68
(Xtr, Ytr), (Xdev, Ydev), _, stoi, itos = bolumle('names.txt', BAGLAM)
V = len(stoi)

torch.manual_seed(42)
model = Sequential([
    Embedding(V, EMB),
    FlattenConsecutive(2), Linear(EMB*2, GIZLI, bias=False),   BatchNorm1d(GIZLI), Tanh(),
    FlattenConsecutive(2), Linear(GIZLI*2, GIZLI, bias=False), BatchNorm1d(GIZLI), Tanh(),
    FlattenConsecutive(2), Linear(GIZLI*2, GIZLI, bias=False), BatchNorm1d(GIZLI), Tanh(),
    Linear(GIZLI, V),
])
son_katmani_kucult(model)
print("--- WaveNet egitiliyor (20.000 adim) ---")
egit(model, Xtr, Ytr, adim=20000, sessiz=True)
print(f"dev loss = {degerlendir(model, Xdev, Ydev):.4f}")
print()
model.egitim_kipi(False)

C = model.layers[0].weight
lin = [model.layers[2], model.layers[6], model.layers[10]]
bns = [model.layers[3], model.layers[7], model.layers[11]]
son = model.layers[13]


def lin_to_conv(W, c_in):
    """Linear (2*c_in, c_out) -> Conv1d agirligi (c_out, c_in, 2)
    Birlesik vektor [x_sol, x_sag] @ W  =  x_sol @ W[:c_in] + x_sag @ W[c_in:]"""
    return torch.stack([W[:c_in].T, W[c_in:].T], dim=2)


conv_w = [lin_to_conv(lin[0].weight, EMB),
          lin_to_conv(lin[1].weight, GIZLI),
          lin_to_conv(lin[2].weight, GIZLI)]


def bn_eval(x_BCT, bn):
    # BatchNorm tahmin kipinde kanal basina sabit bir olcekleme
    mean = bn.running_mean.reshape(1, -1, 1)
    var = bn.running_var.reshape(1, -1, 1)
    g = bn.gamma.reshape(1, -1, 1)
    b = bn.beta.reshape(1, -1, 1)
    return g * (x_BCT - mean) / torch.sqrt(var + bn.eps) + b


# ==========================================================
# 1) stride=2 konvolusyon == bizim agac (ayni pencereler)
# ==========================================================
@torch.no_grad()
def conv_agac(X):
    x = C[X].transpose(1, 2)              # (B, EMB, 8)  Conv1d kanal-once ister
    for w, bn in zip(conv_w, bns):
        x = F.conv1d(x, w, stride=2)      # uzunluk 8 -> 4 -> 2 -> 1
        x = torch.tanh(bn_eval(x, bn))
    return x.squeeze(2) @ son.weight + son.bias


with torch.no_grad():
    a = model(Xdev[:2000])
    b = conv_agac(Xdev[:2000])
print("=" * 64)
print("1) Conv1d (kernel=2, stride=2) ile ayni agac")
print("=" * 64)
print(f"  kendi WaveNet'im vs Conv1d, 2000 dev ornegi")
print(f"  logits'ler arasi maksimum fark: {(a - b).abs().max().item():.2e}")
print(f"  dev loss (Conv1d ile): {F.cross_entropy(conv_agac(Xdev), Ydev).item():.4f}")
print()
print("  Yani FlattenConsecutive(2) + Linear, kernel 2 stride 2 bir")
print("  konvolusyonun ta kendisi. Ayni agirliklar, ayni sonuc.")
print()


# ==========================================================
# 2) DILATED konvolusyon: bir ismin butun pozisyonlari tek geciste
# ==========================================================
@torch.no_grad()
def conv_dilated(dizi):
    """dizi: (T,) harf indeksleri, basinda 8 nokta var.
    Katmanlar: dilation 1, 2, 4. Her cikis pozisyonu kendi 8'lik
    penceresini goruyor, pencereler paylasilan ara sonuclari tekrar
    kullaniyor."""
    x = C[dizi].T.unsqueeze(0)            # (1, EMB, T)
    for w, bn, d in zip(conv_w, bns, [1, 2, 4]):
        x = F.conv1d(x, w, dilation=d)
        x = torch.tanh(bn_eval(x, bn))
    return (x.squeeze(0).T) @ son.weight + son.bias     # (T-7, V)


kelimeler, _, _ = veri_yukle('names.txt')
isim = 'sophia'
dizi = torch.tensor([0] * BAGLAM + [stoi[h] for h in isim])
tek_gecis = conv_dilated(dizi)

pencereler = torch.stack([dizi[t:t + BAGLAM] for t in range(len(isim) + 1)])
with torch.no_grad():
    tek_tek = model(pencereler)

print("=" * 64)
print(f"2) Dilated Conv1d: '{isim}' isminin butun pozisyonlari tek geciste")
print("=" * 64)
print(f"  tahmin edilecek pozisyon sayisi : {len(isim) + 1} (her harf + bitis)")
print(f"  kendi modelim: {len(isim) + 1} ayri forward pass")
print(f"  dilated Conv1d: 1 forward pass")
print(f"  maksimum fark: {(tek_gecis - tek_tek).abs().max().item():.2e}")
print()

# hesap tasarrufu: ilk katmanda kac kez 2'li birlestirme yapiliyor
P = len(isim) + 1
agac_islem = P * (4 + 2 + 1)
dil_islem = (len(dizi) - 1) + (len(dizi) - 3) + (len(dizi) - 7)
print(f"  Linear/conv uygulama sayisi -> tek tek: {agac_islem},  dilated: {dil_islem}")
print()
print("KENDI CUMLELERIMLE:")
print("  Tek tek calistirinca komsu pencereler ayni ara sonuclari (ornegin")
print("  'ph' ciftinin temsilini) tekrar tekrar hesapliyor. Dilated")
print("  konvolusyon her ara sonucu bir kez hesaplayip butun pencerelerle")
print("  paylasiyor. Sonuc birebir ayni, ama is daha az. WaveNet makalesinin")
print("  ses uretiminde bunu kullanmasinin sebebi bu.")
