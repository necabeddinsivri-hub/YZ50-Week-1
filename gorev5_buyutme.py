# Gorev 5 - Modeli buyut, uc modeli tek tabloda karsilastir
# YZ50 Hafta 6 (46:07 - 46:58)

import torch
from katmanlar import (Linear, BatchNorm1d, Tanh, Embedding, Flatten,
                       FlattenConsecutive, Sequential)
from ortak import bolumle, egit, degerlendir, isim_uret, son_katmani_kucult, parametre_sayisi

EMB, ADIM = 24, 60000       # embedding 10 -> 24


def duz_mlp(baglam, V, gizli=200):
    torch.manual_seed(42)
    m = Sequential([
        Embedding(V, EMB), Flatten(),
        Linear(baglam * EMB, gizli, bias=False), BatchNorm1d(gizli), Tanh(),
        Linear(gizli, V),
    ])
    son_katmani_kucult(m)
    return m


def wavenet(V, gizli=128):
    torch.manual_seed(42)
    m = Sequential([
        Embedding(V, EMB),
        FlattenConsecutive(2), Linear(EMB*2, gizli, bias=False),   BatchNorm1d(gizli), Tanh(),
        FlattenConsecutive(2), Linear(gizli*2, gizli, bias=False), BatchNorm1d(gizli), Tanh(),
        FlattenConsecutive(2), Linear(gizli*2, gizli, bias=False), BatchNorm1d(gizli), Tanh(),
        Linear(gizli, V),
    ])
    son_katmani_kucult(m)
    return m


deneyler = [
    ("baglam 3, duz MLP",  3, lambda V: duz_mlp(3, V)),
    ("baglam 8, duz MLP",  8, lambda V: duz_mlp(8, V)),
    ("baglam 8, WaveNet",  8, lambda V: wavenet(V)),
]

tablo = []
for ad, baglam, kur in deneyler:
    (Xtr, Ytr), (Xdev, Ydev), _, stoi, itos = bolumle('names.txt', baglam)
    m = kur(len(stoi))
    print(f"--- {ad}  ({parametre_sayisi(m)} parametre) ---")
    egit(m, Xtr, Ytr, adim=ADIM, sessiz=True)
    tr, dv = degerlendir(m, Xtr, Ytr), degerlendir(m, Xdev, Ydev)
    isimler = isim_uret(m, itos, baglam, 8)
    tablo.append((ad, parametre_sayisi(m), tr, dv, isimler))
    print(f"    train = {tr:.4f}   dev = {dv:.4f}")
    print(f"    isimler: {', '.join(isimler)}")
    print()

print("=" * 62)
print(f"KARSILASTIRMA TABLOSU  (embedding = {EMB}, {ADIM} adim)")
print("=" * 62)
print(f"{'model':<22}{'parametre':>12}{'train':>10}{'dev loss':>12}")
print("-" * 62)
for ad, n, tr, dv, _ in tablo:
    print(f"{ad:<22}{n:>12}{tr:>10.4f}{dv:>12.4f}")
print("-" * 62)
print()
d3, d8, dw = tablo[0][3], tablo[1][3], tablo[2][3]
print(f"baglami 3'ten 8'e cikarmak (duz)     : {d3 - d8:+.4f}")
print(f"ayni 8 harfi hiyerarsik birlestirmek : {d8 - dw:+.4f}")
print(f"toplam kazanc                        : {d3 - dw:+.4f}")
print()
print("Referans: Karpathy'nin videodaki sonucu 1.993 (200.000 adim).")
