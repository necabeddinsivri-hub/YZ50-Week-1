# Gorev 6 - Ayni WaveNet'i Turkce isimlerle egitmek
# YZ50 Hafta 6
#
# Hafta 4'le adil karsilastirma icin ayni veri alt kumesini kullaniyorum:
# listenin her 4. ismi (59.024 isim), ayni karistirma tohumu (42).

import torch
from katmanlar import (Linear, BatchNorm1d, Tanh, Embedding, Flatten,
                       FlattenConsecutive, Sequential)
from ortak import bolumle, egit, degerlendir, isim_uret, son_katmani_kucult, parametre_sayisi

EMB, ADIM = 24, 60000


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


sonuc = {}
for ad, baglam, kur in [("baglam 3, duz MLP", 3, lambda V: duz_mlp(3, V)),
                        ("baglam 8, WaveNet", 8, lambda V: wavenet(V))]:
    (Xtr, Ytr), (Xdev, Ydev), _, stoi, itos = bolumle('isimler.txt', baglam, alt_kume=4)
    V = len(stoi)
    m = kur(V)
    print(f"--- {ad}  (alfabe {V}, {parametre_sayisi(m)} parametre, {Xtr.shape[0]} ornek) ---")
    egit(m, Xtr, Ytr, adim=ADIM, sessiz=True)
    tr, dv = degerlendir(m, Xtr, Ytr), degerlendir(m, Xdev, Ydev)
    isimler = isim_uret(m, itos, baglam, 20, tohum=42)
    sonuc[ad] = (parametre_sayisi(m), tr, dv, isimler)
    print(f"    train = {tr:.4f}   dev = {dv:.4f}")
    print()

print("=" * 60)
print("TURKCE KARSILASTIRMA")
print("=" * 60)
print(f"{'model':<30}{'parametre':>12}{'dev loss':>12}")
print(f"{'Hafta 3 bigram sayim':<30}{'729':>12}{2.5830:>12.4f}")
print(f"{'Hafta 4 MLP (baglam 3, emb 10)':<30}{'12530':>12}{2.3183:>12.4f}")
for ad, (n, tr, dv, _) in sonuc.items():
    print(f"{'Hafta 6 ' + ad:<30}{n:>12}{dv:>12.4f}")
print()

d3 = sonuc["baglam 3, duz MLP"][2]
d8 = sonuc["baglam 8, WaveNet"][2]
print(f"Ayni ayarlarda baglam 3 -> baglam 8 WaveNet kazanci: {d3 - d8:+.4f}")
print()

print("--- WaveNet'in urettigi 20 Turkce isim ---")
for i, isim in enumerate(sonuc["baglam 8, WaveNet"][3]):
    print(f"  {isim:<16}", end="\n" if i % 4 == 3 else "")
print()
print("--- karsilastirma: baglam 3 MLP'nin urettikleri ---")
print("  " + ", ".join(sonuc["baglam 3, duz MLP"][3][:12]))
