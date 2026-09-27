# Gorev 3 - WaveNet: 8 harfi ikiser ikiser, uc katmanda birlestirmek
# YZ50 Hafta 6 (21:36 - 37:41)
#
# Duz modelde 8 harf tek seferde ezilip 200 sayiya sikistiriliyordu.
# WaveNet'te once komsu harfler ikiser ikiser birlesiyor, sonra o
# ciftler ikiser ikiser, sonra o dortluler. Uc adimda tek vektore.
#
# NOT: Bu dosyada BatchNorm'u videodaki gibi HATALI haliyle kullaniyorum
# (duzeltilmis=False). Sekilleri yazdirirken hatayi gorecegiz, Gorev 4'te
# duzeltecegiz.

import torch
from katmanlar import (Linear, BatchNorm1d, Tanh, Embedding,
                       FlattenConsecutive, Sequential)
from ortak import bolumle, egit, degerlendir, isim_uret, son_katmani_kucult, parametre_sayisi

BAGLAM, EMB, GIZLI = 8, 10, 68       # 68: parametre sayisi duz modele yakin olsun diye
(Xtr, Ytr), (Xdev, Ydev), _, stoi, itos = bolumle('names.txt', BAGLAM)
V = len(stoi)

torch.manual_seed(42)
model = Sequential([
    Embedding(V, EMB),
    FlattenConsecutive(2), Linear(EMB * 2, GIZLI, bias=False),   BatchNorm1d(GIZLI, duzeltilmis=False), Tanh(),
    FlattenConsecutive(2), Linear(GIZLI * 2, GIZLI, bias=False), BatchNorm1d(GIZLI, duzeltilmis=False), Tanh(),
    FlattenConsecutive(2), Linear(GIZLI * 2, GIZLI, bias=False), BatchNorm1d(GIZLI, duzeltilmis=False), Tanh(),
    Linear(GIZLI, V),
])
son_katmani_kucult(model)
print(f"parametre sayisi: {parametre_sayisi(model)}  (duz baglam-8 MLP: 22097)")
print()


# ==========================================================
# SEKILLERI TAKIP ET: 4 orneklik kucuk bir batch gecir
# ==========================================================
ix = torch.randint(0, Xtr.shape[0], (4,))
model(Xtr[ix])

aciklamalar = {
    'Embedding':  "4 isim, 8 harf, her harf 10 sayi. Henuz hicbir sey birlesmedi.",
    'Flatten1':   "Komsu harfler ikiser birlesti: 8 harf -> 4 cift, her cift 2x10=20 sayi.",
    'Linear1':    "Her cift ayri ayri 20 -> 68'e donustu. Ayni agirliklar 4 cifte de uygulaniyor.",
    'BN1':        "Sekil degismez, sadece normalize eder.",
    'Tanh1':      "Sekil degismez.",
    'Flatten2':   "Ciftler ikiser birlesti: 4 cift -> 2 dortlu, her dortlu 2x68=136 sayi.",
    'Linear2':    "Her dortlu 136 -> 68'e donustu.",
    'BN2':        "Sekil degismez.",
    'Tanh2':      "Sekil degismez.",
    'Flatten3':   "Dortluler birlesti: 2 dortlu -> 1 sekizli. Orta eksen 1 oldugu icin atildi.",
    'Linear3':    "Artik 8 harfin tamami tek bir 68'lik vektorde.",
    'BN3':        "2 boyutlu girdi, burada BatchNorm dogru calisiyor.",
    'Tanh3':      "Sekil degismez.",
    'Linear4':    "Cikis: 27 harf icin ham skor (logits).",
}
sayac = {}
print("=" * 78)
print(f"{'katman':<12}{'cikti sekli':<16}aciklama")
print("=" * 78)
for katman in model.layers:
    ad = katman.__class__.__name__
    kisa = {'Embedding': 'Embedding', 'FlattenConsecutive': 'Flatten',
            'Linear': 'Linear', 'BatchNorm1d': 'BN', 'Tanh': 'Tanh'}[ad]
    sayac[kisa] = sayac.get(kisa, 0) + 1
    anahtar = kisa if kisa == 'Embedding' else f"{kisa}{sayac[kisa]}"
    print(f"{anahtar:<12}{str(tuple(katman.out.shape)):<16}{aciklamalar[anahtar]}")
print()

print("KENDI CUMLELERIMLE:")
print("  Duz modelde 8 harf ilk katmanda birden ezilip 200 sayiya iniyordu.")
print("  Burada her katman sadece IKI komsu seyi birlestiriyor. Ilk katman")
print("  harf ciftlerini ('em', 'ma'), ikinci katman hece benzeri dortluleri,")
print("  ucuncu katman butun baglami goruyor. Bilgi bir agac gibi yukari")
print("  cikiyor ve her seviyede biraz daha soyutlasiyor.")
print()
print("  Onemli nokta: Linear katman 3 boyutlu girdiye uygulandiginda son")
print("  eksen uzerinden calisiyor, ortadaki eksen (4, 2...) 'toplu islem'")
print("  gibi davranıyor. Yani AYNI agirlik matrisi her cifte ayri ayri")
print("  uygulaniyor. Bu, parametre paylasimi demek.")
print()


# ==========================================================
# BATCHNORM'UN CALISAN ISTATISTIKLERINE BAK
# ==========================================================
print("=" * 78)
print("DIKKAT: BatchNorm'un calisan ortalamasinin sekli")
print("=" * 78)
for i, katman in enumerate(model.layers):
    if isinstance(katman, BatchNorm1d):
        print(f"  katman {i:>2}: running_mean sekli = {tuple(katman.running_mean.shape)}")
print()
print("  Beklenen: (1, 1, 68) veya (68,) -- her kanal icin TEK ortalama.")
print("  Gorulen:  ilk BatchNorm'da (1, 4, 68)!")
print("  Yani 4 cift pozisyonunun her biri icin AYRI ortalama tutuluyor.")
print("  Bu bir hata. Gorev 4'te nedenini ve duzeltmesini gosteriyorum.")
print()


# ==========================================================
# EGITIM (hatali BatchNorm ile)
# ==========================================================
print("--- egitim (30.000 adim, HATALI BatchNorm ile) ---")
egit(model, Xtr, Ytr, adim=30000)
tr, dv = degerlendir(model, Xtr, Ytr), degerlendir(model, Xdev, Ydev)
print(f"\ntrain = {tr:.4f}   dev = {dv:.4f}")
print("isimler:", ", ".join(isim_uret(model, itos, BAGLAM, 8)))
