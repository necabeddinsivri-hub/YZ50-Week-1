# Gorev 4 - BatchNorm1d'nin 3 boyutlu girdide yanlis calismasi
# YZ50 Hafta 6 (37:41 - 46:07)

import torch
from katmanlar import (Linear, BatchNorm1d, Tanh, Embedding,
                       FlattenConsecutive, Sequential)
from ortak import bolumle, egit, degerlendir, son_katmani_kucult, parametre_sayisi

BAGLAM, EMB, GIZLI, ADIM = 8, 10, 68, 30000
(Xtr, Ytr), (Xdev, Ydev), _, stoi, itos = bolumle('names.txt', BAGLAM)
V = len(stoi)


def wavenet(duzeltilmis):
    torch.manual_seed(42)
    m = Sequential([
        Embedding(V, EMB),
        FlattenConsecutive(2), Linear(EMB*2, GIZLI, bias=False),   BatchNorm1d(GIZLI, duzeltilmis=duzeltilmis), Tanh(),
        FlattenConsecutive(2), Linear(GIZLI*2, GIZLI, bias=False), BatchNorm1d(GIZLI, duzeltilmis=duzeltilmis), Tanh(),
        FlattenConsecutive(2), Linear(GIZLI*2, GIZLI, bias=False), BatchNorm1d(GIZLI, duzeltilmis=duzeltilmis), Tanh(),
        Linear(GIZLI, V),
    ])
    son_katmani_kucult(m)
    return m


# ==========================================================
# 1) HATAYI GOSTER
# ==========================================================
print("=" * 70)
print("1) HATA NE")
print("=" * 70)
print()
x = torch.randn(32, 4, 68)     # ilk BatchNorm'a giren tensor'un sekli
print(f"BatchNorm'a giren tensor: {tuple(x.shape)}  = (batch, pozisyon, kanal)")
print()
yanlis = x.mean(0, keepdim=True)
dogru = x.mean((0, 1), keepdim=True)
print(f"  x.mean(0)       -> sekil {tuple(yanlis.shape)}  <- HATALI")
print(f"  x.mean((0, 1))  -> sekil {tuple(dogru.shape)}  <- DOGRU")
print()
print("KENDI CUMLELERIMLE:")
print("  Hafta 4'te BatchNorm'a hep 2 boyutlu tensor giriyordu: (batch, kanal).")
print("  Ortalamayi 0. eksende almak = her kanal icin batch uzerinden ortalama.")
print("  Dogruydu.")
print()
print("  WaveNet'te tensor 3 boyutlu oldu: (batch, pozisyon, kanal).")
print("  Kodu degistirmeden mean(0) dersek, ortalama sadece batch uzerinden")
print("  aliniyor ve 4 pozisyonun HER BIRI icin ayri ortalama cikiyor.")
print("  Yani 'ilk cift', 'ikinci cift', ... ayri ayri normalize ediliyor.")
print()
print("  Oysa bu 4 pozisyon ayni Linear katmandan, ayni agirliklarla geciyor.")
print("  Ayni 68 kanal, ayni anlam. Hepsinin ortak istatistigi olmali.")
print("  Dogru olan: hem batch (0) hem pozisyon (1) ekseninde ortalama almak.")
print("  O zaman her kanal icin 32 x 4 = 128 ornek uzerinden TEK ortalama cikar.")
print()


# ==========================================================
# 2) NEDEN SESSIZ BIR HATA
# ==========================================================
print("=" * 70)
print("2) NEDEN FARK EDILMIYOR")
print("=" * 70)
print()
print("  Kod hata vermiyor. Broadcasting (1,4,68) seklindeki ortalamayi")
print("  (32,4,68) seklindeki tensor'dan sorunsuz cikariyor. Model egitiliyor,")
print("  loss dusuyor. Tek iz: running_mean'in sekli (1,4,68) oluyor, (1,1,68)")
print("  olmasi gerekirken. Sekli yazdirmasaydim hic gormezdim.")
print()
print("  Hafta 3'teki keepdim tuzagi ve Hafta 2'deki zero-grad bug'i ile ayni")
print("  aileden: calisan ama yanlis sonuc veren sessiz hatalar.")
print()


# ==========================================================
# 3) IKISINI EGIT, KARSILASTIR
# ==========================================================
print("=" * 70)
print("3) DUZELTMEDEN ONCE VE SONRA")
print("=" * 70)
print()
sonuc = {}
for duz in [False, True]:
    m = wavenet(duz)
    etiket = "duzeltilmis (0,1)" if duz else "hatali (0)"
    print(f"--- {etiket} ---")
    egit(m, Xtr, Ytr, adim=ADIM, sessiz=True)
    tr, dv = degerlendir(m, Xtr, Ytr), degerlendir(m, Xdev, Ydev)
    rm = [tuple(l.running_mean.shape) for l in m.layers if isinstance(l, BatchNorm1d)]
    sonuc[duz] = (tr, dv, rm)
    print(f"    train = {tr:.4f}   dev = {dv:.4f}")
    print(f"    running_mean sekilleri: {rm}")
    print()

print(f"{'':<22}{'train':>10}{'dev':>10}")
print(f"{'hatali BatchNorm':<22}{sonuc[False][0]:>10.4f}{sonuc[False][1]:>10.4f}")
print(f"{'duzeltilmis BatchNorm':<22}{sonuc[True][0]:>10.4f}{sonuc[True][1]:>10.4f}")
fark = sonuc[False][1] - sonuc[True][1]
print(f"{'iyilesme':<22}{'':>10}{fark:>+10.4f}")
print()
print("YORUM:")
print("  Duzeltme kucuk ama olculebilir bir fark yaratiyor. Neden kucuk?")
print("  Cunku hatali versiyon da bir tur normalizasyon yapiyordu, sadece")
print("  istatistikleri daha az ornekten (32 yerine) ve pozisyon bazinda")
print("  tahmin ediyordu. Duzeltilmis versiyon her kanal icin 4 kat fazla")
print("  ornek (128) kullaniyor, yani ortalama ve varyans tahmini daha")
print("  istikrarli. Ayrica tahmin zamani calisan istatistikler dogru")
print("  seklinde, butun pozisyonlar icin ortak.")
