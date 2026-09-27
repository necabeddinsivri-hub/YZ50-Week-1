# Gorev 2 - Baglami 3'ten 8'e cikar, baska hicbir sey degismesin
# YZ50 Hafta 6 (17:11 - 21:36)
#
# Bu, karsilastirma tabani. WaveNet'in ne kazandirdigini olcmek icin
# once "sadece baglami buyutmek" ne kazandiriyor onu gormem lazim.

import torch
from katmanlar import Linear, BatchNorm1d, Tanh, Embedding, Flatten, Sequential
from ortak import bolumle, egit, degerlendir, isim_uret, son_katmani_kucult, parametre_sayisi

EMB, GIZLI, ADIM = 10, 200, 30000
sonuclar = {}

for BAGLAM in [3, 8]:
    (Xtr, Ytr), (Xdev, Ydev), _, stoi, itos = bolumle('names.txt', BAGLAM)
    V = len(stoi)
    torch.manual_seed(42)
    model = Sequential([
        Embedding(V, EMB),
        Flatten(),
        Linear(BAGLAM * EMB, GIZLI, bias=False), BatchNorm1d(GIZLI), Tanh(),
        Linear(GIZLI, V),
    ])
    son_katmani_kucult(model)
    print(f"--- baglam = {BAGLAM}, parametre = {parametre_sayisi(model)} ---")
    egit(model, Xtr, Ytr, adim=ADIM, sessiz=True)
    tr, dv = degerlendir(model, Xtr, Ytr), degerlendir(model, Xdev, Ydev)
    sonuclar[BAGLAM] = (parametre_sayisi(model), tr, dv, isim_uret(model, itos, BAGLAM, 8))
    print(f"    train = {tr:.4f}   dev = {dv:.4f}")
    print(f"    isimler: {', '.join(sonuclar[BAGLAM][3])}")
    print()

p3, _, d3, _ = sonuclar[3]
p8, _, d8, _ = sonuclar[8]
print("=" * 56)
print("KARSILASTIRMA")
print("=" * 56)
print(f"{'':<14}{'parametre':>12}{'dev loss':>12}")
print(f"{'baglam 3':<14}{p3:>12}{d3:>12.4f}")
print(f"{'baglam 8':<14}{p8:>12}{d8:>12.4f}")
print(f"{'fark':<14}{p8-p3:>+12}{d8-d3:>+12.4f}")
print()
print(f"Parametre {p8/p3:.2f} kat artti (+{p8-p3}), dev loss {d3-d8:.4f} dustu.")
print()
print("Neden parametre bu kadar artti? Degisen tek katman ilk Linear:")
print(f"  baglam 3: 3*10 = 30 giris  x 200 = {30*200} agirlik")
print(f"  baglam 8: 8*10 = 80 giris  x 200 = {80*200} agirlik")
print("Sekiz harfin hepsi TEK katmanda, TEK seferde ezilip 200 sayiya")
print("sikistiriliyor. Harfler arasindaki yapiyi ogrenecek ara adim yok.")
print("Gorev 3'teki WaveNet tam bu sorunu cozuyor.")
