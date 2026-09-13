# Gorev 5 - Baslangic loss'u neden yuksek, tanh neden doyuyor, Kaiming init
# YZ50 Hafta 4 (Part 3, 4:19 - 40:40)

import math
import torch
import torch.nn.functional as F
from ortak import veri_yukle, veri_kur, bol

kelimeler, stoi, itos = veri_yukle('names.txt')
BAGLAM, EMB, GIZLI = 3, 10, 200
train, dev, test = bol(kelimeler)
Xtr, Ytr = veri_kur(train, stoi, BAGLAM)
Xdev, Ydev = veri_kur(dev, stoi, BAGLAM)


def model_kur(w1_olcek, b2_olcek, w2_olcek, tohum=2147483647):
    g = torch.Generator().manual_seed(tohum)
    C = torch.randn((27, EMB), generator=g)
    W1 = torch.randn((BAGLAM * EMB, GIZLI), generator=g) * w1_olcek
    b1 = torch.randn(GIZLI, generator=g) * 0.01
    W2 = torch.randn((GIZLI, 27), generator=g) * w2_olcek
    b2 = torch.randn(27, generator=g) * b2_olcek
    ps = [C, W1, b1, W2, b2]
    for p in ps: p.requires_grad = True
    return ps


def ileri(ps, Xb, h_dondur=False):
    C, W1, b1, W2, b2 = ps
    emb = C[Xb]
    on_aktivasyon = emb.view(-1, BAGLAM * EMB) @ W1 + b1
    h = torch.tanh(on_aktivasyon)
    logits = h @ W2 + b2
    if h_dondur:
        return logits, h, on_aktivasyon
    return logits


# ==========================================================
# 5a) BASLANGIC LOSS'U NEDEN COK YUKSEK
# ==========================================================
print("=" * 62)
print("5a) BASLANGIC LOSS'U")
print("=" * 62)
print()
print("Model hicbir sey bilmiyorken NE KADAR loss beklemeliyiz?")
print("27 harf var, hicbir fikri yoksa hepsine esit 1/27 olasilik verir.")
print(f"  beklenen loss = -log(1/27) = {-math.log(1/27):.4f}")
print()

ps_kotu = model_kur(w1_olcek=1.0, b2_olcek=1.0, w2_olcek=1.0)
with torch.no_grad():
    logits = ileri(ps_kotu, Xtr[:32])
    L = F.cross_entropy(logits, Ytr[:32]).item()
print(f"  KOTU init ile gercek loss = {L:.4f}   <- 6 kat fazla")
print()
print("  Sebep: son katmanin cikislari rastgele buyuk. Model 'a harfi")
print("  kesinlikle gelmez, z kesinlikle gelir' gibi guclu ama YANLIS")
print("  iddialarda bulunuyor. Ilk adimlar bu iddialari sondurmekle geciyor.")
print()
print("  ilk 5 logit (kotu init):", [round(v, 2) for v in logits[0][:5].tolist()])
print()

ps_iyi = model_kur(w1_olcek=1.0, b2_olcek=0.0, w2_olcek=0.01)
with torch.no_grad():
    logits2 = ileri(ps_iyi, Xtr[:32])
    L2 = F.cross_entropy(logits2, Ytr[:32]).item()
print(f"  IYI init (son katman ~0) ile loss = {L2:.4f}   <- beklenen degere cok yakin")
print("  ilk 5 logit (iyi init):", [round(v, 2) for v in logits2[0][:5].tolist()])
print()


# ==========================================================
# 5b) TANH SATURATION
# ==========================================================
print("=" * 62)
print("5b) TANH DOYMASI (SATURATION)")
print("=" * 62)
print()


def histogram(h, baslik):
    """tanh ciktilarinin dagilimini metin histogramiyla goster"""
    h = h.detach().flatten()
    kenarlar = [-1.0, -0.99, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 1.0]
    print(baslik)
    for i in range(len(kenarlar) - 1):
        a, b = kenarlar[i], kenarlar[i + 1]
        adet = ((h >= a) & (h < b)).sum().item()
        oran = adet / h.nelement()
        cubuk = '#' * int(oran * 100)
        print(f"  [{a:>5.2f},{b:>5.2f})  {oran*100:>5.1f}%  {cubuk}")
    doymus = (h.abs() > 0.99).float().mean().item()
    print(f"  --> |h| > 0.99 olan oran: {doymus*100:.1f}%")
    print()
    return doymus


with torch.no_grad():
    _, h_kotu, on_kotu = ileri(ps_kotu, Xtr[:1000], h_dondur=True)
d1 = histogram(h_kotu, "KOTU init (W1 olcek = 1.0):")

print("  Neden onemli? tanh'in turevi 1 - h^2.")
print("  h = 0.99 ise turev = 1 - 0.98 = 0.02  -> gradient neredeyse yok")
print("  h = 0.00 ise turev = 1 - 0.00 = 1.00  -> gradient tam gecer")
print()
print("  Hafta 2'de kendi micrograd'imda '1 - t**2' satirini yazmistim.")
print("  Iste o satir burada isliyor: doymus neuron ogrenmiyor.")
print()

# Kaiming init
kaiming = (5/3) / math.sqrt(BAGLAM * EMB)
print(f"  Kaiming init olcegi = gain / sqrt(fan_in) = (5/3) / sqrt({BAGLAM*EMB}) = {kaiming:.4f}")
print()

ps_kaiming = model_kur(w1_olcek=kaiming, b2_olcek=0.0, w2_olcek=0.01)
with torch.no_grad():
    _, h_iyi, on_iyi = ileri(ps_kaiming, Xtr[:1000], h_dondur=True)
d2 = histogram(h_iyi, "KAIMING init (W1 olcek = 0.30):")

print(f"  doymus neuron orani: {d1*100:.1f}%  ->  {d2*100:.1f}%")
print()


# ==========================================================
# 5c) IKI INIT'IN EGITIM SONUCU
# ==========================================================
print("=" * 62)
print("5c) IKI INIT'IN EGITIM SONUCU (20.000 adim)")
print("=" * 62)
print()


def egit(ps, adim_sayisi=20000, kayit=False):
    gr = torch.Generator().manual_seed(1)
    ilk_loss = None
    for adim in range(adim_sayisi):
        ix = torch.randint(0, Xtr.shape[0], (64,), generator=gr)
        loss = F.cross_entropy(ileri(ps, Xtr[ix]), Ytr[ix])
        if adim == 0:
            ilk_loss = loss.item()
        for p in ps: p.grad = None
        loss.backward()
        lr = 0.1 if adim < adim_sayisi * 0.7 else 0.01
        for p in ps: p.data += -lr * p.grad
    with torch.no_grad():
        tr = F.cross_entropy(ileri(ps, Xtr), Ytr).item()
        dv = F.cross_entropy(ileri(ps, Xdev), Ydev).item()
    return ilk_loss, tr, dv


print(f"{'init':<22}{'ilk loss':>10}{'train':>9}{'dev':>9}")
for isim, ps in [("kotu (hepsi rastgele)", model_kur(1.0, 1.0, 1.0)),
                 ("sadece son katman ~0", model_kur(1.0, 0.0, 0.01)),
                 ("Kaiming + son katman", model_kur(kaiming, 0.0, 0.01))]:
    il, tr, dv = egit(ps)
    print(f"{isim:<22}{il:>10.4f}{tr:>9.4f}{dv:>9.4f}")
print()
print("Not: bu kucuk agda fark buyuk gorunmuyor cunku tek gizli katman var.")
print("Katman sayisi arttikca bu farklar birikir ve derin aglarda kritik olur.")
