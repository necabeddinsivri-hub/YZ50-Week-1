# Gorev 6 - BatchNorm katmani
# YZ50 Hafta 4 (Part 3, 40:40 - 1:04:50)

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


def model_kur(batchnorm, tohum=2147483647):
    g = torch.Generator().manual_seed(tohum)
    C  = torch.randn((27, EMB), generator=g)
    W1 = torch.randn((BAGLAM * EMB, GIZLI), generator=g) * kaiming
    W2 = torch.randn((GIZLI, 27), generator=g) * 0.01
    b2 = torch.zeros(27)
    ps = [C, W1, W2, b2]
    # BatchNorm varsa b1'e gerek yok: normalize etme onu zaten siliyor
    if batchnorm:
        bn_kazanc = torch.ones((1, GIZLI))
        bn_kaydirma = torch.zeros((1, GIZLI))
        ps += [bn_kazanc, bn_kaydirma]
    else:
        b1 = torch.randn(GIZLI, generator=g) * 0.01
        ps += [b1]
    for p in ps: p.requires_grad = True
    # tahmin sirasinda kullanilacak calisan istatistikler
    durum = {'ort': torch.zeros((1, GIZLI)), 'std': torch.ones((1, GIZLI))}
    return ps, durum


def ileri(ps, durum, Xb, batchnorm, egitimde=True, h_dondur=False):
    if batchnorm:
        C, W1, W2, b2, bn_kazanc, bn_kaydirma = ps
    else:
        C, W1, W2, b2, b1 = ps

    emb = C[Xb]
    on = emb.view(-1, BAGLAM * EMB) @ W1

    if batchnorm:
        if egitimde:
            ort = on.mean(0, keepdim=True)
            std = on.std(0, keepdim=True)
            with torch.no_grad():          # calisan ortalamayi guncelle
                durum['ort'] = 0.999 * durum['ort'] + 0.001 * ort
                durum['std'] = 0.999 * durum['std'] + 0.001 * std
        else:
            ort, std = durum['ort'], durum['std']
        on = bn_kazanc * (on - ort) / (std + 1e-5) + bn_kaydirma
    else:
        on = on + b1

    h = torch.tanh(on)
    logits = h @ W2 + b2
    if h_dondur:
        return logits, h
    return logits


def egit(batchnorm, adim_sayisi=20000):
    ps, durum = model_kur(batchnorm)
    gr = torch.Generator().manual_seed(1)
    for adim in range(adim_sayisi):
        ix = torch.randint(0, Xtr.shape[0], (64,), generator=gr)
        loss = F.cross_entropy(ileri(ps, durum, Xtr[ix], batchnorm, True), Ytr[ix])
        for p in ps: p.grad = None
        loss.backward()
        lr = 0.1 if adim < adim_sayisi * 0.7 else 0.01
        for p in ps: p.data += -lr * p.grad
    with torch.no_grad():
        tr = F.cross_entropy(ileri(ps, durum, Xtr, batchnorm, False), Ytr).item()
        dv = F.cross_entropy(ileri(ps, durum, Xdev, batchnorm, False), Ydev).item()
    return ps, durum, tr, dv


print("=" * 62)
print("BATCHNORM NE YAPIYOR")
print("=" * 62)
print()
print("Gorev 5'te ogrendik: tanh'a giren sayilar cok buyukse neuron doyuyor.")
print("Kaiming init bunu BASLANGICTA duzeltiyor. Ama egitim ilerledikce")
print("agirliklar degisiyor ve dagilim yine kayabiliyor.")
print()
print("BatchNorm'un fikri: her adimda, tanh'a girmeden hemen once,")
print("sayilari ZORLA ortalamasi 0 standart sapmasi 1 olacak sekilde")
print("normalize et. Sonra ogrenilebilir bir kazanc ve kaydirma ekle,")
print("boylece ag isterse bu normalizasyonu kismen geri alabilsin.")
print()

# aktivasyon dagilimlarini karsilastir
def doymus_oran(ps, durum, batchnorm):
    with torch.no_grad():
        _, h = ileri(ps, durum, Xtr[:1000], batchnorm, True, h_dondur=True)
    return (h.abs() > 0.99).float().mean().item()


print("--- egitim ONCESI doymus neuron orani ---")
for bn in [False, True]:
    ps, durum = model_kur(bn)
    print(f"  BatchNorm {'VAR ' if bn else 'YOK '}: {doymus_oran(ps, durum, bn)*100:>5.1f}%")
print()

print("--- 20.000 adim egitim sonuclari ---")
print(f"{'model':<16}{'train':>9}{'dev':>9}{'egitim sonrasi doyma':>24}")
sonuc = {}
for bn in [False, True]:
    ps, durum, tr, dv = egit(bn)
    d = doymus_oran(ps, durum, bn)
    sonuc[bn] = (tr, dv)
    isim = "BatchNorm'lu" if bn else "BatchNorm'suz"
    print(f"{isim:<16}{tr:>9.4f}{dv:>9.4f}{d*100:>22.1f}%")
print()

fark = sonuc[False][1] - sonuc[True][1]
print(f"dev loss farki: {fark:+.4f}")
print()
print("Bu kucuk agda BatchNorm buyuk kazanc saglamiyor - beklenen bir sonuc,")
print("cunku tek gizli katman var ve Kaiming init zaten isi cozuyor. Asil")
print("degeri derin aglarda: 10 katmanda her katmanin dagilimini elle")
print("ayarlamak imkansiz, BatchNorm bunu otomatik yapiyor.")
print()

print("=" * 62)
print("BATCHNORM'UN TUHAF YANI")
print("=" * 62)
print()
print("Bir ornegin ciktisi, ayni batch'teki DIGER orneklere bagli hale geliyor.")
print("Kulaga bug gibi geliyor ama pratikte hafif bir gurultu (regularization)")
print("etkisi yaratip ezberlemeyi zorlastiriyor.")
print()
print("Tahmin zamani batch olmadigi icin egitim boyunca biriktirdigimiz")
print("'calisan ortalama' ve 'calisan std' kullaniliyor:")
print(f"  calisan ortalama ilk 5 deger: {[round(v,3) for v in durum['ort'][0][:5].tolist()]}")
print(f"  calisan std      ilk 5 deger: {[round(v,3) for v in durum['std'][0][:5].tolist()]}")
print()
print("Ayrica: BatchNorm varsa b1 bias'ina gerek yok. Ortalama cikarma")
print("islemi bias'i zaten siliyor. Kodda b1'i o yuzden kaldirdim.")
