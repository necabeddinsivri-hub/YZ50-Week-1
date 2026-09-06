# Gorev 3 - Negative Log Likelihood: modelin kalitesini tek sayiyla olcme
# YZ50 Hafta 3

import torch

kelimeler = open('names.txt', 'r').read().splitlines()

harfler = sorted(list(set(''.join(kelimeler))))
stoi = {h: i + 1 for i, h in enumerate(harfler)}
stoi['.'] = 0
itos = {i: h for h, i in stoi.items()}

N = torch.zeros((27, 27), dtype=torch.int32)
for k in kelimeler:
    hs = ['.'] + list(k) + ['.']
    for h1, h2 in zip(hs, hs[1:]):
        N[stoi[h1], stoi[h2]] += 1


def loss_hesapla(smoothing, isimler):
    """smoothing: her hucreye eklenen sahte sayim"""
    P = (N + smoothing).float()
    P = P / P.sum(1, keepdim=True)

    log_olasilik_toplami = 0.0
    adet = 0
    for k in isimler:
        hs = ['.'] + list(k) + ['.']
        for h1, h2 in zip(hs, hs[1:]):
            p = P[stoi[h1], stoi[h2]]
            log_olasilik_toplami += torch.log(p)
            adet += 1

    nll = -log_olasilik_toplami / adet
    return nll.item()


# --- tek bir isim uzerinde adim adim gorelim ---
print("--- 'emma' isminin olasiliklari ---")
P = (N + 1).float()
P = P / P.sum(1, keepdim=True)

toplam = 0.0
for h1, h2 in zip(['.'] + list('emma'), list('emma') + ['.']):
    p = P[stoi[h1], stoi[h2]].item()
    lp = torch.log(torch.tensor(p)).item()
    toplam += lp
    print(f"  {h1} -> {h2}:  olasilik = {p:.4f}   log = {lp:.4f}")
print(f"  toplam log olasilik = {toplam:.4f}")
print(f"  negatif ortalama (bu ismin loss'u) = {-toplam/5:.4f}")
print()


# --- tum veri seti uzerinde loss ---
print("--- tum veri setinde loss (smoothing degerine gore) ---")
for s in [0, 1, 10, 100]:
    L = loss_hesapla(s, kelimeler)
    print(f"  smoothing = {s:<5} -> loss = {L:.4f}")
print()

print("not: smoothing=0 iken bazi bigram'lar hic gorulmemis olabilir,")
print("     olasilik 0 cikar, log(0) = -sonsuz olur ve loss patlar.")
print()

# --- karsilastirma: tamamen rastgele model ---
import math
print(f"tamamen rastgele model (her harf esit): loss = {math.log(27):.4f}")
print(f"bizim model:                            loss = {loss_hesapla(1, kelimeler):.4f}")
print()
print("yani model rastgeleden belirgin sekilde iyi.")
