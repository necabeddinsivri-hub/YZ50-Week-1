# Gorev 6 (ek) - Trigram modeli, train/dev/test bolmesi, smoothing ayari
# YZ50 Hafta 3

import torch
import random

kelimeler = open('names.txt', 'r').read().splitlines()

harfler = sorted(list(set(''.join(kelimeler))))
stoi = {h: i + 1 for i, h in enumerate(harfler)}
stoi['.'] = 0
itos = {i: h for h, i in stoi.items()}
K = 27


# --- veriyi %80 / %10 / %10 bol ---
random.seed(42)
random.shuffle(kelimeler)
n1 = int(0.8 * len(kelimeler))
n2 = int(0.9 * len(kelimeler))
train, dev, test = kelimeler[:n1], kelimeler[n1:n2], kelimeler[n2:]
print(f"train: {len(train)}   dev: {len(dev)}   test: {len(test)}")
print()


# --- BIGRAM (karsilastirma icin) ---
N2 = torch.zeros((K, K), dtype=torch.int32)
for k in train:
    hs = ['.'] + list(k) + ['.']
    for a, b in zip(hs, hs[1:]):
        N2[stoi[a], stoi[b]] += 1


def bigram_loss(veri, s):
    P = (N2 + s).float()
    P = P / P.sum(1, keepdim=True)
    t, c = 0.0, 0
    for k in veri:
        hs = ['.'] + list(k) + ['.']
        for a, b in zip(hs, hs[1:]):
            t += torch.log(P[stoi[a], stoi[b]])
            c += 1
    return (-t / c).item()


# --- TRIGRAM: iki onceki harfe bak ---
N3 = torch.zeros((K, K, K), dtype=torch.int32)
for k in train:
    hs = ['.', '.'] + list(k) + ['.']
    for a, b, c in zip(hs, hs[1:], hs[2:]):
        N3[stoi[a], stoi[b], stoi[c]] += 1


def trigram_loss(veri, s):
    P = (N3 + s).float()
    P = P / P.sum(2, keepdim=True)
    t, n = 0.0, 0
    for k in veri:
        hs = ['.', '.'] + list(k) + ['.']
        for a, b, c in zip(hs, hs[1:], hs[2:]):
            t += torch.log(P[stoi[a], stoi[b], stoi[c]])
            n += 1
    return (-t / n).item()


# --- smoothing gucunu DEV setine gore ayarla ---
print("--- trigram: smoothing taramasi (dev loss'una gore) ---")
en_iyi_s, en_iyi_loss = None, 999
for s in [0.01, 0.1, 0.5, 1, 3, 10]:
    L = trigram_loss(dev, s)
    print(f"  smoothing = {s:<6} dev loss = {L:.4f}")
    if L < en_iyi_loss:
        en_iyi_loss, en_iyi_s = L, s
print(f"  en iyi smoothing: {en_iyi_s}")
print()


# --- son karsilastirma ---
print("--- BIGRAM vs TRIGRAM (test seti) ---")
bl = bigram_loss(test, 1)
tl = trigram_loss(test, en_iyi_s)
print(f"  bigram  test loss = {bl:.4f}")
print(f"  trigram test loss = {tl:.4f}")
print(f"  iyilesme = {bl - tl:.4f}")
print()


# --- trigram'dan isim uret ---
P3 = (N3 + en_iyi_s).float()
P3 = P3 / P3.sum(2, keepdim=True)

g = torch.Generator().manual_seed(2147483647)
print("--- trigram'in urettigi 10 isim ---")
for _ in range(10):
    isim = []
    i1, i2 = 0, 0
    while True:
        p = P3[i1, i2]
        ix = torch.multinomial(p, num_samples=1, generator=g).item()
        if ix == 0:
            break
        isim.append(itos[ix])
        i1, i2 = i2, ix
    print("  ", ''.join(isim))
