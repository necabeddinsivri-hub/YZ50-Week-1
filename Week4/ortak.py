# Ortak yardimcilar - diger scriptler bunu import ediyor
import torch
import random


def veri_yukle(dosya='names.txt'):
    kelimeler = open(dosya, 'r', encoding='utf-8').read().splitlines()
    harfler = sorted(list(set(''.join(kelimeler))))
    stoi = {h: i + 1 for i, h in enumerate(harfler)}
    stoi['.'] = 0
    itos = {i: h for h, i in stoi.items()}
    return kelimeler, stoi, itos


def veri_kur(kelime_listesi, stoi, baglam=3):
    X, Y = [], []
    for k in kelime_listesi:
        pencere = [0] * baglam
        for harf in k + '.':
            ix = stoi[harf]
            X.append(pencere)
            Y.append(ix)
            pencere = pencere[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def bol(kelimeler, tohum=42):
    """train %80 / dev %10 / test %10"""
    kelimeler = list(kelimeler)
    random.seed(tohum)
    random.shuffle(kelimeler)
    n1 = int(0.8 * len(kelimeler))
    n2 = int(0.9 * len(kelimeler))
    return kelimeler[:n1], kelimeler[n1:n2], kelimeler[n2:]
