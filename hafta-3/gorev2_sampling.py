# Gorev 2 - Sayimlari olasiliga cevirip isim uretme
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


# --- sayimlari olasiliga cevir ---
# her SATIR kendi icinde toplami 1 olacak sekilde bolunur
P = N.float()
P = P / P.sum(1, keepdim=True)     # <-- keepdim=True COK onemli

print("kontrol: her satirin toplami 1 mi?")
print(P.sum(1)[:5])
print()


# --- keepdim tuzagi ---
print("--- keepdim tuzagi ---")
toplam_dogru = N.sum(1, keepdim=True)    # sekil (27, 1)
toplam_yanlis = N.sum(1)                 # sekil (27,)
print("keepdim=True  sekil:", toplam_dogru.shape, " -> satirlara boler (DOGRU)")
print("keepdim=False sekil:", toplam_yanlis.shape, " -> sutunlara boler (YANLIS)")

P_yanlis = N.float() / toplam_yanlis
print("yanlis versiyonda satir toplami:", P_yanlis.sum(1)[0].item(), "(1 olmasi lazimdi)")
print()


# --- modelden isim uret ---
g = torch.Generator().manual_seed(2147483647)

print("--- modelin urettigi 10 isim ---")
for _ in range(10):
    isim = []
    ix = 0                                   # nokta ile basla
    while True:
        p = P[ix]                            # bu harften sonra ne gelir?
        ix = torch.multinomial(p, num_samples=1, replacement=True,
                               generator=g).item()
        if ix == 0:                          # nokta geldi, isim bitti
            break
        isim.append(itos[ix])
    print("  ", ''.join(isim))
print()


# --- karsilastirma: tamamen rastgele model ---
print("--- egitilmemis (tamamen rastgele) model ne uretirdi ---")
P_rastgele = torch.ones((27, 27)) / 27.0
for _ in range(3):
    isim = []
    ix = 0
    while True:
        ix = torch.multinomial(P_rastgele[ix], num_samples=1,
                               generator=g).item()
        if ix == 0:
            break
        isim.append(itos[ix])
    print("  ", ''.join(isim))
