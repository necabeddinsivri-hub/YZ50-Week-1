# Gorev 1 - Baglam penceresi ve embedding tablosu
# YZ50 Hafta 4 (Part 2, 9:03 - 18:35)
#
# Gecen hafta model TEK harfe bakiyordu. Bu hafta UC harfe bakacak.
# Ve harfleri artik 27'lik one-hot vektorle degil, ogrenilen kucuk
# vektorlerle (embedding) temsil edecegiz.

import torch

kelimeler = open('names.txt', 'r').read().splitlines()

harfler = sorted(list(set(''.join(kelimeler))))
stoi = {h: i + 1 for i, h in enumerate(harfler)}
stoi['.'] = 0
itos = {i: h for h, i in stoi.items()}

BAGLAM = 3        # kac onceki harfe bakiyoruz


def veri_kur(kelime_listesi):
    X, Y = [], []
    for k in kelime_listesi:
        pencere = [0] * BAGLAM          # ... ile basla
        for harf in k + '.':
            ix = stoi[harf]
            X.append(pencere)
            Y.append(ix)
            pencere = pencere[1:] + [ix]   # pencereyi kaydir
    return torch.tensor(X), torch.tensor(Y)


X, Y = veri_kur(kelimeler[:5])

print("--- ilk 5 isim icin veri seti ---")
print("isimler:", kelimeler[:5])
print()
print("baglam        -> hedef")
for i in range(min(20, len(X))):
    baglam = ''.join(itos[j.item()] for j in X[i])
    print(f"  {baglam}       ->  {itos[Y[i].item()]}")
print()

X, Y = veri_kur(kelimeler)
print("tum veri: X sekli =", X.shape, " Y sekli =", Y.shape)
print()


# --- EMBEDDING TABLOSU ---
# 27 harf, her biri 2 boyutlu bir vektorle temsil edilecek
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)

print("--- embedding tablosu ---")
print("C sekli:", C.shape, " (27 harf, her biri 2 sayi)")
print("'a' harfinin vektoru:", C[stoi['a']].tolist())
print("'.' sembolunun vektoru:", C[0].tolist())
print()

# indeksleme ile embedding cekme - PyTorch bunu dogrudan destekliyor
ornek = C[X[:3]]
print("C[X[:3]] sekli:", ornek.shape, " (3 ornek, 3 harf, 2 sayi)")
print()
print("ilk ornegin 3 harfinin vektorleri:")
print(ornek[0])
print()

print("KARSILASTIRMA:")
print("  Hafta 3: her harf 27'lik one-hot vektor, 27x27 = 729 parametre")
print("  Hafta 4: her harf 2'lik ogrenilen vektor, 27x2 =  54 parametre")
print("  Ustelik embedding OGRENILIYOR: benzer harfler yakin dusuyor.")
