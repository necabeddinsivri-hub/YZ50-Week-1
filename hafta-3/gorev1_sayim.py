# Gorev 1 - Bigram'lari sayma
# YZ50 Hafta 3

import torch

kelimeler = open('names.txt', 'r').read().splitlines()
print("toplam isim:", len(kelimeler))
print("ilk 5:", kelimeler[:5])
print()

# --- 1a) once python dictionary ile sayalim ---
sayim = {}
for k in kelimeler:
    harfler = ['.'] + list(k) + ['.']
    for h1, h2 in zip(harfler, harfler[1:]):
        ikili = (h1, h2)
        sayim[ikili] = sayim.get(ikili, 0) + 1

en_cok = sorted(sayim.items(), key=lambda x: -x[1])[:10]
print("en cok gecen 10 bigram:")
for ikili, adet in en_cok:
    print("  ", ikili[0], "->", ikili[1], ":", adet)
print()

# --- 1b) simdi 27x27 tensor ile sayalim ---
harfler = sorted(list(set(''.join(kelimeler))))
stoi = {h: i + 1 for i, h in enumerate(harfler)}   # a=1, b=2, ...
stoi['.'] = 0                                      # nokta = 0
itos = {i: h for h, i in stoi.items()}

N = torch.zeros((27, 27), dtype=torch.int32)

for k in kelimeler:
    hs = ['.'] + list(k) + ['.']
    for h1, h2 in zip(hs, hs[1:]):
        N[stoi[h1], stoi[h2]] += 1

print("tablo boyutu:", N.shape)
print("toplam bigram:", N.sum().item())
print()

# --- 1c) tabloyu ekrana yazdir (gorsellestirme) ---
print("sayim tablosu (ilk 8x8 kose):")
print("     ", "".join(f"{itos[j]:>6}" for j in range(8)))
for i in range(8):
    satir = f"{itos[i]:>3} :"
    for j in range(8):
        satir += f"{N[i, j].item():>6}"
    print(satir)
print()

# istersen matplotlib ile de cizdirilebilir:
# import matplotlib.pyplot as plt
# plt.figure(figsize=(16,16)); plt.imshow(N, cmap='Blues')




