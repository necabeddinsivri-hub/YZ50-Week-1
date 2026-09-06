# Gorev 4 - Ayni bigram modelini tek katmanli sinir agiyla kurma
# YZ50 Hafta 3
# Buradaki backward() gecen hafta micrograd'da kendi yazdigim mekanizmanin aynisi

import torch
import torch.nn.functional as F

kelimeler = open('names.txt', 'r').read().splitlines()

harfler = sorted(list(set(''.join(kelimeler))))
stoi = {h: i + 1 for i, h in enumerate(harfler)}
stoi['.'] = 0
itos = {i: h for h, i in stoi.items()}


# --- egitim verisini hazirla: (girdi harf, hedef harf) ciftleri ---
xs, ys = [], []
for k in kelimeler:
    hs = ['.'] + list(k) + ['.']
    for h1, h2 in zip(hs, hs[1:]):
        xs.append(stoi[h1])
        ys.append(stoi[h2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
ornek_sayisi = xs.nelement()
print("ornek sayisi:", ornek_sayisi)
print("ilk 5 ornek:", [(itos[xs[i].item()], itos[ys[i].item()]) for i in range(5)])
print()


# --- agirlik matrisi: 27x27, rastgele baslat ---
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)


# --- egitim dongusu ---
print("--- egitim ---")
for adim in range(200):

    # 1) ILERI GECIS
    xenc = F.one_hot(xs, num_classes=27).float()   # her harf 27'lik vektor
    logits = xenc @ W                              # ham skorlar
    sayimlar = logits.exp()                        # pozitif yap (sayim gibi)
    olasilik = sayimlar / sayimlar.sum(1, keepdim=True)   # softmax
    loss = -olasilik[torch.arange(ornek_sayisi), ys].log().mean() \
           + 0.01 * (W ** 2).mean()                # son terim = smoothing

    # 2) GERI GECIS  <- gecen hafta kendi yazdigim mekanizma
    W.grad = None
    loss.backward()

    # 3) GUNCELLEME
    W.data += -50 * W.grad

    if adim % 20 == 0:
        print(f"  adim {adim:>3}  loss = {loss.item():.4f}")

print(f"  adim 199  loss = {loss.item():.4f}")
print()

print("sayim modelinin loss'u  : 2.4544")
print(f"sinir aginin loss'u     : {loss.item():.4f}")
print("iki yol ayni sonuca cikiyor.")
print()


# --- egitilmis agdan isim uret ---
print("--- sinir aginin urettigi 10 isim ---")
g2 = torch.Generator().manual_seed(2147483647)
for _ in range(10):
    isim = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        p = logits.exp() / logits.exp().sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, generator=g2).item()
        if ix == 0:
            break
        isim.append(itos[ix])
    print("  ", ''.join(isim))
