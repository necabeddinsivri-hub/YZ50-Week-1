# Gorev 5 - Ayni iki modeli Turkce isim listesiyle calistirma
# YZ50 Hafta 3
# Veri: github.com/Stealeristaken/Turkce-Isimler (temizlenmis hali: isimler.txt)

import torch
import torch.nn.functional as F
import time

kelimeler = open('isimler.txt', 'r', encoding='utf-8').read().splitlines()
print("toplam Turkce isim:", len(kelimeler))
print("ornekler:", kelimeler[100:106])
print()

# --- Turkce alfabe: 29 harf + nokta = 30 ---
harfler = sorted(list(set(''.join(kelimeler))))
print("alfabedeki harf sayisi:", len(harfler))
print("harfler:", ''.join(harfler))
print()

stoi = {h: i + 1 for i, h in enumerate(harfler)}
stoi['.'] = 0
itos = {i: h for h, i in stoi.items()}
K = len(stoi)      # 30


# ============ MODEL 1: SAYIM ============
print("=== MODEL 1: SAYIM ===")
N = torch.zeros((K, K), dtype=torch.int32)
for k in kelimeler:
    hs = ['.'] + list(k) + ['.']
    for h1, h2 in zip(hs, hs[1:]):
        N[stoi[h1], stoi[h2]] += 1

P = (N + 1).float()
P = P / P.sum(1, keepdim=True)

# loss
toplam, adet = 0.0, 0
for k in kelimeler:
    hs = ['.'] + list(k) + ['.']
    for h1, h2 in zip(hs, hs[1:]):
        toplam += torch.log(P[stoi[h1], stoi[h2]])
        adet += 1
sayim_loss = (-toplam / adet).item()
print("sayim modeli loss:", round(sayim_loss, 4))

g = torch.Generator().manual_seed(42)
print("urettigi 10 isim:")
for _ in range(10):
    isim, ix = [], 0
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, generator=g).item()
        if ix == 0:
            break
        isim.append(itos[ix])
    print("  ", ''.join(isim))
print()


# ============ MODEL 2: SINIR AGI ============
print("=== MODEL 2: SINIR AGI ===")
# sinir agi kismi yavas oldugu icin 50.000 isimle egitiyorum
kelimeler_ag = kelimeler[::5][:50000]
print("sinir agi icin kullanilan isim sayisi:", len(kelimeler_ag))

xs, ys = [], []
for k in kelimeler_ag:
    hs = ['.'] + list(k) + ['.']
    for h1, h2 in zip(hs, hs[1:]):
        xs.append(stoi[h1])
        ys.append(stoi[h2])
xs, ys = torch.tensor(xs), torch.tensor(ys)
n = xs.nelement()
print("ornek sayisi:", n)

g2 = torch.Generator().manual_seed(42)
W = torch.randn((K, K), generator=g2, requires_grad=True)

xenc = F.one_hot(xs, num_classes=K).float()

t0 = time.time()
for adim in range(100):
    logits = xenc @ W
    sayimlar = logits.exp()
    olasilik = sayimlar / sayimlar.sum(1, keepdim=True)
    loss = -olasilik[torch.arange(n), ys].log().mean() + 0.01 * (W ** 2).mean()

    W.grad = None
    loss.backward()
    W.data += -50 * W.grad

    if adim % 20 == 0:
        print(f"  adim {adim:>3}  loss = {loss.item():.4f}")
print(f"  adim  99  loss = {loss.item():.4f}   ({time.time()-t0:.1f} saniye)")
print()

print("sayim modeli loss :", round(sayim_loss, 4))
print("sinir agi loss    :", round(loss.item(), 4))
print()

g3 = torch.Generator().manual_seed(42)
print("sinir aginin urettigi 10 isim:")
for _ in range(10):
    isim, ix = [], 0
    while True:
        e = F.one_hot(torch.tensor([ix]), num_classes=K).float()
        lg = e @ W
        p = lg.exp() / lg.exp().sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, generator=g3).item()
        if ix == 0:
            break
        isim.append(itos[ix])
    print("  ", ''.join(isim))
