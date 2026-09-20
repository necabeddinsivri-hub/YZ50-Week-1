# Gorev 1 - Modeli kucuk adimlara bolup her ara degiskenin gradient'ini almak
# YZ50 Hafta 5 (Part 4, ilk yari)
#
# Gecen hafta loss.backward() diyip geciyorduk. Bu hafta o satirin
# arkasinda ne oldugunu gorecegiz: once modeli 20 kucuk adima boluyoruz,
# her adimin ciktisini ayri bir degiskende tutuyoruz.

import torch
import torch.nn.functional as F

kelimeler = open('names.txt', 'r').read().splitlines()
harfler = sorted(list(set(''.join(kelimeler))))
stoi = {h: i + 1 for i, h in enumerate(harfler)}
stoi['.'] = 0
itos = {i: h for h, i in stoi.items()}
V = 27              # alfabe boyutu
BAGLAM = 3


def veri_kur(liste):
    X, Y = [], []
    for k in liste:
        p = [0] * BAGLAM
        for h in k + '.':
            ix = stoi[h]
            X.append(p); Y.append(ix)
            p = p[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


import random
random.seed(42)
random.shuffle(kelimeler)
n1 = int(0.8 * len(kelimeler))
Xtr, Ytr = veri_kur(kelimeler[:n1])

EMB, GIZLI = 10, 64
g = torch.Generator().manual_seed(2147483647)

C  = torch.randn((V, EMB), generator=g)
W1 = torch.randn((BAGLAM * EMB, GIZLI), generator=g) * (5/3) / ((BAGLAM*EMB)**0.5)
b1 = torch.randn(GIZLI, generator=g) * 0.1      # BatchNorm'la gereksiz ama
W2 = torch.randn((GIZLI, V), generator=g) * 0.1  # egzersiz icin duruyor
b2 = torch.randn(V, generator=g) * 0.1
bngain = torch.randn((1, GIZLI)) * 0.1 + 1.0
bnbias = torch.randn((1, GIZLI)) * 0.1

parametreler = [C, W1, b1, W2, b2, bngain, bnbias]
print("parametre sayisi:", sum(p.nelement() for p in parametreler))
for p in parametreler:
    p.requires_grad = True

n = 32                      # kucuk batch: elle hesap kolay olsun
ix = torch.randint(0, Xtr.shape[0], (n,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]


# ============ ILERI GECIS, 20 KUCUK ADIM ============
emb = C[Xb]                                        # embedding'leri cek
embcat = emb.view(emb.shape[0], -1)                # duzlestir

# Linear katman 1
hprebn = embcat @ W1 + b1

# BatchNorm
bnmeani = 1/n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)     # Bessel duzeltmesi
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

# Aktivasyon
h = torch.tanh(hpreact)

# Linear katman 2
logits = h @ W2 + b2

# Cross entropy, elle
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes                 # sayisal guvenlik
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

print("loss =", loss.item())
print()


# ============ PyTorch'un gradient'lerini al ============
ara_degiskenler = [logprobs, probs, counts, counts_sum, counts_sum_inv,
                   norm_logits, logit_maxes, logits, h, hpreact, bnraw,
                   bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
                   embcat, emb]
for t in ara_degiskenler:
    t.retain_grad()          # ara degiskenlerin grad'ini sakla

for p in parametreler:
    p.grad = None
loss.backward()

isimler = ['logprobs', 'probs', 'counts', 'counts_sum', 'counts_sum_inv',
           'norm_logits', 'logit_maxes', 'logits', 'h', 'hpreact', 'bnraw',
           'bnvar_inv', 'bnvar', 'bndiff2', 'bndiff', 'hprebn', 'bnmeani',
           'embcat', 'emb']

print("--- ara degiskenler ve PyTorch'un hesapladigi gradient'ler ---")
print(f"{'degisken':<16}{'sekil':<16}{'grad var mi':<12}{'grad normu':>12}")
for isim, t in zip(isimler, ara_degiskenler):
    var = "evet" if t.grad is not None else "HAYIR"
    norm = t.grad.abs().sum().item() if t.grad is not None else 0.0
    print(f"{isim:<16}{str(tuple(t.shape)):<16}{var:<12}{norm:>12.4f}")
print()
print("Simdi bu 19 gradient'i tek tek ELLE hesaplayacagiz (gorev 2).")
