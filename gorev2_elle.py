# Gorev 2 - Butun gradient'leri ELLE yaz, cmp ile dogrula
# YZ50 Hafta 5 (Part 4, Egzersiz 1)

import torch
import torch.nn.functional as F
import random

kelimeler = open('names.txt', 'r').read().splitlines()
harfler = sorted(list(set(''.join(kelimeler))))
stoi = {h: i + 1 for i, h in enumerate(harfler)}
stoi['.'] = 0
V, BAGLAM = 27, 3


def veri_kur(liste):
    X, Y = [], []
    for k in liste:
        p = [0] * BAGLAM
        for h in k + '.':
            ix = stoi[h]
            X.append(p); Y.append(ix)
            p = p[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


random.seed(42); random.shuffle(kelimeler)
Xtr, Ytr = veri_kur(kelimeler[:int(0.8 * len(kelimeler))])

EMB, GIZLI, n = 10, 64, 32
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((V, EMB), generator=g)
W1 = torch.randn((BAGLAM*EMB, GIZLI), generator=g) * (5/3)/((BAGLAM*EMB)**0.5)
b1 = torch.randn(GIZLI, generator=g) * 0.1
W2 = torch.randn((GIZLI, V), generator=g) * 0.1
b2 = torch.randn(V, generator=g) * 0.1
bngain = torch.randn((1, GIZLI)) * 0.1 + 1.0
bnbias = torch.randn((1, GIZLI)) * 0.1
parametreler = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parametreler: p.requires_grad = True

ix = torch.randint(0, Xtr.shape[0], (n,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]


# ---------- KARSILASTIRMA FONKSIYONU ----------
def cmp(isim, dt, t):
    """dt: benim elle hesabim, t: PyTorch'un tensor'u"""
    tam = torch.all(dt == t.grad).item()
    yaklasik = torch.allclose(dt, t.grad)
    maks_fark = (dt - t.grad).abs().max().item()
    durum = "EXACT" if tam else ("approx" if yaklasik else "HATA ")
    print(f"{isim:<16} {durum:<8} maks fark: {maks_fark:.2e}")
    return tam or yaklasik


# ---------- ILERI GECIS ----------
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
hprebn = embcat @ W1 + b1
bnmeani = 1/n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
h = torch.tanh(hpreact)
logits = h @ W2 + b2
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

araclar = [logprobs, probs, counts, counts_sum, counts_sum_inv, norm_logits,
           logit_maxes, logits, h, hpreact, bnraw, bnvar_inv, bnvar, bndiff2,
           bndiff, hprebn, bnmeani, embcat, emb]
for t in araclar: t.retain_grad()
for p in parametreler: p.grad = None
loss.backward()

print("=" * 52)
print("GERIYE DOGRU, ADIM ADIM, ELLE")
print("=" * 52)
print()

# ---------- 1) loss -> logprobs ----------
# loss = -ortalama(secilen logprob'lar).  n tane secili eleman var.
# Her seciliye -1/n dusuyor, digerlerine 0.
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
cmp('logprobs', dlogprobs, logprobs)

# ---------- 2) logprobs = probs.log() ----------
# d(log x)/dx = 1/x
dprobs = (1.0 / probs) * dlogprobs
cmp('probs', dprobs, probs)

# ---------- 3) probs = counts * counts_sum_inv ----------
# counts_sum_inv sekli (32,1), probs sekli (32,27) -> BROADCASTING var.
# Ileri geciste yayilan boyut, geri gecisde TOPLANIR.
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

# counts iki yerde kullanildi: burada ve counts_sum'da. Once birinci katki:
dcounts = counts_sum_inv * dprobs

# ---------- 4) counts_sum_inv = counts_sum ** -1 ----------
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)

# ---------- 5) counts_sum = counts.sum(1) ----------
# Toplamin turevi 1 -> gradient butun sutunlara kopyalanir.
dcounts += torch.ones_like(counts) * dcounts_sum     # ikinci katki, TOPLANIYOR
cmp('counts', dcounts, counts)

# ---------- 6) counts = norm_logits.exp() ----------
dnorm_logits = counts * dcounts        # d(e^x)/dx = e^x = counts
cmp('norm_logits', dnorm_logits, norm_logits)

# ---------- 7) norm_logits = logits - logit_maxes ----------
dlogits = dnorm_logits.clone()                        # birinci katki
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
cmp('logit_maxes', dlogit_maxes, logit_maxes)

# ---------- 8) logit_maxes = logits.max(1) ----------
# Sadece maksimum olan elemana gradient gider.
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
cmp('logits', dlogits, logits)

# ---------- 9) logits = h @ W2 + b2 ----------
dh  = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)
cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)

# ---------- 10) h = tanh(hpreact) ----------
dhpreact = (1.0 - h**2) * dh           # Hafta 2'deki satirin ta kendisi
cmp('hpreact', dhpreact, hpreact)

# ---------- 11) hpreact = bngain * bnraw + bnbias ----------
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnraw  = bngain * dhpreact
dbnbias = dhpreact.sum(0, keepdim=True)
cmp('bngain', dbngain, bngain)
cmp('bnraw', dbnraw, bnraw)
cmp('bnbias', dbnbias, bnbias)

# ---------- 12) bnraw = bndiff * bnvar_inv ----------
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
dbndiff = bnvar_inv * dbnraw            # birinci katki
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

# ---------- 13) bnvar_inv = (bnvar + 1e-5) ** -0.5 ----------
dbnvar = (-0.5 * (bnvar + 1e-5)**-1.5) * dbnvar_inv
cmp('bnvar', dbnvar, bnvar)

# ---------- 14) bnvar = 1/(n-1) * bndiff2.sum(0) ----------
dbndiff2 = (1.0/(n-1)) * torch.ones_like(bndiff2) * dbnvar
cmp('bndiff2', dbndiff2, bndiff2)

# ---------- 15) bndiff2 = bndiff ** 2 ----------
dbndiff += (2 * bndiff) * dbndiff2      # ikinci katki
cmp('bndiff', dbndiff, bndiff)

# ---------- 16) bndiff = hprebn - bnmeani ----------
dhprebn = dbndiff.clone()               # birinci katki
dbnmeani = (-dbndiff).sum(0, keepdim=True)
cmp('bnmeani', dbnmeani, bnmeani)

# ---------- 17) bnmeani = 1/n * hprebn.sum(0) ----------
dhprebn += (1.0/n) * torch.ones_like(hprebn) * dbnmeani   # ikinci katki
cmp('hprebn', dhprebn, hprebn)

# ---------- 18) hprebn = embcat @ W1 + b1 ----------
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
cmp('embcat', dembcat, embcat)
cmp('W1', dW1, W1)
cmp('b1', db1, b1)

# ---------- 19) embcat = emb.view(...) ----------
demb = dembcat.view(emb.shape)
cmp('emb', demb, emb)

# ---------- 20) emb = C[Xb] ----------
# Ayni harf birden fazla yerde gectiyse gradient'ler TOPLANIR.
dC = torch.zeros_like(C)
for i in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        dC[Xb[i, j]] += demb[i, j]
cmp('C', dC, C)

print()
print("Hepsi EXACT veya approx ise gecen hafta loss.backward()'in")
print("yaptigi isi bastan sona kendim yazmisim demektir.")
