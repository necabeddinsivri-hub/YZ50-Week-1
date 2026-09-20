# Gorev 3 (ek) - Cross entropy ve BatchNorm'un geriye yayilimini
# TEK IFADEYE indirmek, sonra modeli loss.backward() OLMADAN egitmek
# YZ50 Hafta 5 (Part 4, Egzersiz 2, 3, 4)

import torch
import torch.nn.functional as F
import random

kelimeler = open('names.txt', 'r').read().splitlines()
harfler = sorted(list(set(''.join(kelimeler))))
stoi = {h: i + 1 for i, h in enumerate(harfler)}
stoi['.'] = 0
itos = {i: h for h, i in stoi.items()}
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
n1, n2 = int(0.8*len(kelimeler)), int(0.9*len(kelimeler))
Xtr, Ytr = veri_kur(kelimeler[:n1])
Xdev, Ydev = veri_kur(kelimeler[n1:n2])

EMB, GIZLI, n = 10, 64, 32


def model_kur(tohum=2147483647):
    g = torch.Generator().manual_seed(tohum)
    C  = torch.randn((V, EMB), generator=g)
    W1 = torch.randn((BAGLAM*EMB, GIZLI), generator=g) * (5/3)/((BAGLAM*EMB)**0.5)
    b1 = torch.randn(GIZLI, generator=g) * 0.1
    W2 = torch.randn((GIZLI, V), generator=g) * 0.1
    b2 = torch.randn(V, generator=g) * 0.1
    bngain = torch.randn((1, GIZLI)) * 0.1 + 1.0
    bnbias = torch.randn((1, GIZLI)) * 0.1
    ps = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in ps: p.requires_grad = True
    return ps


# ==========================================================
# EGZERSIZ 2: cross entropy geriye yayilimi tek satirda
# ==========================================================
print("=" * 58)
print("EGZERSIZ 2: cross entropy -> tek satir")
print("=" * 58)
print()
print("Uzun yol 8 adimdi: logprobs, probs, counts_sum_inv, counts_sum,")
print("counts, norm_logits, logit_maxes, logits.")
print()
print("Kagitta sadelestirince hepsi suna iniyor:")
print("    dlogits = (softmax(logits) - one_hot(Y)) / n")
print()

ps = model_kur()
C, W1, b1, W2, b2, bngain, bnbias = ps
g = torch.Generator().manual_seed(2147483647)
ix = torch.randint(0, Xtr.shape[0], (n,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

emb = C[Xb]; embcat = emb.view(emb.shape[0], -1)
hprebn = embcat @ W1 + b1
bnmean = hprebn.mean(0, keepdim=True)
bnvar = hprebn.var(0, keepdim=True, unbiased=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = (hprebn - bnmean) * bnvar_inv
hpreact = bngain * bnraw + bnbias
h = torch.tanh(hpreact)
logits = h @ W2 + b2
for t in [logits, hprebn, hpreact]: t.retain_grad()
loss = F.cross_entropy(logits, Yb)
for p in ps: p.grad = None
loss.backward()

# TEK SATIR
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n

fark = (dlogits - logits.grad).abs().max().item()
print(f"maksimum fark: {fark:.2e}   {'YAKLASIK AYNI' if fark < 1e-8 else 'HATA'}")
print()
print("Neden bu kadar sade? Cunku softmax ve log birbirini kismen goturuyor.")
print("Anlami da guzel: gradient = (modelin tahmini) - (gercek cevap).")
print("Model dogru harfe 1 dedi ve dogruysa gradient sifir, hic duzeltme yok.")
print()


# ==========================================================
# EGZERSIZ 3: BatchNorm geriye yayilimi tek satirda
# ==========================================================
print("=" * 58)
print("EGZERSIZ 3: BatchNorm -> tek satir")
print("=" * 58)
print()
print("Uzun yol 7 adimdi: bnraw, bnvar_inv, bnvar, bndiff2, bndiff,")
print("bnmeani, hprebn (iki katkiyla).")
print()

dhpreact = hpreact.grad
dbnraw = bngain * dhpreact

# TEK SATIR
dhprebn = bnvar_inv/n * (n*dbnraw - dbnraw.sum(0) - n/(n-1)*bnraw*(dbnraw*bnraw).sum(0))

fark2 = (dhprebn - hprebn.grad).abs().max().item()
print("    dhprebn = bnvar_inv/n * (n*dbnraw - dbnraw.sum(0)")
print("              - n/(n-1) * bnraw * (dbnraw*bnraw).sum(0))")
print()
print(f"maksimum fark: {fark2:.2e}   {'YAKLASIK AYNI' if fark2 < 1e-8 else 'HATA'}")
print()
print("Uc terimin anlami:")
print("  n*dbnraw            : dogrudan etki")
print("  -dbnraw.sum(0)      : ortalamayi degistirdigin icin geri odeme")
print("  -...*bnraw*...      : varyansi degistirdigin icin geri odeme")
print()
print("Yani BatchNorm'da bir ornegi oynatmak, ayni batch'teki DIGER")
print("orneklerin ciktisini da degistiriyor. Bu iki duzeltme terimi ondan.")
print()


# ==========================================================
# EGZERSIZ 4: loss.backward() OLMADAN egitim
# ==========================================================
print("=" * 58)
print("EGZERSIZ 4: loss.backward() OLMADAN egitim")
print("=" * 58)
print()

ps = model_kur()
C, W1, b1, W2, b2, bngain, bnbias = ps
gr = torch.Generator().manual_seed(1)
ADIM = 20000
bnmean_calisan = torch.zeros((1, GIZLI))
bnstd_calisan = torch.ones((1, GIZLI))

for adim in range(ADIM):
    ix = torch.randint(0, Xtr.shape[0], (n,), generator=gr)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # --- ILERI ---
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1
    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=True)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    with torch.no_grad():
        bnmean_calisan = 0.999 * bnmean_calisan + 0.001 * bnmean
        bnstd_calisan = 0.999 * bnstd_calisan + 0.001 * bnvar.sqrt()

    # --- GERI: loss.backward() YOK, hepsi elle ---
    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n

    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)

    dhpreact = (1.0 - h**2) * dh
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    dbnraw = bngain * dhpreact

    dhprebn = bnvar_inv/n * (n*dbnraw - dbnraw.sum(0)
                             - n/(n-1)*bnraw*(dbnraw*bnraw).sum(0))

    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)

    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    for i in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            dC[Xb[i, j]] += demb[i, j]

    gradlar = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

    # --- GUNCELLE ---
    lr = 0.1 if adim < 0.7*ADIM else 0.01
    for p, gr_ in zip(ps, gradlar):
        p.data += -lr * gr_

    if adim % 4000 == 0:
        print(f"  adim {adim:>6}  loss = {loss.item():.4f}")

print(f"  adim {ADIM-1:>6}  loss = {loss.item():.4f}")
print()


@torch.no_grad()
def dev_loss():
    emb = C[Xdev]; ec = emb.view(emb.shape[0], -1)
    hp = ec @ W1 + b1
    hp = bngain * (hp - bnmean_calisan) / bnstd_calisan + bnbias
    return F.cross_entropy(torch.tanh(hp) @ W2 + b2, Ydev).item()


print(f"dev loss = {dev_loss():.4f}")
print()
print("Karsilastirma:")
print("  Hafta 3 bigram             : 2.4544")
print("  Hafta 4 MLP (autograd ile) : 2.1486")
print(f"  Hafta 5 MLP (elle gradient): {dev_loss():.4f}")
print()
print("Ayni model, ayni sonuc. Tek fark: bu sefer gradient'leri")
print("PyTorch degil ben hesapladim.")
print()

print("--- elle egitilen modelin urettigi 10 isim ---")
gs = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(10):
    cikti, pencere = [], [0]*BAGLAM
    while True:
        with torch.no_grad():
            e = C[torch.tensor([pencere])]
            hp = e.view(1, -1) @ W1 + b1
            hp = bngain * (hp - bnmean_calisan) / bnstd_calisan + bnbias
            lg = torch.tanh(hp) @ W2 + b2
        p = F.softmax(lg, 1)
        ixx = torch.multinomial(p, 1, generator=gs).item()
        if ixx == 0: break
        cikti.append(itos[ixx])
        pencere = pencere[1:] + [ixx]
    print("  ", ''.join(cikti))
