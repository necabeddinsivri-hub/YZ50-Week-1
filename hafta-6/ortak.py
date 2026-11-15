# ortak.py - veri hazirlama, egitim ve degerlendirme
import random
import torch
import torch.nn.functional as F


def veri_yukle(dosya):
    kelimeler = open(dosya, 'r', encoding='utf-8').read().splitlines()
    harfler = sorted(list(set(''.join(kelimeler))))
    stoi = {h: i + 1 for i, h in enumerate(harfler)}
    stoi['.'] = 0
    itos = {i: h for h, i in stoi.items()}
    return kelimeler, stoi, itos


def veri_kur(liste, stoi, baglam):
    X, Y = [], []
    for k in liste:
        p = [0] * baglam
        for h in k + '.':
            ix = stoi[h]
            X.append(p); Y.append(ix)
            p = p[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def bolumle(dosya, baglam, alt_kume=None):
    kelimeler, stoi, itos = veri_yukle(dosya)
    if alt_kume:
        kelimeler = kelimeler[::alt_kume]
    random.seed(42)
    random.shuffle(kelimeler)
    n1, n2 = int(0.8 * len(kelimeler)), int(0.9 * len(kelimeler))
    tr = veri_kur(kelimeler[:n1], stoi, baglam)
    dv = veri_kur(kelimeler[n1:n2], stoi, baglam)
    te = veri_kur(kelimeler[n2:], stoi, baglam)
    return tr, dv, te, stoi, itos


def egit(model, Xtr, Ytr, adim=30000, batch=32, sessiz=False):
    for p in model.parameters():
        p.requires_grad = True
    g = torch.Generator().manual_seed(1)
    kayit = []
    model.egitim_kipi(True)
    for i in range(adim):
        ix = torch.randint(0, Xtr.shape[0], (batch,), generator=g)
        logits = model(Xtr[ix])
        loss = F.cross_entropy(logits, Ytr[ix])
        for p in model.parameters():
            p.grad = None
        loss.backward()
        lr = 0.1 if i < int(0.75 * adim) else 0.01
        for p in model.parameters():
            p.data += -lr * p.grad
        kayit.append(loss.log10().item())
        if not sessiz and i % 5000 == 0:
            print(f"    adim {i:>6}/{adim}  loss = {loss.item():.4f}")
    return kayit


@torch.no_grad()
def degerlendir(model, X, Y, parca=20000):
    """Bellegi tasirmamak icin veriyi parcalar halinde degerlendirir."""
    model.egitim_kipi(False)
    toplam = 0.0
    for i in range(0, X.shape[0], parca):
        xb, yb = X[i:i+parca], Y[i:i+parca]
        toplam += F.cross_entropy(model(xb), yb, reduction='sum').item()
    return toplam / X.shape[0]


@torch.no_grad()
def isim_uret(model, itos, baglam, adet=10, tohum=2147483647):
    model.egitim_kipi(False)
    g = torch.Generator().manual_seed(tohum)
    sonuc = []
    for _ in range(adet):
        cikti, p = [], [0] * baglam
        while True:
            logits = model(torch.tensor([p]))
            ix = torch.multinomial(F.softmax(logits, 1), 1, generator=g).item()
            if ix == 0:
                break
            cikti.append(itos[ix])
            p = p[1:] + [ix]
        sonuc.append(''.join(cikti))
    return sonuc


def son_katmani_kucult(model):
    """Hafta 4 dersi: son katmanin ciktisi ~0 baslasin, ilk loss ~3.3 olsun."""
    with torch.no_grad():
        model.layers[-1].weight *= 0.1


def parametre_sayisi(model):
    return sum(p.nelement() for p in model.parameters())
