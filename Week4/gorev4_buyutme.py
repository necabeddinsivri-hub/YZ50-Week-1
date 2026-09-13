# Gorev 4 - Modeli buyutme, embedding gorsellestirme, isim uretme
# YZ50 Hafta 4 (Part 2, 1:00:49 - 1:13:24)

import torch
import torch.nn.functional as F
from ortak import veri_yukle, veri_kur, bol

kelimeler, stoi, itos = veri_yukle('names.txt')
BAGLAM = 3
train, dev, test = bol(kelimeler)
Xtr, Ytr = veri_kur(train, stoi, BAGLAM)
Xdev, Ydev = veri_kur(dev, stoi, BAGLAM)


def egit(boyut_emb, gizli, adim_sayisi=30000, sessiz=True):
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((27, boyut_emb), generator=g)
    W1 = torch.randn((BAGLAM * boyut_emb, gizli), generator=g) * 0.2
    b1 = torch.randn(gizli, generator=g) * 0.01
    W2 = torch.randn((gizli, 27), generator=g) * 0.05
    b2 = torch.randn(27, generator=g) * 0.0
    ps = [C, W1, b1, W2, b2]
    for p in ps: p.requires_grad = True

    def ileri(Xb):
        emb = C[Xb]
        h = torch.tanh(emb.view(-1, BAGLAM * boyut_emb) @ W1 + b1)
        return h @ W2 + b2

    gr = torch.Generator().manual_seed(1)
    for adim in range(adim_sayisi):
        ix = torch.randint(0, Xtr.shape[0], (64,), generator=gr)
        loss = F.cross_entropy(ileri(Xtr[ix]), Ytr[ix])
        for p in ps: p.grad = None
        loss.backward()
        lr = 0.1 if adim < adim_sayisi * 0.7 else 0.01
        for p in ps: p.data += -lr * p.grad

    with torch.no_grad():
        tr = F.cross_entropy(ileri(Xtr), Ytr).item()
        dv = F.cross_entropy(ileri(Xdev), Ydev).item()
    return ps, ileri, tr, dv, sum(p.nelement() for p in ps)


print("--- model buyuklugu taramasi ---")
print(f"{'emb':>4}{'gizli':>7}{'parametre':>11}{'train':>9}{'dev':>9}")
sonuclar = []
for emb, gizli in [(2, 100), (2, 300), (10, 200), (10, 300)]:
    ps, ileri, tr, dv, n = egit(emb, gizli)
    sonuclar.append((emb, gizli, n, tr, dv))
    print(f"{emb:>4}{gizli:>7}{n:>11}{tr:>9.4f}{dv:>9.4f}")
print()

en_iyi = min(sonuclar, key=lambda x: x[4])
print(f"en iyi: emb={en_iyi[0]}, gizli={en_iyi[1]}, dev loss={en_iyi[4]:.4f}")
print()


# --- 2 boyutlu embedding'i cizdir ---
print("--- 2 boyutlu embedding: hangi harfler yakin dusuyor ---")
ps, ileri, tr, dv, n = egit(2, 100)
C = ps[0].detach()

# metin tabanli 'grafik'
xmin, xmax = C[:, 0].min().item(), C[:, 0].max().item()
ymin, ymax = C[:, 1].min().item(), C[:, 1].max().item()
GEN, YUK = 62, 22
izgara = [[' '] * GEN for _ in range(YUK)]
for i in range(27):
    gx = int((C[i, 0].item() - xmin) / (xmax - xmin + 1e-9) * (GEN - 2))
    gy = int((1 - (C[i, 1].item() - ymin) / (ymax - ymin + 1e-9)) * (YUK - 1))
    izgara[gy][gx] = itos[i]
print("+" + "-" * GEN + "+")
for satir in izgara:
    print("|" + ''.join(satir) + "|")
print("+" + "-" * GEN + "+")
print()

# sesli harflerin nerede oldugunu sayisal olarak da gorelim
sesliler = ['a', 'e', 'i', 'o', 'u']
print("sesli harflerin koordinatlari:")
for h in sesliler:
    v = C[stoi[h]]
    print(f"  {h}: ({v[0]:>6.2f}, {v[1]:>6.2f})")
print()

# sesliler birbirine mi yakin, sessizlere mi?
import itertools
def uzaklik(a, b):
    return (C[stoi[a]] - C[stoi[b]]).pow(2).sum().sqrt().item()

sesli_ici = [uzaklik(a, b) for a, b in itertools.combinations(sesliler, 2)]
sessizler = ['b', 'c', 'd', 'f', 'g']
karisik = [uzaklik(a, b) for a in sesliler for b in sessizler]
print(f"sesli-sesli ortalama uzaklik   : {sum(sesli_ici)/len(sesli_ici):.3f}")
print(f"sesli-sessiz ortalama uzaklik  : {sum(karisik)/len(karisik):.3f}")
print()

# matplotlib varsa gorsel de kaydet
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0], C[:, 1], s=200, c='lightsteelblue')
    for i in range(27):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
                 ha="center", va="center", color='black', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title("Ogrenilen 2 boyutlu harf embedding'leri")
    plt.savefig('embedding.png', dpi=110, bbox_inches='tight')
    print("embedding.png kaydedildi")
except ImportError:
    print("(matplotlib yok, gorsel atlandi)")
print()


# --- isim uret ---
print("--- MLP'nin urettigi 15 isim ---")
gs = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(15):
    cikti, pencere = [], [0] * BAGLAM
    while True:
        with torch.no_grad():
            logits = ileri(torch.tensor([pencere]))
        p = F.softmax(logits, dim=1)
        ix = torch.multinomial(p, num_samples=1, generator=gs).item()
        if ix == 0:
            break
        cikti.append(itos[ix])
        pencere = pencere[1:] + [ix]
    print("  ", ''.join(cikti))
print()
print("Hafta 3 bigram'in urettikleri: cexze, konimittain, llayn, ka, da")
