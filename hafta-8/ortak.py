# ortak.py - veri, tokenizer, batch cekme, loss olcumu, egitim dongusu
# YZ50 Hafta 8
#
# Hafta 7'deki ortak.py'nin ayni isi yapan hali. Iki fark var:
#   1) artik hangi dosyadan okuyacagimi secebiliyorum (Shakespeare / Turkce)
#   2) egitim dongusu gecen sureyi de olcuyor (Gorev 3 icin lazim)

import time
import torch

BLOCK = 8          # Hafta 7 ile ayni baglam uzunlugu (karsilastirma icin sabit)
BATCH = 32
torch.manual_seed(1337)


class Veri:
    """Bir metin dosyasini alir, karakter tokenizer'i kurar, train/val boler."""

    def __init__(self, yol='input.txt'):
        self.metin = open(yol, 'r', encoding='utf-8').read()
        self.harfler = sorted(set(self.metin))
        self.V = len(self.harfler)
        self.stoi = {h: i for i, h in enumerate(self.harfler)}
        self.itos = {i: h for h, i in self.stoi.items()}
        d = torch.tensor(self.encode(self.metin), dtype=torch.long)
        n = int(0.9 * len(d))
        self.train_data, self.val_data = d[:n], d[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join(self.itos[i] for i in l)

    def get_batch(self, bolum, block=BLOCK, batch=BATCH, g=None):
        d = self.train_data if bolum == 'train' else self.val_data
        ix = torch.randint(len(d) - block, (batch,), generator=g)
        x = torch.stack([d[i:i + block] for i in ix])
        y = torch.stack([d[i + 1:i + block + 1] for i in ix])
        return x, y


# varsayilan veri: Shakespeare (Hafta 7 ile ayni dosya)
veri = Veri('input.txt')
V = veri.V
encode, decode, get_batch = veri.encode, veri.decode, veri.get_batch
train_data, val_data, metin, harfler = (veri.train_data, veri.val_data,
                                        veri.metin, veri.harfler)


@torch.no_grad()
def loss_olc(model, d=None, tekrar=200, block=BLOCK, batch=BATCH):
    """train ve val loss'u 200 batch ortalamasi olarak olcer.

    Tek batch'in loss'u cok gurultulu; 200 batch ortalamasi alinca
    kucuk farklar (0,01 mertebesi) anlamli hale geliyor. Olcumde her
    seferinde ayni generator tohumunu kullaniyorum ki iki model ayni
    batch'ler uzerinde karsilastirilsin.
    """
    d = d or veri
    model.eval()
    sonuc = {}
    g = torch.Generator().manual_seed(1234)
    for bolum in ['train', 'val']:
        kayip = torch.zeros(tekrar)
        for k in range(tekrar):
            X, Y = d.get_batch(bolum, block=block, batch=batch, g=g)
            _, loss = model(X, Y)
            kayip[k] = loss.item()
        sonuc[bolum] = kayip.mean().item()
    model.train()
    return sonuc


def egit(model, adim=5000, lr=1e-3, block=BLOCK, batch=BATCH, d=None,
         sessiz=False, her=1000):
    """AdamW ile egitir, gecen sureyi dondurur."""
    d = d or veri
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1)
    t0 = time.time()
    for i in range(adim):
        xb, yb = d.get_batch('train', block=block, batch=batch, g=g)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not sessiz and i % her == 0:
            s = loss_olc(model, d=d, tekrar=50, block=block, batch=batch)
            print(f"    adim {i:>5}  train {s['train']:.4f}  val {s['val']:.4f}"
                  f"  ({time.time() - t0:.0f} sn)")
    return time.time() - t0


def param_sayisi(model):
    return sum(p.nelement() for p in model.parameters())
