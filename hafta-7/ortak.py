# ortak.py - veri, tokenizer, batch cekme, loss olcumu
# YZ50 Hafta 7

import torch

BLOCK = 8        # baglam uzunlugu (kac karakter geriye bakiyoruz)
BATCH = 32
torch.manual_seed(1337)

metin = open('input.txt', 'r', encoding='utf-8').read()
harfler = sorted(set(metin))
V = len(harfler)

stoi = {h: i for i, h in enumerate(harfler)}
itos = {i: h for h, i in stoi.items()}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return ''.join(itos[i] for i in l)


veri = torch.tensor(encode(metin), dtype=torch.long)
n = int(0.9 * len(veri))
train_data, val_data = veri[:n], veri[n:]


def get_batch(bolum, block=BLOCK, batch=BATCH, g=None):
    d = train_data if bolum == 'train' else val_data
    ix = torch.randint(len(d) - block, (batch,), generator=g)
    x = torch.stack([d[i:i + block] for i in ix])
    y = torch.stack([d[i + 1:i + block + 1] for i in ix])
    return x, y


@torch.no_grad()
def loss_olc(model, tekrar=200, block=BLOCK):
    model.eval()
    sonuc = {}
    g = torch.Generator().manual_seed(1234)
    for bolum in ['train', 'val']:
        kayip = torch.zeros(tekrar)
        for k in range(tekrar):
            X, Y = get_batch(bolum, block=block, g=g)
            _, loss = model(X, Y)
            kayip[k] = loss.item()
        sonuc[bolum] = kayip.mean().item()
    model.train()
    return sonuc


def egit(model, adim=5000, lr=1e-3, block=BLOCK, sessiz=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1)
    for i in range(adim):
        xb, yb = get_batch('train', block=block, g=g)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not sessiz and i % 1000 == 0:
            s = loss_olc(model, tekrar=50, block=block)
            print(f"    adim {i:>5}  train {s['train']:.4f}  val {s['val']:.4f}")
    return model
