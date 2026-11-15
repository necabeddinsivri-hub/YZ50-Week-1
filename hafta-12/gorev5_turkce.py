# Gorev 5 (ek) - Ayni modeli Turkce metinle egit, Turkce modeli olc
# YZ50 Hafta 12
#
# Iki soru var:
#   (1) Ayni GPT-2'yi Turkce metinle egitince ne uretiyor?
#   (2) HellaSwag Ingilizce; Turkce modeli nasil olcerdim?
#
# (2) icin olcegi Hafta 9 ve 11'den tasiyorum: KARAKTER BASINA BIT.
# Ayrilmis (modelin gormedigi) Turkce metinde "bir karakteri kodlamak
# kac bit tutuyor"u olcuyorum. Uc modeli ayni olcekte koyuyorum:
#   - bu gorevde Turkce egittigim model
#   - Gorev 3'te FineWeb-Edu'da egittigim model
#   - OpenAI'in GPT-2'si
# Ve ayni olcumu Ingilizce val dosyasinda da yapiyorum: boylece 2 x 3'luk
# bir tablo cikiyor ve her modelin hangi dilde ne bildigi gorunuyor.

import json
import math
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

import agirlik
from bicim import bin as bn
from gpt2 import GPT, GPTConfig
from tokenizer import enc
from veri import ShardYukleyici

torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')
e = enc()
EOT = e.eot_token
KL_TR = 'veri/shardlar_tr'
os.makedirs(KL_TR, exist_ok=True)
sonuc = {}


def kaydet():
    json.dump(sonuc, open('sonuc_gorev5.json', 'w'), ensure_ascii=False,
              indent=1)


# =====================================================================
# 5a) TURKCE METNI TOKEN'LA, BOL
# =====================================================================
print("=" * 78)
print("5a) TURKCE METIN -> GPT-2 TOKENIZER -> DOSYALAR")
print("=" * 78)
metin = open('turkce.txt', encoding='utf-8').read()
tok = np.array([EOT] + e.encode_ordinary(metin), dtype=np.uint16)
n_val = len(tok) // 10                   # ilk %10 val
np.save(os.path.join(KL_TR, 'turkce_val_000000.npy'), tok[:n_val])
np.save(os.path.join(KL_TR, 'turkce_train_000001.npy'), tok[n_val:])
print(f"  karakter : {bn(len(metin))}")
print(f"  token    : {bn(len(tok))}  ({len(metin)/len(tok):.2f} karakter/token)")
print(f"  val      : {bn(n_val)} token, egitim: {bn(len(tok)-n_val)} token")
print(f"  Karsilastirma: FineWeb-Edu'da token basina "
      f"{json.load(open('sonuc_gorev1.json'))['kar_token']:.2f} karakter. Ayni "
      f"tokenizer Turkce'yi iki kat sik boluyor (Hafta 10-11).")
sonuc['veri'] = dict(karakter=len(metin), token=int(len(tok)),
                     val=int(n_val), train=int(len(tok) - n_val),
                     kar_token=len(metin) / len(tok))
kaydet()

# =====================================================================
# 5b) EGITIM
# =====================================================================
B, T, ACCUM = 4, 256, 4
TOPLAM = B * T * ACCUM
MAX_ADIM = int(os.environ.get('TR_ADIM', 170))
WARMUP, MAX_LR, MIN_LR = 15, 6e-4, 6e-5
VAL_ARALIK = 20
BASLANGICLAR = ["Türkiye'nin başkenti", "Bugün hava çok güzel,"]

egitim = ShardYukleyici(B, T, 'train', klasor=KL_TR, onek='turkce')
val = ShardYukleyici(B, T, 'val', klasor=KL_TR, onek='turkce')


def lr_ver(a):
    if a < WARMUP:
        return MAX_LR * (a + 1) / WARMUP
    o = min(1.0, (a - WARMUP) / (MAX_ADIM - WARMUP))
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * o)) * (MAX_LR - MIN_LR)


ham = GPT(GPTConfig(vocab_size=50304, flash=True), weight_tying=True,
          ozel_init=True)
opt, _ = ham.configure_optimizers(0.1, MAX_LR, sessiz=True)
model = torch.compile(ham)


@torch.no_grad()
def uret(m, baslangic, n=2, uzunluk=40, tohum=42):
    m.eval()
    g = torch.Generator().manual_seed(tohum)
    x = torch.tensor([e.encode(baslangic)] * n, dtype=torch.long)
    hedef = x.size(1) + uzunluk
    while x.size(1) < hedef:
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            lg, _ = m(x)
        p = F.softmax(lg[:, -1, :].float(), dim=-1)
        p[:, 50257:] = 0
        tp, ti = torch.topk(p, 50, dim=-1)
        x = torch.cat([x, torch.gather(ti, -1, torch.multinomial(tp, 1, generator=g))], 1)
    m.train()
    return [e.decode(r.tolist()) for r in x]


@torch.no_grad()
def val_loss(adim=10):
    model.eval()
    val.basa_al()
    t = 0.0
    for _ in range(adim):
        x, y = val.sonraki()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            _, l = model(x, y)
        t += l.item()
    model.train()
    return t / adim


print()
print("=" * 78)
print(f"5b) AYNI 124M MODELI TURKCE METINDE EGIT ({MAX_ADIM} adim)")
print("=" * 78)
print(f"  egitim verisi {bn(egitim.toplam_token)} token; plan "
      f"{bn(MAX_ADIM*TOPLAM)} token = "
      f"{MAX_ADIM*TOPLAM/egitim.toplam_token:.1f} devir")
kayit, vals, ornekler = [], [], []
t_bas = time.time()
saf = 0.0
for adim in range(MAX_ADIM + 1):
    son = adim == MAX_ADIM
    if adim % VAL_ARALIK == 0 or son:
        v = val_loss()
        vals.append(dict(adim=adim, val=v, token=egitim.verilen_token,
                         devir=egitim.verilen_token / egitim.toplam_token))
        print(f"  {adim:>4}  val {v:.4f}  (devir "
              f"{egitim.verilen_token/egitim.toplam_token:.2f})", flush=True)
    if adim in (0, MAX_ADIM // 2) or son:
        ornekler.append(dict(adim=adim, metin={b: uret(ham, b) for b in BASLANGICLAR}))
        for b, mm in ornekler[-1]['metin'].items():
            print(f"        > {mm[0]!r}", flush=True)
    sonuc['egitim'] = dict(kayit=kayit, val=vals, ornekler=ornekler,
                           max_adim=MAX_ADIM, toplam=TOPLAM)
    kaydet()
    if son:
        break
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    bir = 0.0
    for _ in range(ACCUM):
        x, y = egitim.sonraki()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            _, l = model(x, y)
        (l / ACCUM).backward()
        bir += l.item() / ACCUM
    norm = torch.nn.utils.clip_grad_norm_(ham.parameters(), 1.0)
    for g in opt.param_groups:
        g['lr'] = lr_ver(adim)
    opt.step()
    dt = time.time() - t0
    if adim > 0:
        saf += dt
    kayit.append(dict(adim=adim, loss=bir, norm=norm.item(), dt=dt,
                      token=egitim.verilen_token))
    if adim % 20 == 19:
        print(f"  {adim:>4}  train {bir:.4f}  {TOPLAM/dt:.0f} tok/sn", flush=True)

sonuc['egitim'].update(sure=time.time() - t_bas, saf=saf,
                       gorulen=egitim.verilen_token,
                       tok_sn=(len(kayit) - 1) * TOPLAM / max(saf, 1e-9),
                       en_iyi_val=min(v['val'] for v in vals),
                       en_iyi_val_adim=min(vals, key=lambda v: v['val'])['adim'])
kaydet()
torch.save(ham.state_dict(), 'veri/checkpoint/model_turkce.pt')

# =====================================================================
# 5c) TURKCE MODELI NASIL OLCERIM: KARAKTER BASINA BIT, 2 x 3 TABLO
# =====================================================================
print()
print("=" * 78)
print("5c) KARAKTER BASINA BIT: 3 MODEL x 2 DIL (ayrilmis val metinleri)")
print("=" * 78)


@torch.no_grad()
def bpc(m, yukleyici, n=20):
    """Ayni token'lar uzerinde nat/token ve bit/karakter."""
    m.eval()
    yukleyici.basa_al()
    nat, tokn, kar = 0.0, 0, 0
    for _ in range(n):
        x, y = yukleyici.sonraki()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            lg, _ = m(x)
        l = F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.view(-1),
                            reduction='sum')
        nat += l.item()
        tokn += y.numel()
        kar += sum(len(e.decode(r.tolist())) for r in y)
    return dict(nat_token=nat / tokn, bpc=nat / kar / math.log(2),
                token=tokn, karakter=kar)


ingilizce_val = ShardYukleyici(4, 256, 'val')       # FineWeb-Edu val
modeller = {}
m_en = GPT(GPTConfig(vocab_size=50304, flash=True), weight_tying=True)
m_en.load_state_dict(torch.load('veri/checkpoint/model_son.pt', map_location='cpu'))
modeller['Benim (FineWeb-Edu, 1 saat)'] = m_en
modeller['Benim (Turkce, bu gorev)'] = ham
modeller["OpenAI GPT-2 (124M)"] = agirlik.openai_gpt2()

tablo = {}
for ad, m in modeller.items():
    tablo[ad] = dict(tr=bpc(m, val), en=bpc(m, ingilizce_val))
    print(f"  {ad:<32} TR {tablo[ad]['tr']['bpc']:.3f} bit/kar   "
          f"EN {tablo[ad]['en']['bpc']:.3f} bit/kar", flush=True)
sonuc['bpc'] = tablo
kaydet()
print("  -> sonuc_gorev5.json, veri/checkpoint/model_turkce.pt")
