# Gorev 2 ve 3 - Dogrulama loss'u, ornek metin ve egitim
# YZ50 Hafta 12  (video 3:23:10 - 3:43:05)
#
# Hafta 11'de hizlandirdigim GPT-2'yi (124M) burada gercek veride
# egitiyorum. Hafta 11'in olcum tablosundaki en iyi satiri kurdum:
#   bfloat16 autocast (AMX) + flash attention + vocab 50.304 +
#   torch.compile  ->  Hafta 11'de 405 token/sn
# GPT-3'un egitim ayarlari da Hafta 11'den: AdamW (0.9, 0.95), weight
# decay yalniz 2 boyutlu parametrelere, clipping 1.0, warmup + cosine.
#
# YENI OLAN IKI SEY (Gorev 2):
#   - belli araliklarla VAL loss: model o token'lari hic gormuyor
#   - belli araliklarla ORNEK METIN: ayni baslangic, ayni tohum, boylece
#     degisen tek sey modelin kendisi

import json
import math
import os
import time

import torch
from torch.nn import functional as F

from bicim import bin as bn
from gpt2 import GPT, GPTConfig
from tokenizer import enc
from veri import ShardYukleyici

torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')

# ------------------------------- ayarlar -------------------------------
B, T, ACCUM = 4, 256, 4                  # 4 x 256 x 4 = 4.096 token/adim
TOPLAM = B * T * ACCUM
MAX_ADIM = int(os.environ.get('MAX_ADIM', 330))
WARMUP = 15
MAX_LR, MIN_LR = 6e-4, 6e-5
VAL_ARALIK, VAL_ADIM = 25, 10            # 10 x 1.024 = 10.240 val token'i
ORNEK_ARALIK = 50
SURE_SINIRI = 80 * 60                    # guvenlik: 80 dakikayi gecme
DERLE = os.environ.get('DERLE', '1') == '1'
CKPT = 'veri/checkpoint'
os.makedirs(CKPT, exist_ok=True)

e = enc()
egitim = ShardYukleyici(B, T, 'train')
val = ShardYukleyici(B, T, 'val')


def lr_ver(adim):
    if adim < WARMUP:
        return MAX_LR * (adim + 1) / WARMUP
    if adim >= MAX_ADIM:
        return MIN_LR
    oran = (adim - WARMUP) / (MAX_ADIM - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * oran)) * (MAX_LR - MIN_LR)


model = GPT(GPTConfig(vocab_size=50304, flash=True), weight_tying=True,
            ozel_init=True)
ham_model = model          # derlenmemis hali: ornek uretmek icin
opt, wd = model.configure_optimizers(0.1, MAX_LR, betas=(0.9, 0.95),
                                     eps=1e-8, sessiz=True)
if DERLE:
    model = torch.compile(model)

print("=" * 80)
print("2-3) GPT-2 (124M) FINEWEB-EDU'DA EGITIM")
print("=" * 80)
print(f"  parametre       : {bn(ham_model.param_sayisi())}")
print(f"  egitim verisi   : {bn(egitim.toplam_token)} token "
      f"({len(egitim.dosyalar)} dosya)")
print(f"  val verisi      : {bn(val.toplam_token)} token (1 dosya)")
print(f"  adim basina     : {B} x {T} x {ACCUM} = {bn(TOPLAM)} token")
print(f"  plan            : {MAX_ADIM} adim = {bn(MAX_ADIM*TOPLAM)} token")
print(f"  lr              : {MAX_LR:.0e} tepe, warmup {WARMUP}, cosine "
      f"-> {MIN_LR:.0e}")
print(f"  torch.compile   : {'acik' if DERLE else 'kapali'}")
print()

BASLANGIC = "Hello, I'm a language model,"


@torch.no_grad()
def ornek_uret(n=2, uzunluk=32, top_k=50, tohum=42):
    """Ayni baslangic, ayni tohum: degisen tek sey model."""
    ham_model.eval()
    g = torch.Generator().manual_seed(tohum)
    x = torch.tensor([e.encode(BASLANGIC)] * n, dtype=torch.long)
    while x.size(1) < len(e.encode(BASLANGIC)) + uzunluk:
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            logits, _ = ham_model(x)
        p = F.softmax(logits[:, -1, :].float(), dim=-1)
        p[:, 50257:] = 0           # yuvarlama icin eklenen 47 satir hic token degil
        tp, ti = torch.topk(p, top_k, dim=-1)
        sec = torch.multinomial(tp, 1, generator=g)
        x = torch.cat([x, torch.gather(ti, -1, sec)], dim=1)
    ham_model.train()
    return [e.decode(r.tolist()) for r in x]


@torch.no_grad()
def val_loss():
    model.eval()
    val.basa_al()             # her olcumde AYNI val token'lari
    top = 0.0
    for _ in range(VAL_ADIM):
        x, y = val.sonraki()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            _, loss = model(x, y)
        top += loss.item()
    model.train()
    return top / VAL_ADIM


kayit, vals, ornekler = [], [], []
sonuc = dict(B=B, T=T, accum=ACCUM, toplam=TOPLAM, max_adim=MAX_ADIM,
             warmup=WARMUP, max_lr=MAX_LR, min_lr=MIN_LR,
             val_aralik=VAL_ARALIK, val_adim=VAL_ADIM, baslangic=BASLANGIC,
             param=ham_model.param_sayisi(), wd=wd,
             train_token_mevcut=egitim.toplam_token,
             val_token_mevcut=val.toplam_token, taban=math.log(50257))


def kaydet():
    sonuc.update(kayit=kayit, val=vals, ornekler=ornekler)
    json.dump(sonuc, open('sonuc_gorev23.json', 'w'),
              ensure_ascii=False, indent=1)


t_bas = time.time()
egitim_suresi = 0.0          # yalniz optimizer adimlari (val/ornek haric)
print(f"  {'adim':>5}{'loss':>9}{'val':>9}{'norm':>8}{'lr':>10}"
      f"{'sure':>8}{'tok/sn':>8}")
print("  " + "-" * 57)
for adim in range(MAX_ADIM + 1):
    son = adim == MAX_ADIM

    if adim % VAL_ARALIK == 0 or son:
        t0 = time.time()
        v = val_loss()
        vals.append(dict(adim=adim, val=v, token=egitim.verilen_token,
                         t=time.time() - t_bas))
        print(f"  {adim:>5}{'':>9}{v:>9.4f}   (val, {time.time()-t0:.0f} sn)",
              flush=True)
    if adim % ORNEK_ARALIK == 0 or son:
        metinler = ornek_uret()
        ornekler.append(dict(adim=adim, token=egitim.verilen_token,
                             metin=metinler))
        for m in metinler:
            print(f"        > {m!r}", flush=True)
        kaydet()
    if son or time.time() - t_bas > SURE_SINIRI:
        break

    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    birikmis = 0.0
    for _ in range(ACCUM):
        x, y = egitim.sonraki()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / ACCUM
        birikmis += loss.item()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = lr_ver(adim)
    for g in opt.param_groups:
        g['lr'] = lr
    opt.step()
    dt = time.time() - t0
    if adim > 0:                       # adim 0 derlemeyi iceriyor
        egitim_suresi += dt
    kayit.append(dict(adim=adim, loss=birikmis, norm=norm.item(), lr=lr,
                      dt=dt, tok_sn=TOPLAM / dt, token=egitim.verilen_token))
    if adim < 3 or adim % 10 == 9:
        print(f"  {adim:>5}{birikmis:>9.4f}{'':>9}{norm.item():>8.3f}"
              f"{lr:>10.2e}{dt:>7.1f}s{TOPLAM/dt:>8.0f}", flush=True)
    kaydet()

toplam_sure = time.time() - t_bas
adim_sayisi = len(kayit)
olculen = [k for k in kayit if k['adim'] > 0]
tok_sn = len(olculen) * TOPLAM / max(egitim_suresi, 1e-9)
sonuc.update(toplam_sure=toplam_sure, egitim_suresi=egitim_suresi,
             adim_sayisi=adim_sayisi, gorulen_token=egitim.verilen_token,
             tok_sn=tok_sn, derleme_adimi=kayit[0]['dt'] if kayit else None,
             son_train=sum(k['loss'] for k in kayit[-10:]) / min(10, len(kayit)),
             son_val=vals[-1]['val'], devir=egitim.devir,
             gorulen_oran=egitim.verilen_token / egitim.toplam_token)
kaydet()
torch.save(ham_model.state_dict(), os.path.join(CKPT, 'model_son.pt'))

print("  " + "-" * 57)
print(f"  toplam sure       : {toplam_sure/60:.1f} dk "
      f"(saf egitim {egitim_suresi/60:.1f} dk)")
print(f"  optimizer adimi   : {adim_sayisi}")
print(f"  gorulen token     : {bn(egitim.verilen_token)} "
      f"(egitim verisinin %{100*sonuc['gorulen_oran']:.0f}'i)")
print(f"  token / saniye    : {tok_sn:.0f} (derleme adimi haric)")
print(f"  son train loss    : {sonuc['son_train']:.4f} (son 10 adim ort.)")
print(f"  son val loss      : {sonuc['son_val']:.4f}")
print(f"  -> {CKPT}/model_son.pt, sonuc_gorev23.json")
