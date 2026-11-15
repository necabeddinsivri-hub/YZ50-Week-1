# Gorev 3 - GPT-3 makalesinin egitim ayarlari
# YZ50 Hafta 11  (video 2:14:55 - 2:46:52)
#
# GPT-3 makalesinin ek B'sinde egitim ayarlari yaziyor. GPT-2 makalesi
# bunlari vermiyor, o yuzden Karpathy GPT-3'un sayilarini kullaniyor.
# Dort parca var:
#   (1) AdamW, belirli beta'larla
#   (2) gradient clipping (norm 1.0)
#   (3) warmup'li cosine learning rate
#   (4) weight decay yalniz 2 boyutlu parametrelere
# Ustune bir de gradient accumulation: kucuk bellekte buyuk batch.

import gc
import json
import math
import os
import time

import torch

from gpt2 import GPT, GPTConfig
from tokenizer import enc
from veri import VeriYukleyici

torch.manual_seed(1337)
sonuc = {}


def kaydet():
    json.dump(sonuc, open('sonuc_gorev3.json', 'w'),
              ensure_ascii=False, indent=1)


# =====================================================================
# 3a) ADAMW VE WEIGHT DECAY AYRIMI
# =====================================================================
print("=" * 80)
print("3a) AdamW VE WEIGHT DECAY'IN HANGI PARAMETRELERE UYGULANDIGI")
print("=" * 80)
cfg = GPTConfig(block_size=256, vocab_size=50304, flash=True)
model = GPT(cfg, weight_tying=True, ozel_init=True)
opt, wd = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4,
                                     betas=(0.9, 0.95), eps=1e-8)
print()
print(f"""  BETA'LAR NE YAPIYOR
  beta1 = 0.9  : gradyanin kendi ortalamasi (momentum). Son ~10 adimin
                 gradyanini harmanliyor, tek bir batch'in gurultusune
                 kapilmayi azaltiyor.
  beta2 = 0.95 : gradyanin KARESININ ortalamasi. Her parametrenin kendi
                 adim boyunu belirliyor. Varsayilan 0.999, GPT-3 0.95
                 kullaniyor - yani daha kisa hafiza, degisime daha
                 hizli tepki. Dil modellerinde gradyan olcegi hizli
                 degistigi icin mantikli.
  eps   = 1e-8 : sifira bolmeyi engelleyen guvenlik payi.

  WEIGHT DECAY NEDEN SADECE MATRISLERE
  Yukarida {wd['decay_tensor']} tensor'e uyguladim, {wd['nodecay_tensor']} tensor'e uygulamadim.
  Uygulanan {wd['decay_param']:,} parametre, uygulanmayan {wd['nodecay_param']:,}.
  Yani parametrelerin %{100*wd['decay_param']/(wd['decay_param']+wd['nodecay_param']):.2f}'i decay aliyor.

  Sebebi: weight decay "bu agirligi kucuk tut" diyen bir ceza ve bu
  bir MATRIS icin anlamli - matrisin isi girdiyi dondurup olceklemek,
  buyuk degerler asiri uyum isareti. Bias ve LayerNorm gain'i ise
  matris degil: bias dagilimi kaydiriyor, gain olcegi geri veriyor.
  LayerNorm gain'i 1'den baslar ve 1 civarinda kalmasi DOGRUDUR; onu
  sifira cekmek o katmani kapatmak demek. O yuzden boyuta gore
  ayiriyorum: dim >= 2 ise matris, degilse bias/norm.
""".replace(',', '.'))
sonuc['wd'] = wd

# =====================================================================
# 3b) WARMUP'LI COSINE LEARNING RATE
# =====================================================================
print("=" * 80)
print("3b) WARMUP + COSINE LEARNING RATE")
print("=" * 80)
MAX_LR = 6e-4           # GPT-3 makalesinde 125M model icin
MIN_LR = MAX_LR * 0.1   # makalede son deger tepe degerin %10'u
WARMUP = 10
MAX_ADIM = 50


def lr_ver(adim):
    # 1) warmup: dogrusal olarak tirmaniyor
    if adim < WARMUP:
        return MAX_LR * (adim + 1) / WARMUP
    # 2) bittikten sonra sabit minimum
    if adim > MAX_ADIM:
        return MIN_LR
    # 3) arada cosine ile iniyor
    oran = (adim - WARMUP) / (MAX_ADIM - WARMUP)
    katsayi = 0.5 * (1.0 + math.cos(math.pi * oran))
    return MIN_LR + katsayi * (MAX_LR - MIN_LR)


egri_lr = [lr_ver(a) for a in range(MAX_ADIM + 5)]
print(f"  max_lr {MAX_LR:.1e}, min_lr {MIN_LR:.1e}, "
      f"warmup {WARMUP} adim, toplam {MAX_ADIM} adim")
print(f"  {'adim':>6}{'lr':>12}   bolge")
print("  " + "-" * 44)
for a in [0, 4, 9, 10, 20, 30, 40, 49, 50, 52]:
    b = 'warmup' if a < WARMUP else ('cosine' if a <= MAX_ADIM else 'sabit min')
    print(f"  {a:>6}{lr_ver(a):>12.3e}   {b}")
print("  " + "-" * 44)
print(f"""
  WARMUP NEDEN VAR
  Ilk adimda model hicbir sey bilmiyor ve gradyan buyuk. Tam hizla
  adim atarsam modeli daha baslamadan bozuyorum. Warmup, ilk adimlari
  kucuk tutup Adam'in ikinci moment tahmininin (beta2 ortalamasi)
  oturmasina zaman veriyor.

  COSINE NEDEN
  Sonuna dogru kucuk adimlar gerekiyor: model artik iyi bir yerde,
  buyuk adim onu o yerden firlatir. Cosine, basta yavas azalip ortada
  hizlanan, sonda yine yavaslayan yumusak bir inis veriyor.
""")
sonuc['lr'] = dict(max_lr=MAX_LR, min_lr=MIN_LR, warmup=WARMUP,
                   max_adim=MAX_ADIM, egri=egri_lr)
kaydet()

# =====================================================================
# 3c) GRADIENT ACCUMULATION: LOSS NEDEN BOLUNUYOR
# =====================================================================
print("=" * 80)
print("3c) GRADIENT ACCUMULATION - LOSS'U NEDEN BOLUYORUZ")
print("=" * 80)
print("""  KENDI CUMLELERIMLE
  F.cross_entropy varsayilan olarak ORTALAMA aliyor: batch'teki butun
  token'larin loss'unu topluyor ve token sayisina boluyor. Yani
  loss.backward() bana "bu batch'in ORTALAMA gradyani"ni veriyor.

  Gradient accumulation'da modele batch'i parca parca veriyorum ve
  gradyanlari ustune ekliyorum (zero_grad'i arada cagirmiyorum).
  Bolmezsem elde ettigim sey N tane ORTALAMANIN TOPLAMI oluyor. Oysa
  istedigim sey, buyuk batch'i tek seferde verseydim alacagim sey:
  butun token'larin ORTALAMASI. Toplami ortalamaya cevirmek icin N'e
  bolmem lazim - her mikro adimda loss'u N'e bolerek bunu yapiyorum.

  Neden onemli: bolmezsem gradyanin boyu N kat buyuk olur. Bu da
  ogrenme oranini sessizce N kat buyutmek demek. Yani accumulation
  adim sayisini degistirdigimde modelin ogrenme hizi da degisirdi -
  oysa accumulation sadece BELLEK hilesi olmali, optimizer'in
  gormemesi gereken bir ayrinti.

  SIMDI BUNU KANITLIYORUM. Bu bir aritmetik iddiasi, modelin
  buyuklugunden bagimsiz; o yuzden kucuk bir modelde olcuyorum ki
  buyuk batch'in tamami bellege sigsin ve ucunu yan yana koyabilsin.
""")
kcfg = GPTConfig(block_size=64, vocab_size=1024, n_layer=2, n_head=4,
                 n_embd=128, flash=True)
torch.manual_seed(7)
km = GPT(kcfg, weight_tying=True, ozel_init=True)
TOPLAM_B, N = 16, 4           # 16 dizi = 4 mikro batch x 4 dizi
MB = TOPLAM_B // N
xk = torch.randint(0, 1024, (TOPLAM_B, 64))
yk = torch.randint(0, 1024, (TOPLAM_B, 64))


def grad_vektoru():
    return torch.cat([p.grad.flatten() for p in km.parameters()
                      if p.grad is not None])


# A) tek seferde buyuk batch
km.zero_grad(set_to_none=True)
_, la = km(xk, yk)
la.backward()
gA = grad_vektoru().clone()

# B) 4 mikro batch, loss N'e BOLUNEREK
km.zero_grad(set_to_none=True)
lb = 0.0
for i in range(N):
    _, l = km(xk[i*MB:(i+1)*MB], yk[i*MB:(i+1)*MB])
    l = l / N
    lb += l.item()
    l.backward()
gB = grad_vektoru().clone()

# C) 4 mikro batch, BOLMEDEN
km.zero_grad(set_to_none=True)
lc = 0.0
for i in range(N):
    _, l = km(xk[i*MB:(i+1)*MB], yk[i*MB:(i+1)*MB])
    lc += l.item()
    l.backward()
gC = grad_vektoru().clone()

fark_AB = (gA - gB).abs().max().item()
fark_AC = (gA - gC).abs().max().item()
oran_CA = (gC.norm() / gA.norm()).item()
kos_AB = torch.nn.functional.cosine_similarity(gA, gB, dim=0).item()
print(f"  {'':<34}{'loss':>10}{'grad normu':>13}{'A ile fark':>13}")
print("  " + "-" * 72)
print(f"  {'A) tek seferde 16 dizi':<34}{la.item():>10.4f}"
      f"{gA.norm():>13.5f}{'-':>13}")
print(f"  {'B) 4 mikro batch, loss/4':<34}{lb:>10.4f}"
      f"{gB.norm():>13.5f}{fark_AB:>13.2e}")
print(f"  {'C) 4 mikro batch, bolmeden':<34}{lc:>10.4f}"
      f"{gC.norm():>13.5f}{fark_AC:>13.2e}")
print("  " + "-" * 72)
print(f"  B ile A'nin kosinus benzerligi : {kos_AB:.8f}")
print(f"  C'nin normu A'nin normunun     : {oran_CA:.4f} kati "
      f"(beklenen {N})")
print(f"""
  Sonuc net. Bolerek biriktirdigimde (B) gradyan tek seferde
  hesaplanandan (A) {fark_AB:.1e} kadar farkli - yani kayan nokta
  yuvarlamasi seviyesinde AYNI. Bolmeden biriktirdigimde (C) gradyan
  tam {oran_CA:.2f} kat buyuk cikiyor, yani N kat. Yonu ayni (ayni
  dogru uzerinde) ama boyu N kat; bu da ogrenme oranini N kat
  buyutmekle ayni sey.

  Hafta 5'te manuel geri yayilimda kurdugum "EXACT mi approx mi"
  olcutu burada yine ise yaradi: B ile A arasindaki fark EXACT degil
  ama 1e-8 mertebesinde, yani islem sirasindan gelen yuvarlama.
""")
sonuc['accum_kanit'] = dict(loss_A=la.item(), loss_B=lb, loss_C=lc,
                            norm_A=gA.norm().item(), norm_B=gB.norm().item(),
                            norm_C=gC.norm().item(), fark_AB=fark_AB,
                            fark_AC=fark_AC, oran_CA=oran_CA,
                            kosinus_AB=kos_AB, N=N, toplam_B=TOPLAM_B)
kaydet()
del km
gc.collect()

# =====================================================================
# 3d) HEPSINI BIRLESTIR: GERCEK EGITIM KOSUSU
# =====================================================================
print("=" * 80)
print("3d) HEPSI BIR ARADA - 124M MODELI EGIT")
print("=" * 80)
B, T = 4, 256
ACCUM = 8                     # 8 * 1024 = 8192 token'lik toplam batch
TOPLAM_TOKEN = B * T * ACCUM
print(f"""  AYARLAR
    mikro batch       : B={B}, T={T} -> {B*T} token
    accumulation adimi: {ACCUM}
    toplam batch      : {TOPLAM_TOKEN:,} token
    adim sayisi       : {MAX_ADIM}
    AdamW betas       : (0.9, 0.95), eps 1e-8
    weight decay      : 0.1, yalniz 2 boyutlu parametrelere
    gradient clipping : norm 1.0
    lr                : {MAX_LR:.1e} tepe, warmup {WARMUP}, cosine
    attention         : flash (SDPA)
    hassasiyet        : bfloat16 autocast (AMX)
    vocab             : {cfg.vocab_size}

  OLCEK NOTU - durust olmak gerekirse
  GPT-3 makalesi 125M model icin 0,5 MILYON token'lik batch kullaniyor,
  Karpathy videoda 524.288 token. Benimki {TOPLAM_TOKEN:,} token, yani
  makalenin yaklasik {524288//TOPLAM_TOKEN} kat altinda. Sebebi donanim:
  bir mikro adim bu CPU'da saniyeler suruyor, 524.288 token'lik tek bir
  optimizer adimi {524288//(B*T)} mikro adim demek. Bunun bedeli daha
  GURULTULU gradyan: kucuk batch'te gradyan yonu daha cok sapiyor ve
  grad norm'un daha dalgali olmasini bekliyorum. Asagidaki tabloda
  buna bakacagim.
""".replace(',', '.'))

e = enc()
yuk = VeriYukleyici('input.txt', B, T, lambda s: e.encode(s), sessiz=True)
torch.set_float32_matmul_precision('high')
torch.manual_seed(1337)
model = GPT(cfg, weight_tying=True, ozel_init=True)
opt, _ = model.configure_optimizers(0.1, MAX_LR, betas=(0.9, 0.95),
                                    eps=1e-8, sessiz=True)
# Gorev 2'de olctugum en iyi ayari kullaniyorum: bf16 + flash +
# vocab 50304 + torch.compile. Derleme 72 saniye tutuyor ve kendini
# 21 adimda amorti ediyor; burada 400 mikro adim atacagim, yani
# fazlasiyla deger.
print("  torch.compile derleniyor (ilk adim uzun surecek)...", flush=True)
t_derle = time.time()
model = torch.compile(model)
sonuc['derleme_sarma'] = time.time() - t_derle
print(f"  {'adim':>5}{'loss':>9}{'grad_norm':>11}{'lr':>11}"
      f"{'sure':>8}{'tok/sn':>9}")
print("  " + "-" * 56)
kayit = []
t_bas = time.time()
for adim in range(MAX_ADIM):
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    birikmis = 0.0
    for mikro in range(ACCUM):
        xb, yb = yuk.sonraki()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        # BOLME: ortalamalarin toplami degil, toplamin ortalamasi
        loss = loss / ACCUM
        birikmis += loss.item()
        loss.backward()
    # gradient clipping: gradyanin normunu 1.0'i gecmeyecek sekilde kis
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = lr_ver(adim)
    for g in opt.param_groups:
        g['lr'] = lr
    opt.step()
    dt = time.time() - t0
    kayit.append(dict(adim=adim, loss=birikmis, norm=norm.item(), lr=lr,
                      dt=dt, tok_sn=TOPLAM_TOKEN / dt,
                      token=yuk.verilen_token))
    sonuc['egitim'] = dict(B=B, T=T, accum=ACCUM, toplam_token=TOPLAM_TOKEN,
                           kayit=kayit)
    kaydet()                  # Hafta 10 dersi: her adimda diske yaz
    if adim < 5 or adim % 5 == 4 or adim == MAX_ADIM - 1:
        print(f"  {adim:>5}{birikmis:>9.4f}{norm.item():>11.4f}{lr:>11.3e}"
              f"{dt:>7.1f}s{TOPLAM_TOKEN/dt:>9.0f}", flush=True)
print("  " + "-" * 56)
sure = time.time() - t_bas
print(f"  {MAX_ADIM} adim {sure/60:.1f} dakika surdu, "
      f"{yuk.verilen_token:,} token islendi".replace(',', '.'))
print(f"  loss {kayit[0]['loss']:.4f} -> {kayit[-1]['loss']:.4f}  "
      f"(taban ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.4f})")
normlar = [k['norm'] for k in kayit]
print(f"  grad norm: ilk {normlar[0]:.3f}, en yuksek {max(normlar):.3f}, "
      f"son {normlar[-1]:.3f}")
kirpilan = sum(1 for n in normlar if n > 1.0)
print(f"  1.0'in uzerine cikip KIRPILAN adim sayisi: {kirpilan}/{MAX_ADIM}")
sonuc['egitim'].update(sure=sure, kirpilan=kirpilan,
                       taban=math.log(cfg.vocab_size))
kaydet()
print()
print("  -> sonuc_gorev3.json kaydedildi")
