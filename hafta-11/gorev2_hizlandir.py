# Gorev 2 - Egitimi hizlandir
# YZ50 Hafta 11  (video 1:22:18 - 2:14:55)
#
# DONANIM NOTU - bu gorevin tamami buna bagli
# Mail "bu hafta GPU gerekiyor, Colab'in T4'u yeterli" diyor. Bende
# GPU yok: iki cekirdekli bir Intel Xeon CPU. Ama CPU'yu acinca ilginc
# bir sey cikti: islemcide AMX (amx_bf16, amx_tile) var. AMX, Intel'in
# matris carpma birimi - yani GPU'daki tensor core'un CPU'daki karsiligi.
# Bu, bfloat16'nin burada EMULE EDILMEDIGI, donanimda calistigi anlamina
# geliyor. Yani "GPU yok" bu gorevi anlamsiz kilmiyor, sadece hangi
# adimin kazandirdigini degistiriyor.
#
# Videodaki bes adimin bu donanimdaki durumu:
#   TF32            -> CUDA'ya ozgu. Ampere tensor core'larinin 19 bitlik
#                      formati; CPU'da yok. ATLIYORUM. Ama ayni FIKRIN
#                      CPU karsiligi var ve onu olcuyorum (asagida).
#   bfloat16        -> VAR, AMX ile donanimda. OLCUYORUM.
#   flash attention -> PyTorch'un CPU icin flash cekirdegi var. OLCUYORUM.
#   torch.compile   -> CPU'da inductor C++ backend'i ile calisir. OLCUYORUM.
#   vocab 50304     -> donanimdan bagimsiz, OLCUYORUM.
#
# Yani bes adimin dordu olculuyor, biri (TF32) donanim sebebiyle
# atlaniyor ve yerine CPU esdegeri olculuyor. Mail "en az iki adimi
# olcmen yeterli" diyordu.

import gc
import json
import statistics
import time

import torch

from gpt2 import GPT, GPTConfig
from tokenizer import enc
from veri import VeriYukleyici

torch.manual_seed(1337)
B, T = 4, 256                 # adim basina 1024 token
ISINMA, OLCUM = 2, 6

print("=" * 80)
print("2) HIZLANDIRMA - ADIM SURESI VE TOKEN/SN")
print("=" * 80)
print(f"  olcum sekli: B={B}, T={T} -> adim basina {B*T} token")
print(f"  her ayarda {ISINMA} isinma adimi + {OLCUM} olculen adim, "
      f"medyan aliniyor")
print(f"  torch {torch.__version__}, cekirdek sayisi {torch.get_num_threads()}")
print()
print("""  OLCUM HIJYENI
  Videoda her adimdan once torch.cuda.synchronize() var. Sebebi: GPU
  cagrilari ASENKRON, yani Python satiri bitti diye hesap bitmis
  olmuyor; senkronize etmezsen kuyruga atma suresini olcersin. Bende
  GPU olmadigi icin buna gerek yok - CPU islemleri zaten senkron.
  Ama farki bilerek atliyorum, unuttugum icin degil.
""")

e = enc()
yuk = VeriYukleyici('input.txt', B, T, lambda s: e.encode(s), sessiz=True)
X, Y = [], []
for _ in range(ISINMA + OLCUM):
    a, b = yuk.sonraki()
    X.append(a)
    Y.append(b)

sonuc = dict(B=B, T=T, isinma=ISINMA, olcum=OLCUM,
             torch=torch.__version__, cekirdek=torch.get_num_threads(),
             satirlar=[])


def kaydet():
    json.dump(sonuc, open('sonuc_gorev2.json', 'w'),
              ensure_ascii=False, indent=1)


def olc(ad, aciklama, *, flash=False, bf16=False, derle=False,
        vocab=50257, matmul='highest'):
    """Bir ayari olcup satir dondurur."""
    torch.set_float32_matmul_precision(matmul)
    torch.manual_seed(1337)
    cfg = GPTConfig(block_size=T, vocab_size=vocab, flash=flash)
    model = GPT(cfg, weight_tying=True, ozel_init=True)
    opt, _ = model.configure_optimizers(0.1, 3e-4, sessiz=True)

    derleme_suresi = 0.0
    if derle:
        t0 = time.time()
        model = torch.compile(model)
        sarma = time.time() - t0      # asil derleme ilk cagrida oluyor

    def adim(i):
        opt.zero_grad(set_to_none=True)
        if bf16:
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                _, loss = model(X[i], Y[i])
        else:
            _, loss = model(X[i], Y[i])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss.item()

    # isinma (derleme varsa asil derleme burada oluyor)
    t0 = time.time()
    for i in range(ISINMA):
        kayip = adim(i)
    isinma_suresi = time.time() - t0
    if derle:
        derleme_suresi = sarma + isinma_suresi

    sureler = []
    for i in range(ISINMA, ISINMA + OLCUM):
        t0 = time.time()
        kayip = adim(i)
        sureler.append(time.time() - t0)

    med = statistics.median(sureler)
    satir = dict(ad=ad, aciklama=aciklama, medyan=med, en_iyi=min(sureler),
                 en_kotu=max(sureler), tok_sn=B * T / med,
                 sureler=sureler, loss=kayip, vocab=vocab,
                 flash=flash, bf16=bf16, derle=derle, matmul=matmul,
                 derleme_suresi=derleme_suresi)
    sonuc['satirlar'].append(satir)
    kaydet()                       # Hafta 10 dersi: her adimda kaydet

    taban = sonuc['satirlar'][0]['medyan']
    print(f"  {ad:<34}{med:>8.2f}s{B*T/med:>10.0f}{taban/med:>9.2f}x")
    del model, opt
    gc.collect()
    return satir


print(f"  {'ayar':<34}{'medyan':>9}{'tok/sn':>10}{'kazanc':>9}")
print("  " + "-" * 62)

olc('1. taban (fp32, elle attention)',
    'float32, attention elle hesaplaniyor, V=50257')

olc('2. + fp32 matmul "high"',
    'TF32 CUDA-ozel; CPU karsiligi: fp32 matmul icin dusuk hassasiyet',
    matmul='high')

olc('3. + bfloat16 autocast',
    'ileri gecis bf16, AMX donaniminda',
    matmul='high', bf16=True)

olc('4. + flash attention (SDPA)',
    'F.scaled_dot_product_attention, TxT matrisi bellege yazilmiyor',
    matmul='high', bf16=True, flash=True)

olc('5. + vocab 50257 -> 50304',
    '50304 = 128 * 393, "guzel sayi"',
    matmul='high', bf16=True, flash=True, vocab=50304)

print("  " + "-" * 62)
print()

# torch.compile en sona: derleme suresi uzun ve digerlerinden bagimsiz
print("  torch.compile olculuyor (derleme uzun surebilir)...", flush=True)
c = olc('6. + torch.compile',
        'inductor C++ backend, cekirdekleri birlestirip tekrar derliyor',
        matmul='high', bf16=True, flash=True, vocab=50304, derle=True)
print("  " + "-" * 62)

# =====================================================================
# OZET
# =====================================================================
s = sonuc['satirlar']
taban = s[0]
en_iyi = min(s, key=lambda r: r['medyan'])
# her adimin TEK BASINA kazanci (bir onceki satira gore)
for i, r in enumerate(s):
    r['adim_kazanci'] = (s[i-1]['medyan'] / r['medyan']) if i else 1.0
kaydet()

print()
print("=" * 80)
print("HANGI ADIM NE KAZANDIRDI (bir onceki satira gore)")
print("=" * 80)
print(f"  {'ayar':<34}{'medyan':>9}{'bu adim':>10}{'toplam':>9}")
print("  " + "-" * 62)
for r in s:
    print(f"  {r['ad']:<34}{r['medyan']:>9.2f}{r['adim_kazanci']:>9.2f}x"
          f"{taban['medyan']/r['medyan']:>8.2f}x")
print("  " + "-" * 62)
print(f"  en iyi ayar: {en_iyi['ad']}  "
      f"({taban['medyan']:.2f}s -> {en_iyi['medyan']:.2f}s, "
      f"{taban['medyan']/en_iyi['medyan']:.2f}x)")
print(f"  token/sn   : {taban['tok_sn']:.0f} -> {en_iyi['tok_sn']:.0f}")
if c['derleme_suresi']:
    kazanc_sn = taban['medyan'] - c['medyan']
    basabas = c['derleme_suresi'] / kazanc_sn if kazanc_sn > 0 else float('inf')
    print(f"  torch.compile derleme suresi: {c['derleme_suresi']:.0f}s "
          f"-> kendini {basabas:.0f} adimda amorti ediyor")
    sonuc['basabas'] = basabas
kaydet()
print()
print("  -> sonuc_gorev2.json kaydedildi")
