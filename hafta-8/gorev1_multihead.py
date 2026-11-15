# Gorev 1 - Tek head'i multi-head attention'a cevir
# YZ50 Hafta 8  (video 1:21:59 - 1:24:25)
#
# Yaptigim sey: 32 boyutluk tek head'i, 4 tane 8 boyutluk head'e bolmek.
# Toplam boyut AYNI kaliyor. Sadece "kim kime bakiyor" karari 4 ayri
# kanalda bagimsizca veriliyor, sonra ciktilar yan yana ekleniyor.

import json
import torch
from ortak import (veri, V, BLOCK, BATCH, loss_olc, egit, encode, decode,
                   param_sayisi, harfler, metin)
from katmanlar import (BigramDilModeli, AblasyonModeli, N_EMBD, uret)

ADIM, LR = 5000, 1e-3          # Hafta 7 ile bire bir ayni egitim butcesi
sonuc = {}

print("=" * 70)
print("GOREV 1 - TEK HEAD -> MULTI-HEAD")
print("=" * 70)
print(f"  veri          : input.txt (Tiny Shakespeare)")
print(f"  karakter      : {len(metin):,}".replace(',', '.'))
print(f"  alfabe (V)    : {V}")
print(f"  block_size    : {BLOCK}   batch: {BATCH}")
print(f"  n_embd        : {N_EMBD}")
print(f"  egitim        : {ADIM} adim, AdamW, lr={LR}")
print()
print("  Hafta 7 ile ayni ayarlar. Tek degisken multi-head olsun diye")
print("  block/batch/n_embd/adim/lr'ye hic dokunmuyorum.")
print()


# ---------------------------------------------------------------------
# 1) BIGRAM TABANI  (tablo satir 1)
# ---------------------------------------------------------------------
print("-" * 70)
print("1) BIGRAM TABANI")
print("-" * 70)
torch.manual_seed(1337)
bigram = BigramDilModeli()
print(f"  parametre: {param_sayisi(bigram):,}".replace(',', '.'))
sure = egit(bigram, adim=ADIM, lr=1e-2, sessiz=True)     # bigram lr=1e-2
s = loss_olc(bigram)
print(f"  train {s['train']:.4f}   val {s['val']:.4f}   ({sure:.0f} sn)")
sonuc['1_bigram'] = dict(ad='bigram (taban)', param=param_sayisi(bigram),
                         train=s['train'], val=s['val'], sure=sure)
print()


# ---------------------------------------------------------------------
# 2) TEK HEAD  (tablo satir 2) - Hafta 7'nin modeli
# ---------------------------------------------------------------------
print("-" * 70)
print("2) TEK HEAD SELF-ATTENTION  (Hafta 7)")
print("-" * 70)
torch.manual_seed(1337)
tek = AblasyonModeli('tek_head')
print(f"  parametre: {param_sayisi(tek):,}".replace(',', '.'))
print(f"  yapi     : 1 head x {N_EMBD} boyut")
sure = egit(tek, adim=ADIM, lr=LR, sessiz=True)
s_tek = loss_olc(tek)
print(f"  train {s_tek['train']:.4f}   val {s_tek['val']:.4f}   ({sure:.0f} sn)")
sonuc['2_tek_head'] = dict(ad='tek head', param=param_sayisi(tek),
                           train=s_tek['train'], val=s_tek['val'], sure=sure)
print()


# ---------------------------------------------------------------------
# 3) MULTI-HEAD  (tablo satir 3)
# ---------------------------------------------------------------------
print("-" * 70)
print("3) MULTI-HEAD ATTENTION  (4 head x 8 boyut)")
print("-" * 70)
torch.manual_seed(1337)
multi = AblasyonModeli('multi_head', n_head=4)
print(f"  parametre: {param_sayisi(multi):,}".replace(',', '.'))
print(f"  yapi     : 4 head x {N_EMBD // 4} boyut  (toplam {N_EMBD}, ayni)")
print()
print("  Boyutlarin akisi (B=32, T=8, C=32):")
print("    girdi x            : (32, 8, 32)")
print("    her head cikti     : (32, 8,  8)      <- 8 = 32/4")
print("    4 head yan yana    : (32, 8, 32)      <- torch.cat(dim=-1)")
print("    proj               : (32, 8, 32)")
print()
sure = egit(multi, adim=ADIM, lr=LR, sessiz=True)
s_multi = loss_olc(multi)
print(f"  train {s_multi['train']:.4f}   val {s_multi['val']:.4f}   ({sure:.0f} sn)")
sonuc['3_multi_head'] = dict(ad='multi-head (4x8)', param=param_sayisi(multi),
                             train=s_multi['train'], val=s_multi['val'],
                             sure=sure)
print()


# ---------------------------------------------------------------------
# YAN YANA KARSILASTIRMA  (gorevde istenen)
# ---------------------------------------------------------------------
print("=" * 70)
print("TEK HEAD vs MULTI-HEAD")
print("=" * 70)
print(f"  {'model':<22}{'param':>9}{'train':>10}{'val':>10}")
print("  " + "-" * 51)
for k in ['2_tek_head', '3_multi_head']:
    r = sonuc[k]
    print(f"  {r['ad']:<22}{r['param']:>9,}{r['train']:>10.4f}{r['val']:>10.4f}"
          .replace(',', '.'))
print("  " + "-" * 51)
fark = s_tek['val'] - s_multi['val']
print(f"  val kazanci: {fark:+.4f}")
print()
print(f"  Parametre farki sadece {param_sayisi(multi) - param_sayisi(tek):,} "
      f"(proj katmani).".replace(',', '.'))
print("  Yani kazanc modelin BUYUMESINDEN degil, isi BOLMESINDEN geliyor.")
print()


# ---------------------------------------------------------------------
# HEAD'LER GERCEKTEN FARKLI SEYLERE MI BAKIYOR?
# ---------------------------------------------------------------------
print("=" * 70)
print("4 HEAD NEYE BAKIYOR?")
print("=" * 70)
cumle = "First Ci"
idx = torch.tensor([encode(cumle)])
multi.eval()
with torch.no_grad():
    multi(idx)

semalar = " .:-=+*#%@"
agirliklar = []
for hi, h in enumerate(multi.govde.heads):
    wei = h.son_wei[0]                     # (8,8)
    agirliklar.append(wei.numpy())
    print(f"  --- head {hi} ---")
    print("        " + "".join(f"{c:>3}" for c in cumle))
    for i, c in enumerate(cumle):
        satir = ""
        for j in range(8):
            if j > i:
                satir += "   "
            else:
                k = int(wei[i, j].item() * (len(semalar) - 1)
                        / max(wei[i].max().item(), 1e-9))
                satir += f"  {semalar[k]}"
        print(f"  {c!r:>5}{satir}")
    # son satirin en cok baktigi yer
    w = wei[7, :8]
    print(f"        son token ({cumle[7]!r}) en cok {w.argmax().item()}. "
          f"token'a bakiyor ({cumle[w.argmax().item()]!r}, {w.max():.3f}); "
          f"duz ortalama 0.125 olurdu")
    print()

# head'ler birbirinden ne kadar farkli? son satirlarin ortalama mutlak farki
import itertools
print("  Head'lerin son satirlari arasindaki ortalama mutlak fark:")
for a, b in itertools.combinations(range(4), 2):
    d = abs(agirliklar[a][7] - agirliklar[b][7]).mean()
    print(f"    head {a} - head {b}: {d:.4f}")
print()
print("  Sifira yakin olsalardi head'ler ayni seyi ogrenmis olurdu ve")
print("  multi-head bir ise yaramazdi. Degerler sifirdan uzak: head'ler")
print("  gercekten farkli desenlere yerlesmis.")
print()

# uretim ornegi
gg = torch.Generator().manual_seed(42)
print("  multi-head modelin urettigi 300 karakter:")
print(decode(uret(multi, torch.zeros((1, 1), dtype=torch.long), 300,
                  g=gg)[0].tolist()))
print()

json.dump(sonuc, open('sonuclar.json', 'w'), indent=1, ensure_ascii=False)
import numpy as np
np.save('head_agirliklari.npy', np.stack(agirliklar))
print("  -> sonuclar.json ve head_agirliklari.npy kaydedildi")
