# Gorev 4 - HellaSwag'de olc: once kendi modelim, sonra OpenAI'in GPT-2'si
# YZ50 Hafta 12  (video 3:28:23 - 3:56:21)
#
# Kullanim:
#   python3 gorev4_hellaswag.py benim  1000
#   python3 gorev4_hellaswag.py openai 1000
#   python3 gorev4_hellaswag.py openai 10042 fp32   # tam set, referans kontrolu
#
# Her ornegin sonucunu tek tek kaydediyorum; karsilastirma ve analiz
# gorev4_karsilastir.py'de.

import json
import math
import os
import sys
import time

import torch

import agirlik
import hellaswag as hs
from gpt2 import GPT, GPTConfig

ad = sys.argv[1] if len(sys.argv) > 1 else 'benim'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
BF16 = not (len(sys.argv) > 3 and sys.argv[3] == 'fp32')
torch.manual_seed(0)

if ad == 'benim':
    model = GPT(GPTConfig(vocab_size=50304, flash=True), weight_tying=True)
    model.load_state_dict(torch.load('veri/checkpoint/model_son.pt',
                                     map_location='cpu'))
    model.eval()
    aciklama = 'kendi modelim (FineWeb-Edu, Gorev 3)'
elif ad == 'turkce':
    model = GPT(GPTConfig(vocab_size=50304, flash=True), weight_tying=True)
    model.load_state_dict(torch.load('veri/checkpoint/model_turkce.pt',
                                     map_location='cpu'))
    model.eval()
    aciklama = 'Turkce egittigim model (Gorev 5)'
elif ad == 'openai':
    model = agirlik.openai_gpt2(flash=True)
    aciklama = "OpenAI'in GPT-2 (124M) agirliklari, kendi GPT sinifimda"
else:
    raise SystemExit('model: benim | turkce | openai')

veri = hs.ornekler(N)
print("=" * 78)
print(f"4) HELLASWAG ({hs.DIL}) - {aciklama}")
print("=" * 78)
print(f"  ornek sayisi: {len(veri)} / 10.042, hassasiyet: "
      f"{'bf16' if BF16 else 'fp32'}")
print()

# Uzun kosular icin: yarida kalirsa kaldigi yerden devam etsin
ARA = f"ara_hellaswag{'' if hs.DIL == 'en' else '_' + hs.DIL}_{ad}_{N}{'' if BF16 else '_fp32'}.json"
kayit = json.load(open(ARA)) if os.path.exists(ARA) else []
yapilan = {k['ind'] for k in kayit}
dogru = sum(k['p'] == k['etiket'] for k in kayit)
dogru_norm = sum(k['pn'] == k['etiket'] for k in kayit)
if kayit:
    print(f"  ara kayittan devam: {len(kayit)} ornek zaten yapilmis")
t0 = time.time()
for i, ornek in enumerate(veri, 1):
    if ornek['ind'] in yapilan:
        continue
    tok, msk, etiket = hs.hazirla(ornek)
    p, pn, top, ort = hs.tahmin(model, tok, msk, bf16=BF16)
    dogru += p == etiket
    dogru_norm += pn == etiket
    kayit.append(dict(ind=ornek['ind'], etiket=etiket, p=p, pn=pn,
                      tur=ornek['split_type'], top=top, ort=ort))
    if i % 500 == 0:
        json.dump(kayit, open(ARA, 'w'))
    if i % 100 == 0 or i == len(veri):
        print(f"  {i:>6}  acc {dogru/i:.4f}  acc_norm {dogru_norm/i:.4f}"
              f"   ({time.time()-t0:.0f} sn)", flush=True)

n = len(veri)
acc, accn = dogru / n, dogru_norm / n
# Rastgele tahminin bu orneklem buyuklugundeki standart hatasi
se = math.sqrt(0.25 * 0.75 / n)
print()
print(f"  acc      : {dogru}/{n} = {acc:.4f}")
print(f"  acc_norm : {dogru_norm}/{n} = {accn:.4f}")
print(f"  rastgele : 0.2500, bu n'de standart hata {se:.4f} "
      f"(%95 aralik {0.25-1.96*se:.3f} - {0.25+1.96*se:.3f})")
print(f"  sure     : {time.time()-t0:.0f} sn")

dil = '' if hs.DIL == 'en' else '_' + hs.DIL
dosya = f"sonuc_hellaswag{dil}_{ad}_{n}{'' if BF16 else '_fp32'}.json"
json.dump(dict(model=ad, dil=hs.DIL, aciklama=aciklama, n=n, bf16=BF16, acc=acc,
               acc_norm=accn, dogru=dogru, dogru_norm=dogru_norm,
               sure=time.time() - t0, kayit=kayit),
          open(dosya, 'w'), ensure_ascii=False)
print(f"  -> {dosya}")
