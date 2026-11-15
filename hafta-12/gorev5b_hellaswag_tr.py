# Gorev 5 (ek), ikinci yol - Turkce HellaSwag
# YZ50 Hafta 12
#
# malhajar/hellaswag-tr: HellaSwag dogrulama setinin Turkce cevirisi
# (10.041 ornek; Ingilizce setteki ilk ornek, ind 24, bu sette yok).
# Ayni kodla, yalniz veri dosyasini degistirerek kosuyorum:
#   HS_DIL=tr python3 gorev4_hellaswag.py turkce 1000
#   HS_DIL=tr python3 gorev4_hellaswag.py openai 1000
#   HS_DIL=tr python3 gorev4_hellaswag.py benim  1000
# Bu dosya uc sonucu yan yana koyuyor.

import json
import math
import os

sonuc = {}
print("=" * 78)
print("5) TURKCE HELLASWAG - ilk 1.000 ornek")
print("=" * 78)
for ad, etiket in [('turkce', 'Turkce modelim'), ('openai', 'OpenAI GPT-2'),
                   ('benim', 'FineWeb-Edu modelim')]:
    f = f'sonuc_hellaswag_tr_{ad}_1000.json'
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    se = math.sqrt(0.25 * 0.75 / r['n'])
    z = (r['acc_norm'] - 0.25) / se
    sonuc[ad] = dict(etiket=etiket, acc=r['acc'], acc_norm=r['acc_norm'],
                     n=r['n'], z=z, ci=[0.25 - 1.96 * se, 0.25 + 1.96 * se])
    print(f"  {etiket:<22} acc {r['acc']:.4f}  acc_norm {r['acc_norm']:.4f}  "
          f"z {z:+.2f}")
print("  rastgele: 0.2500; %95 aralik "
      f"{0.25-1.96*math.sqrt(0.1875/1000):.3f} - {0.25+1.96*math.sqrt(0.1875/1000):.3f}")
# ---------------------------------------------------------------------
# MODELSIZ TABAN CIZGISI: baglamla en cok kelime paylasan devami sec.
# OpenAI'in modeli neredeyse hic Turkce gormedigi halde Turkce sette
# rastgelenin ustune cikti. Bunun bir "anlama" mi yoksa yuzeysel bir
# ipucu mu oldugunu ayirt etmek icin hic model kullanmayan bir secici
# kuruyorum. Esitlikte puan esit bolunuyor (2 devam esitse 1/2).
# ---------------------------------------------------------------------
import re
import pyarrow.parquet as pq


def kelimeler(t):
    return {w for w in re.findall(r"\w+", t.lower()) if len(w) >= 3}


def ortusme_tabani(yol, n=1000):
    satir = pq.read_table(yol).to_pylist()
    satir = satir if n is None else satir[:n]
    puan = 0.0
    secim = {}
    for o in satir:
        b = kelimeler(o['ctx'])
        p = [len(b & kelimeler(e)) for e in o['endings']]
        en = [i for i, x in enumerate(p) if x == max(p)]
        if int(o['label']) in en:
            puan += 1 / len(en)
        secim[o['ind']] = en
    return puan / len(satir), secim


taban = {}
for dil, yol in [('tr', 'veri/ham/hellaswag_tr_validation.parquet'),
                 ('en', 'veri/ham/hellaswag_validation.parquet')]:
    acc, secim = ortusme_tabani(yol)
    taban[dil] = acc
    print(f"  kelime ortusmesi tabani ({dil}, ilk 1000): {acc:.4f}")
sonuc['ortusme_tabani'] = taban

# Hipotez: Turkce'de ekler yuzunden TAM kelimeler eslesmiyor ("adam" -
# "adami"), ama GPT-2 tokenizer'inin alt-kelime PARCALARI eslesiyor.
# Ayni seciciyi kelime yerine GPT-2 token'lariyla kuruyorum (bosluklu
# token'lari ve 2 karakterden kisa parcalari atiyorum).
from tokenizer import enc
_e = enc()


def parcalar(t):
    return {i for i in _e.encode(t) if len(_e.decode([i]).strip()) >= 3}


def parca_tabani(yol, n=1000):
    satir = pq.read_table(yol).to_pylist()[:n]
    puan = 0.0
    for o in satir:
        b = parcalar(o['ctx'])
        p = [len(b & parcalar(' ' + e)) for e in o['endings']]
        en = [i for i, x in enumerate(p) if x == max(p)]
        if int(o['label']) in en:
            puan += 1 / len(en)
    return puan / len(satir)


parca = {}
for dil, yol in [('tr', 'veri/ham/hellaswag_tr_validation.parquet'),
                 ('en', 'veri/ham/hellaswag_validation.parquet')]:
    parca[dil] = parca_tabani(yol)
    print(f"  token parcasi ortusmesi tabani ({dil}, ilk 1000): {parca[dil]:.4f}")
sonuc['parca_tabani'] = parca
tam_en, _ = ortusme_tabani('veri/ham/hellaswag_validation.parquet', n=None)
print(f"  kelime ortusmesi tabani (en, TAM SET 10.042): {tam_en:.4f}")
sonuc['ortusme_tabani_en_tam'] = tam_en
# Ayni dogruluk ayni davranis mi? Tam sette OpenAI'in secimi ile
# ortusme secicisinin (tek basina en iyi buldugu) secimi ne siklikla ayni?
_, secim_en = ortusme_tabani('veri/ham/hellaswag_validation.parquet', n=None)
if os.path.exists('sonuc_hellaswag_openai_10042.json'):
    kk = json.load(open('sonuc_hellaswag_openai_10042.json'))['kayit']
    tek = [r for r in kk if len(secim_en[r['ind']]) == 1]
    ayni = sum(r['pn'] == secim_en[r['ind']][0] for r in tek) / len(tek)
    iki_dogru = sum(r['pn'] == r['etiket'] and secim_en[r['ind']][0] == r['etiket'] for r in tek) / len(tek)
    print(f"  TAM SET (en): OpenAI secimi ortusme secimiyle ayni %{100*ayni:.1f} "
          f"({len(tek)} ornek); ikisi birden dogru %{100*iki_dogru:.1f}")
    sonuc['openai_ortusme_uyumu_en_tam'] = dict(oran=ayni, n=len(tek), ikisi_dogru=iki_dogru)

# OpenAI'in dogru bildikleri, ortusme secicisinin "tek basina en iyi"
# buldugu orneklerde mi yogunlasiyor?
if os.path.exists('sonuc_hellaswag_tr_openai_1000.json'):
    _, secim = ortusme_tabani('veri/ham/hellaswag_tr_validation.parquet')
    k = json.load(open('sonuc_hellaswag_tr_openai_1000.json'))['kayit']
    tek = [r for r in k if len(secim[r['ind']]) == 1]
    ayni = sum(r['pn'] == secim[r['ind']][0] for r in tek) / len(tek)
    print(f"  OpenAI'in secimi, ortusme secicisinin tek basina sectigiyle ayni: "
          f"%{100*ayni:.1f} ({len(tek)} ornek; rastgele beklenti %25)")
    sonuc['openai_ortusme_uyumu'] = dict(oran=ayni, n=len(tek))

json.dump(sonuc, open('sonuc_gorev5b.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev5b.json")
