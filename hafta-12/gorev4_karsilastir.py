# Gorev 4 - HellaSwag sonuclarini karsilastir ve farkin nereden geldigine bak
# YZ50 Hafta 12
#
# gorev4_hellaswag.py iki model icin ornek ornek sonuc yazdi. Burada:
#   - iki modelin dogrulugu ve rastgeleden (%25) ayrilip ayrilmadigi
#   - hangi orneklerde ikisi ayrisiyor
#   - benim modelim "neye gore" seciyor: anlama mi, uzunluga mi?
#   - (varsa) tam setteki OpenAI sonucu Karpathy'nin sayisini tutuyor mu

import json
import math
import os
from collections import Counter

import hellaswag as hs
from tokenizer import enc

e = enc()
J = lambda f: json.load(open(f)) if os.path.exists(f) else None
ben = J('sonuc_hellaswag_benim_1000.json')
oai = J('sonuc_hellaswag_openai_1000.json')
tam = J('sonuc_hellaswag_openai_10042_fp32.json') or J('sonuc_hellaswag_openai_10042.json')
assert ben and oai, 'once gorev4_hellaswag.py iki model icin calismali'
n = ben['n']
veri = hs.ornekler(n)
sonuc = dict(n=n)

print("=" * 78)
print(f"4) HELLASWAG KARSILASTIRMASI - ilk {n} ornek")
print("=" * 78)
se = math.sqrt(0.25 * 0.75 / n)
print(f"  {'model':<26}{'acc':>8}{'acc_norm':>10}{'z (vs %25)':>12}")
print("  " + "-" * 56)
for ad, r in [('rastgele tahmin', None), ('benim modelim', ben),
              ("OpenAI GPT-2 (124M)", oai)]:
    if r is None:
        print(f"  {ad:<26}{0.25:>8.4f}{0.25:>10.4f}{'':>12}")
        continue
    z = (r['acc_norm'] - 0.25) / se
    print(f"  {ad:<26}{r['acc']:>8.4f}{r['acc_norm']:>10.4f}{z:>12.2f}")
    sonuc[r['model']] = dict(acc=r['acc'], acc_norm=r['acc_norm'], z=z)
print("  " + "-" * 56)
print(f"  n={n} icin rastgele tahminin standart hatasi {se:.4f}; %95 aralik "
      f"{0.25-1.96*se:.3f} - {0.25+1.96*se:.3f}")
sonuc['se'] = se
print()

# --------------------------- ayrisma tablosu ---------------------------
kb = {k['ind']: k for k in ben['kayit']}
ko = {k['ind']: k for k in oai['kayit']}
ikisi = sadece_ben = sadece_oai = hicbiri = 0
for ind in kb:
    b = kb[ind]['pn'] == kb[ind]['etiket']
    o = ko[ind]['pn'] == ko[ind]['etiket']
    ikisi += b and o
    sadece_ben += b and not o
    sadece_oai += o and not b
    hicbiri += not b and not o
print("  ORNEK ORNEK (acc_norm):")
print(f"    ikisi de dogru      : {ikisi}")
print(f"    yalniz benimki      : {sadece_ben}")
print(f"    yalniz OpenAI       : {sadece_oai}")
print(f"    ikisi de yanlis     : {hicbiri}")
# Iki model birbirinden bagimsiz ve ikisi de rastgele olsaydi,
# "ikisi de dogru" beklenen sayisi n * p_ben * p_oai olurdu.
beklenen = n * ben['acc_norm'] * oai['acc_norm']
print(f"    (bagimsiz olsalardi 'ikisi de dogru' beklenen: {beklenen:.0f})")
# Bu fark sans eseri mi? 2x2 tablonun phi katsayisi ve bagimsizlik
# varsayimi altinda "ikisi de dogru" sayisinin z degeri.
a_, b_, c_, d_ = ikisi, sadece_ben, sadece_oai, hicbiri
phi = (a_ * d_ - b_ * c_) / math.sqrt((a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_))
p0 = ben['acc_norm'] * oai['acc_norm']
z_ort = (ikisi - n * p0) / math.sqrt(n * p0 * (1 - p0))
print(f"    phi katsayisi {phi:.3f}, ortusme z = {z_ort:.2f}")
sonuc['ayrisma'] = dict(ikisi=ikisi, sadece_ben=sadece_ben,
                        sadece_oai=sadece_oai, hicbiri=hicbiri,
                        beklenen_ikisi=beklenen, phi=phi, z=z_ort)

# Ayirt etme gucu: dort devamin token basina loss'lari ne kadar ayrisiyor?
# Hicbir sey bilmeyen bir model dordune de ayni loss'u verir.
def yayilim(k):
    return sum(max(r['ort']) - min(r['ort']) for r in k['kayit']) / len(k['kayit'])
yb, yo = yayilim(ben), yayilim(oai)
print(f"  AYIRT ETME GUCU (dort devamin ortalama loss'u, en buyuk - en kucuk):")
print(f"    benim {yb:.3f} nat, OpenAI {yo:.3f} nat  ({yo/yb:.1f} kat)")
sonuc['yayilim'] = dict(benim=yb, openai=yo)
print()

# --------------------------- alt gruplar ---------------------------
print("  ALT GRUPLAR (acc_norm):")
gr = {}
for tur in ('indomain', 'zeroshot'):
    idx = [k['ind'] for k in ben['kayit'] if k['tur'] == tur]
    if not idx:
        continue
    a = sum(kb[i]['pn'] == kb[i]['etiket'] for i in idx) / len(idx)
    b = sum(ko[i]['pn'] == ko[i]['etiket'] for i in idx) / len(idx)
    gr[tur] = dict(n=len(idx), benim=a, openai=b)
    print(f"    {tur:<10} n={len(idx):<5} benim {a:.4f}   OpenAI {b:.4f}")
sonuc['gruplar'] = gr
print()

# --------------- benim modelim neye gore seciyor? ---------------
# Bir model hicbir sey anlamiyorsa bile rastgele secmez: genel, sik
# gecen kelimelerden olusan devami "daha olasi" bulur. Bunu olcmek icin
# secilen devamin kisaligina ve "sik kelime" orani yerine en basit
# vekile bakiyorum: devamin token sayisi.
uzunluk_en_kisa = {'benim': 0, 'openai': 0, 'etiket': 0}
uzunluk_en_uzun = {'benim': 0, 'openai': 0, 'etiket': 0}
for ornek in veri:
    L = [len(e.encode(' ' + s)) for s in ornek['endings']]
    kisa, uzun = L.index(min(L)), L.index(max(L))
    for ad, sec in [('benim', kb[ornek['ind']]['pn']),
                    ('openai', ko[ornek['ind']]['pn']),
                    ('etiket', int(ornek['label'])),
                    ('benim_acc', kb[ornek['ind']]['p']),
                    ('openai_acc', ko[ornek['ind']]['p'])]:
        uzunluk_en_kisa[ad] = uzunluk_en_kisa.get(ad, 0) + (sec == kisa)
        uzunluk_en_uzun[ad] = uzunluk_en_uzun.get(ad, 0) + (sec == uzun)
print("  SECILEN DEVAMIN UZUNLUGU (acc_norm secimi):")
print(f"    {'':<14}{'en kisayi sectigi':>20}{'en uzunu sectigi':>20}")
for ad in ('benim', 'openai', 'etiket'):
    ad2 = {'benim': 'benim modelim', 'openai': 'OpenAI', 'etiket': 'dogru cevap'}[ad]
    print(f"    {ad2:<14}{uzunluk_en_kisa[ad]/n:>20.3f}{uzunluk_en_uzun[ad]/n:>20.3f}")
print(f"    acc (TOPLAM loss) ile: benim en kisayi {uzunluk_en_kisa['benim_acc']/n:.3f}, "
      f"OpenAI {uzunluk_en_kisa['openai_acc']/n:.3f}")
print(f"    (bilgisiz bir secici icin beklenen: ~0.25 / ~0.25)")
dag = Counter(kb[i]['pn'] for i in kb)
print(f"    benim modelimin secim dagilimi (0,1,2,3): "
      f"{[dag[i] for i in range(4)]}")
sonuc['uzunluk'] = dict(kisa=uzunluk_en_kisa, uzun=uzunluk_en_uzun,
                        dagilim=[dag[i] for i in range(4)])
print()

# --------------------------- bir ornek ---------------------------
for ornek in veri:
    i = ornek['ind']
    if ko[i]['pn'] == ko[i]['etiket'] and kb[i]['pn'] != kb[i]['etiket'] \
            and len(ornek['ctx']) < 140:
        print("  BIR ORNEK: OpenAI dogru, benimki yanlis")
        print(f"    baglam: {ornek['ctx']!r}")
        for j, s in enumerate(ornek['endings']):
            isaret = ' <- dogru' if j == int(ornek['label']) else ''
            print(f"    [{j}] benim {kb[i]['ort'][j]:.3f}  OpenAI "
                  f"{ko[i]['ort'][j]:.3f}  {s[:70]!r}{isaret}")
        print(f"    (sayi: devam token'i basina ortalama loss; dusuk olan secilir)")
        sonuc['ornek'] = dict(ctx=ornek['ctx'], endings=ornek['endings'],
                              label=int(ornek['label']),
                              benim=kb[i]['ort'], openai=ko[i]['ort'],
                              benim_secim=kb[i]['pn'], openai_secim=ko[i]['pn'])
        break
print()

# --------------------- referans kontrolu ---------------------
if tam:
    hass = 'bf16' if tam.get('bf16') else 'fp32'
    print(f"  REFERANS KONTROLU: OpenAI GPT-2, tam set (10.042), {hass}")
    print(f"    benim kodum   : acc {tam['acc']:.4f}  acc_norm {tam['acc_norm']:.4f}")
    print(f"    Karpathy      : acc 0.2859  acc_norm 0.2955")
    sonuc['referans'] = dict(acc=tam['acc'], acc_norm=tam['acc_norm'],
                             karpathy_acc=0.2859, karpathy_norm=0.2955,
                             hassasiyet=hass, n=tam['n'])

# Ilk 1.000 ornek neden daha kolay? Kaynaga (ActivityNet / WikiHow) bak.
if tam:
    tum = hs.ornekler()
    kaynak = {o['ind']: o['source_id'].split('~')[0] for o in tum}
    ilk = [kaynak[o['ind']] for o in tum[:n]]
    k_sonuc = {}
    for kay in ('activitynet', 'wikihow'):
        kk = [r for r in tam['kayit'] if kaynak[r['ind']] == kay]
        k_sonuc[kay] = dict(ilk_pay=ilk.count(kay) / n,
                            tum_pay=len(kk) / len(tum),
                            openai=sum(r['pn'] == r['etiket'] for r in kk) / len(kk))
        print(f"  {kay:<12} ilk {n}'de %{100*k_sonuc[kay]['ilk_pay']:.0f}, tum sette "
              f"%{100*k_sonuc[kay]['tum_pay']:.0f}; OpenAI acc_norm {k_sonuc[kay]['openai']:.4f}")
    sonuc['kaynak'] = k_sonuc

json.dump(sonuc, open('sonuc_gorev4.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev4.json")
