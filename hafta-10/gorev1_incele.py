# Gorev 1 - Hazir GPT-2'yi incele
# YZ50 Hafta 10  (video 0:03:39 - 0:13:47)
#
# Henuz kendi modelimi yazmadim. Once OpenAI'in egittigi agirliklara
# bakiyorum: hangi tensorler var, ne boyutta, toplam kac parametre.
# Amac mimariyi dosyadan OKUYARAK anlamak.

import json

import numpy as np
import torch

import agirlik

sd = agirlik.state_dict()
cfg = agirlik.config()
sonuc = {}

print("=" * 80)
print("1a) AGIRLIKLARIN LISTESI")
print("=" * 80)
if cfg:
    print(f"  config.json: n_layer={cfg['n_layer']}  n_head={cfg['n_head']}  "
          f"n_embd={cfg['n_embd']}  vocab={cfg['vocab_size']}  "
          f"block={cfg['n_positions']}")
    print()

# tekrar eden blok agirliklarini bir kez gosterip "x12" diye yaziyorum
gorulen = {}
sira = []
for k, v in sd.items():
    sade = k
    for i in range(12):
        sade = sade.replace(f'transformer.h.{i}.', 'transformer.h.N.')
    if sade not in gorulen:
        gorulen[sade] = [0, tuple(v.shape), v.numel()]
        sira.append(sade)
    gorulen[sade][0] += 1

print(f"  {'tensor':<40}{'sekil':>18}{'adet':>7}{'parametre':>14}")
print("  " + "-" * 79)
toplam = 0
for s in sira:
    adet, sekil, n = gorulen[s]
    toplam += adet * n
    say = f"{adet*n:,}".replace(',', '.')
    print(f"  {s:<40}{str(sekil):>18}{adet:>7}{say:>14}")
print("  " + "-" * 79)
print(f"  {'TOPLAM (dosyadaki tensorler)':<40}{'':>18}{'':>7}"
      f"{toplam:>14,}".replace(',', '.'))
print()

# gercek parametre sayisi: maske buffer'lari ve weight tying haric
gercek = sum(v.numel() for k, v in sd.items()
             if not k.endswith('.attn.bias') and not k.endswith('.attn.masked_bias'))
print(f"  Maske buffer'lari ({sum(1 for k in sd if k.endswith('.attn.bias'))} "
      f"adet, ogrenilen parametre degil) cikarilinca: {gercek:,}"
      .replace(',', '.'))
print()

# elle hesap
n_layer = cfg['n_layer'] if cfg else 12
n_embd = cfg['n_embd'] if cfg else 768
V = cfg['vocab_size'] if cfg else 50257
block = cfg['n_positions'] if cfg else 1024
emb = V * n_embd
pos = block * n_embd
blok = 12 * n_embd * n_embd + 13 * n_embd   # kabaca; asagida tam dokumu var
print("=" * 80)
print("1b) PARAMETRE SAYISI NEREDEN GELIYOR")
print("=" * 80)
kalem = [
    ('token embedding (wte)', f'{V} x {n_embd}', V * n_embd),
    ('pozisyon embedding (wpe)', f'{block} x {n_embd}', block * n_embd),
    ('attention c_attn (blok basina)', f'{n_embd} x {3*n_embd} + bias',
     n_embd * 3 * n_embd + 3 * n_embd),
    ('attention c_proj (blok basina)', f'{n_embd} x {n_embd} + bias',
     n_embd * n_embd + n_embd),
    ('mlp c_fc (blok basina)', f'{n_embd} x {4*n_embd} + bias',
     n_embd * 4 * n_embd + 4 * n_embd),
    ('mlp c_proj (blok basina)', f'{4*n_embd} x {n_embd} + bias',
     4 * n_embd * n_embd + n_embd),
    ('ln_1 + ln_2 (blok basina)', f'4 x {n_embd}', 4 * n_embd),
]
blok_toplam = sum(n for ad, _, n in kalem[2:])
print(f"  {'kalem':<34}{'hesap':>24}{'parametre':>16}")
print("  " + "-" * 76)
for ad, h, n in kalem[:2]:
    print(f"  {ad:<34}{h:>24}{n:>16,}".replace(',', '.'))
for ad, h, n in kalem[2:]:
    print(f"  {ad:<34}{h:>24}{n:>16,}".replace(',', '.'))
print("  " + "-" * 76)
print(f"  {'bir blok toplami':<34}{'':>24}{blok_toplam:>16,}".replace(',', '.'))
print(f"  {f'{n_layer} blok':<34}{'':>24}"
      f"{n_layer*blok_toplam:>16,}".replace(',', '.'))
print(f"  {'son ln_f':<34}{f'2 x {n_embd}':>24}{2*n_embd:>16,}".replace(',', '.'))
hesap = emb + pos + n_layer * blok_toplam + 2 * n_embd
print("  " + "-" * 76)
print(f"  {'GENEL TOPLAM':<34}{'':>24}{hesap:>16,}".replace(',', '.'))
print()
print(f"  Dosyadan okunan: {gercek:,}".replace(',', '.'))
print(f"  Elle hesaplanan: {hesap:,}".replace(',', '.'))
print(f"  Fark           : {gercek - hesap:,}".replace(',', '.'))
print()
print("""  124 MILYON NEREDEN CIKIYOR

  Dikkat cekici olan: lm_head (cikis katmani) listede YOK. Olmasi
  gerekirdi - modelin 768 boyuttan 50.257 skora cikmasi lazim. Sebep
  WEIGHT TYING: GPT-2 giris embedding'i (wte) ile cikis katmanini AYNI
  tensor olarak kullaniyor. Dosyada bir kere saklanmis.

  Eger ayri olsalardi fazladan {:,} parametre olurdu, yani model
  {:,} yerine {:,} olurdu. Toplam parametrenin yuzde {:.0f}'i
  tek basina embedding tablosunda - vocab 50.257 oldugu icin.
""".format(emb, hesap, hesap + emb, 100 * emb / hesap).replace(',', '.'))

sonuc['tensor_sayisi'] = len(sd)
sonuc['toplam_param'] = gercek
sonuc['hesap_param'] = hesap
sonuc['emb_param'] = emb
sonuc['tablo'] = [[s, list(gorulen[s][1]), gorulen[s][0], gorulen[s][0]*gorulen[s][2]]
                  for s in sira]

# =====================================================================
# 1c) POZISYON EMBEDDING'LERI
# =====================================================================
print("=" * 80)
print("1c) POZISYON EMBEDDING'LERI (wpe)")
print("=" * 80)
wpe = sd['transformer.wpe.weight'].float()
print(f"  sekil: {tuple(wpe.shape)}  "
      f"(1024 pozisyon, her biri 768 boyutlu bir vektor)")
print(f"  ortalama: {wpe.mean():.4f}   std: {wpe.std():.4f}")
print(f"  min: {wpe.min():.3f}   max: {wpe.max():.3f}")
print()

# ---- olcumler ----
import torch.nn.functional as F
norm = F.normalize(wpe, dim=1)
benzer = norm @ norm.T
uzunluk = wpe.norm(dim=1)

print("  VEKTOR UZUNLUKLARI (norm)")
print(f"  {'pozisyon':<12}{'norm':>8}")
print("  " + "-" * 22)
for i in [0, 1, 2, 5, 10, 100, 500, 1000, 1021, 1022, 1023]:
    isaret = '  <--' if (i == 0 or i == 1023) else ''
    print(f"  {i:<12}{uzunluk[i]:>8.2f}{isaret}")
print(f"  {'medyan':<12}{uzunluk.median():>8.2f}")
print()

print("  AYNI MESAFEDEKI BENZERLIK, BOLGEYE GORE")
print(f"  {'bolge':<14}{'i,i+1':>9}{'i,i+10':>9}{'i,i+50':>9}{'i,i+100':>10}")
print("  " + "-" * 51)
bolge_kayit = []
for ad, a, b in [('0-50', 0, 50), ('50-200', 50, 200), ('200-500', 200, 500),
                 ('500-800', 500, 800), ('800-920', 800, 920)]:
    r = []
    for d in (1, 10, 50, 100):
        v = [benzer[k, k + d].item() for k in range(a, b)]
        r.append(sum(v) / len(v))
    bolge_kayit.append([ad] + r)
    print(f"  {ad:<14}{r[0]:>9.3f}{r[1]:>9.3f}{r[2]:>9.3f}{r[3]:>10.3f}")
print()

print("  ILK POZISYONLARIN KOMSULARIYLA BENZERLIGI")
ilk_komsu = [benzer[k, k + 1].item() for k in range(12)]
print("  " + "  ".join(f"{k}\u2192{k+1}: {v:.3f}" for k, v in
                       enumerate(ilk_komsu[:8])))
print()

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    im = axes[0].imshow(wpe.numpy(), cmap='RdBu_r', aspect='auto',
                        vmin=-0.3, vmax=0.3)
    axes[0].set_xlabel('boyut (768)'); axes[0].set_ylabel('pozisyon (1024)')
    axes[0].set_title('wpe tablosunun tamami', color='#00285A', fontsize=10,
                      loc='left')
    plt.colorbar(im, ax=axes[0], shrink=.85)

    axes[1].plot(uzunluk.numpy(), color='#00285A', lw=1.1)
    axes[1].set_xlabel('pozisyon'); axes[1].set_ylabel('vektor uzunlugu')
    axes[1].set_title('norm: poz 0 aykiri, poz 1023 neredeyse sifir',
                      color='#00285A', fontsize=9.5, loc='left')
    axes[1].annotate(f'poz 0\n{uzunluk[0]:.1f}', (0, uzunluk[0]),
                     textcoords='offset points', xytext=(22, -4), fontsize=8,
                     color='#A3242B')
    axes[1].annotate(f'poz 1023\n{uzunluk[1023]:.2f}', (1023, uzunluk[1023]),
                     textcoords='offset points', xytext=(-54, 16), fontsize=8,
                     color='#A3242B',
                     arrowprops=dict(arrowstyle='->', color='#A3242B', lw=1))
    axes[1].grid(color='#EEF2F7', lw=1); axes[1].set_axisbelow(True)

    for d, c in [(1, '#00285A'), (10, '#0F6DB5'), (50, '#63697A'),
                 (100, '#A3242B')]:
        y = [benzer[k, k + d].item() for k in range(1023 - d)]
        axes[2].plot(y, lw=1.0, color=c, label=f'i, i+{d}')
    axes[2].set_xlabel('pozisyon'); axes[2].set_ylabel('kosinus benzerligi')
    axes[2].set_title('cozunurluk: basta ince, ortada kaba', color='#00285A',
                      fontsize=9.5, loc='left')
    axes[2].legend(fontsize=7.5, frameon=False, loc='lower right')
    axes[2].grid(color='#EEF2F7', lw=1); axes[2].set_axisbelow(True)
    plt.tight_layout(); plt.savefig('wpe.png', dpi=130); plt.close()
    print("  wpe.png kaydedildi")
except ImportError:
    print("  (matplotlib yok)")
print()

print(f"""  NE GORUYORUM - KENDI CUMLELERIMLE

  Hafta 7'de pozisyon tablosunu kendim eklemistim ve orada sadece
  "modelin siralamayi bilmesi lazim" diye dusunuyordum. Burada
  OpenAI'in gercekten EGITTIGI bir tabloya bakiyorum. Uc sey buldum ve
  ucu de beklentimden farkliydi.

  BIRINCI - POZISYON 0 BIR AYKIRI DEGER.
  Normu {uzunluk[0]:.2f}, medyan {uzunluk.median():.2f}. Yani uc kat
  uzun. Ve 0 ile 1 arasindaki benzerlik {benzer[0,1]:.3f} iken 7 ile 8
  arasindaki {benzer[7,8]:.3f}. Yani ilk birkac pozisyon birbirinden
  keskin bicimde ayrilmis, sonrakiler neredeyse ayni.

  Mantikli geliyor: "metnin ilk token'i" ozel bir bilgi. Model bunu
  guclu bir sinyal olarak kodlamis. 500. ile 501. token arasindaki
  fark ise pratikte onemsiz.

  IKINCI - COZUNURLUK BASTA YUKSEK, ORTADA DUSUK.
  Yukaridaki tabloda ayni MESAFEDEKI benzerlige bakiyorum. Ilk 50
  pozisyonda 100 adim otesi {bolge_kayit[0][4]:.3f} benzerlikte;
  ortada (500-800) ayni mesafe {bolge_kayit[3][4]:.3f}. Yani model
  basta "kacinci token" sorusunu ince ayrintisiyla ayirt ediyor,
  ilerledikce kabalasiyor.

  Bu da mantikli: bir cumlenin basinda 3. mi 5. kelime oldugun onemli;
  600. ile 650. token arasindaki fark cok daha az onemli.

  UCUNCU ve en carpici - SON POZISYON HIC EGITILMEMIS.
  Pozisyon 1022'nin normu {uzunluk[1022]:.2f}, pozisyon 1023'un normu
  {uzunluk[1023]:.2f}. Ucurum gibi bir dusus.

  Sebebini dusununce buldum: dil modelinde hedef, girdinin bir token
  kaydirilmis hali. 1024 token'lik bir ornekte en son pozisyonun
  (1023) tahmin edecegi sey 1025. token olurdu - ama o yok. Yani
  1023. pozisyon embedding'i hicbir zaman bir kayba katkida bulunmuyor
  ve gradyan almiyor. Baslangic degerinde kalmis, hatta weight decay
  onu sifira dogru cekmis.

  Bunu Gorev 4'te kendi test kodumda yaptigim hatayla birlikte
  dusununce ilginc oluyor: orada hedefi girdiyle ayni verip bir token
  kaydirmayi atlamistim. Burada ayni kaydirmanin, egitilmis bir
  modelde birakmis oldugu izi goruyorum.

  Ilk yazdigimda "tablonun SONU basindan daha gurultulu" diye
  tahmin etmistim - az egitilmis olmasini bekliyordum. Olcunce yanlis
  cikti: son bolgedeki komsu benzerligi {bolge_kayit[4][1]:.3f},
  ilk bolgede {bolge_kayit[0][1]:.3f}. Yani tam tersi. Tek
  egitilmemis satir 1023 ve o da gurultulu degil, neredeyse SIFIR.
""")

sonuc['wpe'] = dict(
    sekil=list(wpe.shape), ort=float(wpe.mean()), std=float(wpe.std()),
    minv=float(wpe.min()), maxv=float(wpe.max()),
    norm_0=float(uzunluk[0]), norm_medyan=float(uzunluk.median()),
    norm_1022=float(uzunluk[1022]), norm_1023=float(uzunluk[1023]),
    benzer_0_1=float(benzer[0, 1]), benzer_7_8=float(benzer[7, 8]),
    bolge=bolge_kayit, ilk_komsu=ilk_komsu)
json.dump(sonuc, open('sonuc_gorev1.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev1.json kaydedildi")
