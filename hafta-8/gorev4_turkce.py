# Gorev 4 - Ayni modeli kendi derledigim Turkce metinle egit
# YZ50 Hafta 8
#
# Veri: turkce.txt - turkce_derle.py ile kendim derledim (~1 MB).
# Shakespeare dosyasiyla AYNI boyutta olsun diye hedefi 1.050.000 bayt
# tuttum; Karpathy'nin input.txt'si 1.115.394 bayt.
#
# Model: Gorev 3'te sectigim config'in BIRE BIR aynisi. Tek degisen
# alfabe boyutu (V), cunku o veriden geliyor.

import json
import torch
from ortak import Veri, loss_olc, egit, param_sayisi, veri as sh_veri
from katmanlar import GPTDilModeli, uret
from gorev3_buyut import CONFIG, BATCH, ADIM, LR

sonuc = json.load(open('sonuclar.json'))

tr = Veri('turkce.txt')

print("=" * 74)
print("GOREV 4 - TURKCE")
print("=" * 74)
print()
print("-" * 74)
print("VERI KARSILASTIRMASI")
print("-" * 74)
print(f"  {'':<22}{'Shakespeare':>16}{'Turkce':>16}")
print(f"  {'dosya (bayt)':<22}"
      f"{len(sh_veri.metin.encode('utf-8')):>16,}"
      f"{len(tr.metin.encode('utf-8')):>16,}".replace(',', '.'))
print(f"  {'karakter sayisi':<22}{len(sh_veri.metin):>16,}"
      f"{len(tr.metin):>16,}".replace(',', '.'))
print(f"  {'FARKLI karakter (V)':<22}{sh_veri.V:>16}{tr.V:>16}")
print(f"  {'train token':<22}{len(sh_veri.train_data):>16,}"
      f"{len(tr.train_data):>16,}".replace(',', '.'))
print(f"  {'val token':<22}{len(sh_veri.val_data):>16,}"
      f"{len(tr.val_data):>16,}".replace(',', '.'))
print()
print(f"  Shakespeare alfabesi ({sh_veri.V}):")
print(f"    {''.join(sh_veri.harfler)!r}")
print(f"  Turkce alfabesi ({tr.V}):")
print(f"    {''.join(tr.harfler)!r}")
print()

sadece_tr = sorted(set(tr.harfler) - set(sh_veri.harfler))
sadece_sh = sorted(set(sh_veri.harfler) - set(tr.harfler))
print(f"  Sadece Turkce'de olanlar ({len(sadece_tr)}): {''.join(sadece_tr)!r}")
print(f"  Sadece Shakespeare'de olanlar ({len(sadece_sh)}): "
      f"{''.join(sadece_sh)!r}")
print()
print("  Turkce'de 6 ozel harf var: c-g-i-o-s-u uzerine isaretli haller")
print("  (c,g,i,o,s,u -> ç,ğ,ı,ö,ş,ü) hem kucuk hem buyuk = 12 fazla")
print("  karakter. Ustune rakamlar ve bazi noktalama farklari biniyor.")
print()

# ---------------------------------------------------------------------
# EGITIM
# ---------------------------------------------------------------------
print("-" * 74)
print("EGITIM  (Gorev 3 ile ayni config)")
print("-" * 74)
for k, v in CONFIG.items():
    print(f"  {k:<10}: {v}")
print(f"  {'batch':<10}: {BATCH}")
print(f"  {'adim':<10}: {ADIM}   lr: {LR}")
print(f"  {'V':<10}: {tr.V}  (Shakespeare'de {sh_veri.V})")
print()

torch.manual_seed(1337)
model = GPTDilModeli(v=tr.V, **CONFIG)
p = param_sayisi(model)
print(f"  parametre: {p:,}".replace(',', '.'))
print(f"  (Shakespeare modeli {sonuc['8_olcekli_dropout']['param']:,}; "
      f"fark embedding ve lm_head'ten geliyor)".replace(',', '.'))
print()

sure = egit(model, adim=ADIM, lr=LR, block=CONFIG['block'], batch=BATCH,
            d=tr, her=500)
s = loss_olc(model, d=tr, block=CONFIG['block'], batch=BATCH)
print(f"  SONUC: train {s['train']:.4f}  val {s['val']:.4f}  "
      f"({sure / 60:.1f} dk)")
print()

gg = torch.Generator().manual_seed(42)
ornek = tr.decode(uret(model, torch.zeros((1, 1), dtype=torch.long), 1000,
                       g=gg)[0].tolist())
print("  --- 1000 karakter Turkce ornek ---")
print(ornek)
print()
torch.save(model.state_dict(), 'gpt_turkce.pt')

sonuc['9_turkce'] = dict(ad='olcekli (Turkce)', param=p, train=s['train'],
                         val=s['val'], sure=sure, V=tr.V,
                         karakter=len(tr.metin))
sonuc['_ornek_turkce'] = ornek
sonuc['_tr_alfabe'] = ''.join(tr.harfler)

# ---------------------------------------------------------------------
# IKI VAL LOSS DOGRUDAN KARSILASTIRILABILIR MI?
# ---------------------------------------------------------------------
import math
sh = sonuc['8_olcekli_dropout']
print("=" * 74)
print("IKI VAL LOSS DOGRUDAN KARSILASTIRILABILIR MI?")
print("=" * 74)
print(f"  Shakespeare val loss : {sh['val']:.4f}   (V = {sh_veri.V})")
print(f"  Turkce      val loss : {s['val']:.4f}   (V = {tr.V})")
print()
print("  CEVAP: HAYIR, dogrudan karsilastirilamaz.")
print()
print("  Sebep: cross_entropy, 'dogru karakteri bulma' isinin zorluguna")
print("  gore olculuyor ve bu zorluk alfabe boyutuna bagli. Hicbir sey")
print("  ogrenmemis bir model, her karakteri esit olasi sayar ve")
print("  loss = log(V) alir:")
print()
print(f"    Shakespeare: log({sh_veri.V}) = {math.log(sh_veri.V):.4f}")
print(f"    Turkce     : log({tr.V}) = {math.log(tr.V):.4f}")
print(f"    fark       : {math.log(tr.V) - math.log(sh_veri.V):+.4f}")
print()
print("  Yani Turkce model daha baslamadan {:+.4f} handikapli."
      .format(math.log(tr.V) - math.log(sh_veri.V)))
print("  87 secenek arasindan dogruyu bulmak, 65 secenek arasindan")
print("  bulmaktan tanimi geregi daha zor.")
print()
print("  Adil olcut: loss'u baslangictan NE KADAR dustugune bakmak.")
print()
print(f"  {'':<14}{'log(V)':>9}{'val':>9}{'kazanc':>9}{'oran':>9}")
print("  " + "-" * 50)
for ad, V_, val in [('Shakespeare', sh_veri.V, sh['val']),
                    ('Turkce', tr.V, s['val'])]:
    t = math.log(V_)
    print(f"  {ad:<14}{t:>9.4f}{val:>9.4f}{t - val:>9.4f}{val / t:>9.3f}")
print("  " + "-" * 50)
print("  (kazanc = log(V) - val; oran = val / log(V), kucuk olan iyi)")
print()
print("""  IKINCI SEBEP: diller ayni degil.

  Turkce sondan eklemeli. 'gel' kokunden 'gelemeyeceklerini' gibi uzun
  kelimeler cikiyor. Karakter seviyesinde bakarsan Turkce'de bir
  kelimenin sonunu tahmin etmek daha kolay (ek kaliplari duzenli),
  ama kelime sinirini ve hangi ekin gelecegini tahmin etmek daha zor.
  Ingilizce'de kelimeler kisa ve bagimsiz.

  UCUNCU SEBEP: metinler ayni turden degil. Shakespeare tek bir yazarin
  tiyatro metni; ayni kalip surekli tekrar ediyor (KONUSAN ADI, iki
  nokta, satir basi). Benim Turkce korpusum roman, deneme, haber
  karisimi: cok daha cesitli, yani tahmin etmesi daha zor. Bu da
  Turkce modelin aleyhine calisiyor.

  SONUC: iki sayiyi yan yana koyup 'Turkce modeli daha kotu' demek
  yanlis olur. Ayni veri uzerinde yapilan karsilastirmalar (Gorev 1-3'teki
  ablasyon tablosu) anlamli; farkli veri uzerindeki loss'lar degil.
  Bu, Hafta 6'da ogrendigim seyin baska bir hali: bir sayiyi sadece
  OLCUM KOSULLARI ayni oldugunda karsilastirabilirsin.""")
print()

json.dump(sonuc, open('sonuclar.json', 'w'), indent=1, ensure_ascii=False)
print("  -> sonuclar.json guncellendi")
