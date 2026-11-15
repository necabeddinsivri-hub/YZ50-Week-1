# Gorev 6 (ek) - (b) Hafta 8'deki GPT'yi kendi tokenizer'imla Turkce'de egit
# YZ50 Hafta 9  (video 1:43:27 - 1:48:11)
#
# Hafta 8'de ayni modeli HARF seviyesinde egitmistim (V=87, val 1.5841).
# Simdi ayni mimariyi BPE token'lariyla egitiyorum (V=1024).
# Asil soru: iki val loss dogrudan karsilastirilabilir mi?

import json
import math
import time

import torch

from bpe import RegexTokenizer, goster
from model_h8 import GPT, TokenVeri, loss_olc, egit, param_sayisi

# Hafta 8 ile ayni config
CFG = dict(n_embd=128, n_head=4, n_blok=4, block=64, dropout=0.1)
BATCH, ADIM, LR = 32, 4000, 3e-4

# Hafta 8'in harf seviyesindeki sonuclari (raporundan)
H8_VAL, H8_TRAIN, H8_V, H8_PARAM = 1.5841, 1.6233, 87, 822_359

print("=" * 76)
print("6b) AYNI MODEL, BPE TOKEN'LARIYLA")
print("=" * 76)

metin = open('turkce.txt', encoding='utf-8').read()
d = json.load(open('tokenizer_regex.json'))
tok = RegexTokenizer()
tok.merges = {tuple(k): v for k, v in d['merges']}
tok._vocab_kur()
V = len(tok.vocab)

print("  metin token'lara cevriliyor...")
t0 = time.time()
ids = tok.encode(metin)
print(f"  {len(metin):,} karakter -> {len(ids):,} token  ({time.time()-t0:.0f} sn)"
      .replace(',', '.'))
print(f"  sikistirma: {len(metin.encode('utf-8'))/len(ids):.3f}x byte/token")
print()

veri = TokenVeri(ids, V, CFG['block'])
print(f"  vocab (V)   : {V}   (Hafta 8'de {H8_V})")
print(f"  train token : {len(veri.train_data):,}".replace(',', '.'))
print(f"  val token   : {len(veri.val_data):,}".replace(',', '.'))
print()
print("  NOT: Hafta 8'de korpusu SIRAYLA bolmustum ve val loss train'den")
print("  dusuk cikmisti - val dilimi daha homojendi, kendi hatamdi.")
print("  Bu sefer dilimleri karistirarak boluyorum.")
print()

torch.manual_seed(1337)
model = GPT(V, **CFG)
p = param_sayisi(model)
print(f"  parametre: {p:,}   (Hafta 8'de {H8_PARAM:,})".replace(',', '.'))
print(f"  fark embedding ve cikis katmanindan: V {H8_V} -> {V}")
print()

print("-" * 76)
print(f"  EGITIM  ({ADIM} adim, lr={LR})")
print("-" * 76)
sure = egit(model, veri, adim=ADIM, lr=LR, batch=BATCH)
s = loss_olc(model, veri, batch=BATCH)
print(f"  SONUC: train {s['train']:.4f}  val {s['val']:.4f}  "
      f"({sure/60:.1f} dk)")
print()

gg = torch.Generator().manual_seed(42)
cikti = tok.decode(model.generate(torch.zeros((1, 1), dtype=torch.long),
                                  400, g=gg)[0].tolist())
# uretim 0 numarali token'dan basliyor; o token NUL byte'i. Metne zarar
# vermiyor ama Word/PowerPoint gibi XML tabanli dosyalara konulamiyor,
# o yuzden yazdirilamayan kontrol karakterlerini temizliyorum.
cikti = ''.join(c for c in cikti if c == '\n' or c.isprintable())
print("  --- modelin urettigi metin (400 token) ---")
print(cikti)
print()

# ---------------------------------------------------------------------
# KARSILASTIRILABILIR MI?
# ---------------------------------------------------------------------
print("=" * 76)
print("VAL LOSS'U HAFTA 8'INKIYLE KARSILASTIRABILIR MIYIM?")
print("=" * 76)
print(f"  {'':<28}{'Hafta 8 (harf)':>18}{'Hafta 9 (BPE)':>18}")
print("  " + "-" * 66)
print(f"  {'vocab (V)':<28}{H8_V:>18}{V:>18}")
print(f"  {'parametre':<28}{H8_PARAM:>18,}{p:>18,}".replace(',', '.'))
print(f"  {'train loss':<28}{H8_TRAIN:>18.4f}{s['train']:>18.4f}")
print(f"  {'val loss':<28}{H8_VAL:>18.4f}{s['val']:>18.4f}")
print(f"  {'log(V) = egitimsiz taban':<28}{math.log(H8_V):>18.4f}"
      f"{math.log(V):>18.4f}")
print("  " + "-" * 66)
print()

# ayni olcege cevir: karakter basina bit
tr_karakter = len(metin)
tr_token = len(ids)
kar_token = tr_token / tr_karakter          # karakter basina token
h8_bpc = H8_VAL / math.log(2)               # harf modelinde 1 token = 1 karakter
h9_bpc = s['val'] / math.log(2) * kar_token

print("  ORTAK OLCEK: KARAKTER BASINA BIT (bpc)")
print(f"  {'':<34}{'Hafta 8':>14}{'Hafta 9':>14}")
print("  " + "-" * 62)
print(f"  {'val loss (nat / token)':<34}{H8_VAL:>14.4f}{s['val']:>14.4f}")
print(f"  {'karakter basina token':<34}{1.0:>14.4f}{kar_token:>14.4f}")
print(f"  {'KARAKTER BASINA BIT':<34}{h8_bpc:>14.4f}{h9_bpc:>14.4f}")
print("  " + "-" * 62)
print(f"  fark: {h8_bpc - h9_bpc:+.4f} bit/karakter "
      f"({'BPE daha iyi' if h9_bpc < h8_bpc else 'harf daha iyi'})")
print()

print(f"""  CEVAP: HAYIR, HAM LOSS'LAR DOGRUDAN KARSILASTIRILAMAZ.

  Uc sebep var ve ucu de Hafta 8'de Turkce-Shakespeare karsilastirmasinda
  yazdiklarimin devami.

  1) OLCU BIRIMI FARKLI. cross_entropy "bir sonraki TOKEN'i tahmin etme"
     zorlugunu olcuyor. Hafta 8'de token = karakterdi; simdi token =
     ortalama {1/kar_token:.2f} karakter. Yani Hafta 9'daki her tahmin daha
     BUYUK bir parcayi kapsiyor. Ayni sayi ayni isi olcmuyor.

  2) SECENEK SAYISI FARKLI. Egitimsiz taban log(V): Hafta 8'de
     {math.log(H8_V):.4f}, Hafta 9'da {math.log(V):.4f}. BPE modeli daha
     baslamadan {math.log(V)-math.log(H8_V):+.4f} handikapli; {V} secenek
     arasindan secmek {H8_V} arasindan secmekten zor.

  3) VERI BOLMESI FARKLI. Hafta 8'de korpusu sirayla bolmustum ve
     val train'den kolay cikmisti. Bu hafta karistirarak boldum.
     Olcum kosulu degisti.

  ADIL OLCUT: ikisini de ayni birime cevirmek. Karakter basina bit
  en dogal secim, cunku ikisi de ayni metni modelliyor ve karakter
  sayisi ikisinde de ayni.

      bpc = (val_loss / ln 2) x (token sayisi / karakter sayisi)

  Bu olcege gore Hafta 8: {h8_bpc:.4f} bit/karakter, Hafta 9: {h9_bpc:.4f}.
  {'BPE modeli daha iyi.' if h9_bpc < h8_bpc else 'Harf modeli daha iyi.'}

  {'BPE modelinin avantaji sadece loss degil: ayni block_size=64 ile artik ' if h9_bpc < h8_bpc else ''}
  {f'{1/kar_token:.2f} kat daha fazla metin goruyor, yani ayni hesapla daha genis baglam.' if h9_bpc < h8_bpc else ''}

  Bu, Hafta 6, 7 ve 8'de ucer kez karsima cikan dersin dorduncusu:
  bir sayiyi sadece olcum kosullari ayniysa karsilastirabilirsin.
  Degilse once ortak bir birime cevirmen gerekiyor.
""")

torch.save(model.state_dict(), 'gpt_bpe_turkce.pt')
json.dump({
    'V': V, 'param': p, 'train': s['train'], 'val': s['val'], 'sure': sure,
    'token': len(ids), 'karakter': len(metin),
    'karakter_basina_token': kar_token,
    'h8_val': H8_VAL, 'h8_train': H8_TRAIN, 'h8_V': H8_V, 'h8_param': H8_PARAM,
    'h8_bpc': h8_bpc, 'h9_bpc': h9_bpc,
    'ornek': cikti, 'config': dict(CFG, batch=BATCH, adim=ADIM, lr=LR),
}, open('sonuc_gorev6b.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev6b.json kaydedildi")
