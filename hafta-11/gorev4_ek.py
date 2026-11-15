# Gorev 4 (ek) - (b) Ayni modeli iki tokenizer'la Turkce metinde egit
# YZ50 Hafta 11
#
# Mail iki ek gorevden birini istiyor. (a) DDP, birden cok GPU
# gerektiriyor; bende tek GPU bile yok, o yuzden (b)'yi yapiyorum.
#
# SORU: ayni modeli ayni Turkce metinde iki kez egitiyorum. Bir kez
# GPT-2'nin tokenizer'iyla (V=50257), bir kez Hafta 9'da kendi
# yazdigim tokenizer'la (V=1024). Ayni SURE icinde hangisi metni daha
# cok goruyor, ve loss'lari nasil karsilastiririm?
#
# Ikinci soru asil soru. Hafta 8 ve 9'da ogrendigim sey: farkli
# vocab'larla egitilmis modellerin loss'lari DOGRUDAN
# karsilastirilamaz, cunku birinin tahmin ettigi sey 50257 secenekli
# bir piyango, otekinin 1024 secenekli. Ortak olcek: karakter basina
# bit (bits per character). Hafta 9'da da bunu kullanmistim.

import gc
import json
import math
import os
import time

import torch

from bicim import bin as bn
from bpe import RegexTokenizer
from gpt2 import GPT, GPTConfig
from tokenizer import enc
from veri import VeriYukleyici

torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')
sonuc = {}


def kaydet():
    json.dump(sonuc, open('sonuc_gorev4.json', 'w'),
              ensure_ascii=False, indent=1)


# =====================================================================
# TOKENIZER'LARI HAZIRLA
# =====================================================================
print("=" * 80)
print("4) IKI TOKENIZER, AYNI METIN, AYNI SURE")
print("=" * 80)

# Hafta 9'da egittigim tokenizer'i merges listesinden geri kuruyorum
benim = RegexTokenizer()
m = json.load(open('tokenizer_regex.json'))['merges']
benim.merges = {tuple(ikili): yeni for ikili, yeni in m}
benim._vocab_kur()
V_BENIM = 256 + len(benim.merges)
e = enc()
V_GPT2 = 50257

metin = open('turkce.txt', encoding='utf-8').read()
print(f"  metin: turkce.txt, {bn(len(metin))} karakter")
print(f"  GPT-2 tokenizer : V = {bn(V_GPT2)}")
print(f"  Hafta 9 tokenizer: V = {bn(V_BENIM)}  "
      f"({len(benim.merges)} birlestirme + 256 byte)")
print()

# =====================================================================
# SIKISTIRMA: ONCE METNI KAC TOKEN'A BOLUYORLAR
# =====================================================================
print("-" * 80)
print("4a) SIKISTIRMA - ayni metin kac token")
print("-" * 80)
t0 = time.time()
tok_gpt2 = e.encode(metin)
s_gpt2 = time.time() - t0
if os.path.exists('veri/turkce_benim.pt'):
    tok_benim = torch.load('veri/turkce_benim.pt').tolist()
    s_benim = float('nan')
    print("  (Hafta 9 tokenizer'inin cikti onbellegi kullanildi)")
else:
    t0 = time.time()
    tok_benim = benim.encode(metin)
    s_benim = time.time() - t0
    torch.save(torch.tensor(tok_benim, dtype=torch.long),
               'veri/turkce_benim.pt')

print(f"  {'tokenizer':<20}{'V':>8}{'token':>10}{'kar/token':>11}"
      f"{'kodlama':>10}")
print("  " + "-" * 60)
print(f"  {'GPT-2':<20}{V_GPT2:>8}{len(tok_gpt2):>10}"
      f"{len(metin)/len(tok_gpt2):>11.3f}{s_gpt2:>9.1f}s")
print(f"  {'Hafta 9 (benim)':<20}{V_BENIM:>8}{len(tok_benim):>10}"
      f"{len(metin)/len(tok_benim):>11.3f}"
      + (f"{s_benim:>9.1f}s" if s_benim == s_benim else f"{'onbellek':>10}"))
print("  " + "-" * 60)
oran = len(tok_benim) / len(tok_gpt2)
benim_iyi = len(tok_benim) < len(tok_gpt2)
print(f"  Benim tokenizer ayni metni {oran:.2f} kat token'a boluyor "
      f"({'DAHA AZ' if benim_iyi else 'DAHA COK'}).")
print()
print("=" * 80)
print("BU SATIRI YAZARKEN YANILMISIM")
print("=" * 80)
print(f"""  Bu testi kurarken koda su yorumu yazmistim: "1024'luk bir vocab,
  50257'lik bir vocab kadar sikistiramaz; buyuk vocab bu yarisi kaba
  kuvvetle kazanir."

  OLCUM TERSINI SOYLUYOR. 1024 token'li, Turkce'de egitilmis benim
  tokenizer'im, 50257 token'li GPT-2 tokenizer'ini Turkce metinde
  yuzde {100*(1-oran):.0f} farkla yeniyor:

    GPT-2   (V={V_GPT2}) : {len(tok_gpt2)} token, {len(metin)/len(tok_gpt2):.3f} karakter/token
    benim   (V={V_BENIM})  : {len(tok_benim)} token, {len(metin)/len(tok_benim):.3f} karakter/token

  Sebebini soyle anliyorum. Vocab buyuklugu tek basina bir sey
  soylemiyor; onemli olan o parcalarin NEREYE harcandigi. GPT-2'nin
  50 bin birlestirmesi agirlikla Ingilizce kelime parcalarina
  harcanmis - " the", " ing", " tion" gibi. Turkce metinde bu
  parcalarin cogu hic gecmiyor, o yuzden GPT-2 Turkce'yi kisa
  parcalara ve byte'lara dusuruyor. Benim 768 birlestirmem ise
  tamami Turkce'ye harcandi: "lar", "yor", "igi", "ler" gibi ekler.
  Yani 1024'un hepsi ise yariyor, 50257'nin kucuk bir kismi.

  Daha kotusu: bu cumleyi yazarken KENDI Hafta 9 olcumumle de
  celismisim. Orada 351 karakterlik Turkce bir ornekte cl100k
  (V=100277) 148 token vermisti, benim regex tokenizer'im da 148 -
  yani 100 bin vocab ile 1024 vocab BASA BASA gelmisti. O olcum
  zaten "buyukluk tek basina kazandirmaz" diyordu; ben iki hafta
  sonra tersini varsaydim.

  Ders: gecen haftalarin olcumlerini hatirlamak yetmiyor, yeni bir
  iddia yazarken onlara bakmak gerekiyor.
""")
print(f"""  Simdi onemli kisim: sikistirmayi kazanmak, ESIT SURE icinde daha
  cok METIN gormek anlamina geliyor mu? Gelmez, cunku iki etki ayri:
  sikistirma token basina kac karakter tasindigini, model buyuklugu
  ise bir token'i islemenin kac saniye tuttugunu belirliyor. GPT-2'nin
  cikis katmani {V_GPT2} satir, benimki {V_BENIM}; ayni adim suresinde
  degiller. Asil olcum bundan sonra.
""")
sonuc['sikistirma'] = dict(karakter=len(metin), v_gpt2=V_GPT2,
                           v_benim=V_BENIM, token_gpt2=len(tok_gpt2),
                           token_benim=len(tok_benim),
                           kar_token_gpt2=len(metin)/len(tok_gpt2),
                           kar_token_benim=len(metin)/len(tok_benim),
                           oran=oran)
kaydet()

# =====================================================================
# ESIT SURE BUTCESIYLE EGITIM
# =====================================================================
BUTCE = 180.0        # saniye, her iki kosu icin ayni
B, T = 8, 128

print("-" * 80)
print(f"4b) ESIT SURE BUTCESI: her tokenizer icin {BUTCE:.0f} saniye")
print("-" * 80)
print(f"""  MODEL (iki kosuda da ayni mimari)
    n_layer=4, n_head=4, n_embd=128, block_size={T}, B={B}
    weight tying acik, GPT-2 init semasi, flash attention, bf16

  DURUST OLMAK GEREKIRSE: mimari ayni ama PARAMETRE SAYISI ayni
  degil ve olamaz. Embedding tablosu V x n_embd buyuklugunde; V
  degisince model de degisiyor. Bu bir kusur degil, olcumun
  kendisinin bir parcasi: buyuk vocab'in bedeli tam olarak bu.
""")


def kosu(ad, tokenler, V, butce):
    torch.manual_seed(1337)
    cfg = GPTConfig(block_size=T, vocab_size=V, n_layer=4, n_head=4,
                    n_embd=128, flash=True)
    model = GPT(cfg, weight_tying=True, ozel_init=True)
    opt, _ = model.configure_optimizers(0.1, 3e-4, betas=(0.9, 0.95),
                                        sessiz=True)
    p = model.param_sayisi()
    veri = torch.tensor(tokenler, dtype=torch.long)
    karakter_basi_token = len(tokenler) / len(metin)

    poz, adim, egri = 0, 0, []
    t_bas = time.time()
    while time.time() - t_bas < butce:
        if poz + B * T + 1 > len(veri):
            poz = 0
        parca = veri[poz:poz + B * T + 1]
        x = parca[:-1].view(B, T)
        y = parca[1:].view(B, T)
        poz += B * T
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        adim += 1
        egri.append(dict(adim=adim, loss=loss.item(),
                         t=time.time() - t_bas))
    sure = time.time() - t_bas
    gorulen_token = adim * B * T
    gorulen_karakter = gorulen_token / karakter_basi_token
    # son 10 adimin ortalamasi: tek adimin gurultusune bakmamak icin
    son = sum(k['loss'] for k in egri[-10:]) / len(egri[-10:])
    # ORTAK OLCEK: karakter basina bit
    bpc = son * karakter_basi_token / math.log(2)
    taban = math.log(V)
    r = dict(ad=ad, V=V, param=p, adim=adim, sure=sure,
             token=gorulen_token, karakter=gorulen_karakter,
             token_sn=gorulen_token / sure,
             karakter_sn=gorulen_karakter / sure,
             ilk_loss=egri[0]['loss'], son_loss=son, taban=taban,
             taban_orani=son / taban, bpc=bpc,
             taban_bpc=taban * karakter_basi_token / math.log(2),
             egri=egri)
    del model, opt
    gc.collect()
    return r


print("  GPT-2 tokenizer'i ile kosu...", flush=True)
r_gpt2 = kosu('GPT-2 (V=50257)', tok_gpt2, V_GPT2, BUTCE)
sonuc['gpt2'] = r_gpt2
kaydet()
print(f"    {r_gpt2['adim']} adim, loss {r_gpt2['ilk_loss']:.3f} -> "
      f"{r_gpt2['son_loss']:.3f}")

print("  Hafta 9 tokenizer'i ile kosu...", flush=True)
r_benim = kosu('Hafta 9 (V=1024)', tok_benim, V_BENIM, BUTCE)
sonuc['benim'] = r_benim
kaydet()
print(f"    {r_benim['adim']} adim, loss {r_benim['ilk_loss']:.3f} -> "
      f"{r_benim['son_loss']:.3f}")
print()

# =====================================================================
# KARSILASTIRMA
# =====================================================================
print("=" * 80)
print("SONUCLAR")
print("=" * 80)
print(f"  {'':<26}{'GPT-2':>14}{'Hafta 9':>14}{'kat':>9}")
print("  " + "-" * 64)


def sat(ad, a, b, bicim='{:.0f}', ters=False):
    k = (b / a) if a else float('nan')
    print(f"  {ad:<26}{bicim.format(a):>14}{bicim.format(b):>14}"
          f"{k:>8.2f}x")


sat('parametre', r_gpt2['param'], r_benim['param'])
sat('optimizer adimi', r_gpt2['adim'], r_benim['adim'])
sat('gorulen token', r_gpt2['token'], r_benim['token'])
sat('gorulen KARAKTER', r_gpt2['karakter'], r_benim['karakter'])
sat('token/sn', r_gpt2['token_sn'], r_benim['token_sn'])
sat('karakter/sn', r_gpt2['karakter_sn'], r_benim['karakter_sn'])
print("  " + "-" * 64)
sat('son loss (nat)', r_gpt2['son_loss'], r_benim['son_loss'], '{:.4f}')
sat('taban ln(V)', r_gpt2['taban'], r_benim['taban'], '{:.4f}')
sat('loss / taban', r_gpt2['taban_orani'], r_benim['taban_orani'], '{:.4f}')
sat('ORTAK OLCEK: bit/karakter', r_gpt2['bpc'], r_benim['bpc'], '{:.4f}')
print("  " + "-" * 64)

kazanan_metin = ('GPT-2' if r_gpt2['karakter'] > r_benim['karakter']
                 else 'Hafta 9')
kazanan_bpc = 'GPT-2' if r_gpt2['bpc'] < r_benim['bpc'] else 'Hafta 9'
kat_metin = max(r_gpt2['karakter'], r_benim['karakter']) / \
    min(r_gpt2['karakter'], r_benim['karakter'])
print(f"""
  1) AYNI SUREDE HANGISI DAHA COK METIN GORDU
     {kazanan_metin} tokenizer'i, {kat_metin:.2f} kat daha cok karakter.
     Bu sonucu IKI ayri etkiye ayirabiliyorum:
       - sikistirma: GPT-2 token basina {r_gpt2['karakter']/max(r_gpt2['token'],1):.2f} karakter,
         benimki {r_benim['karakter']/max(r_benim['token'],1):.2f} karakter tasiyor
         -> bu yarisi {'benim tokenizer' if benim_iyi else 'GPT-2'} kazaniyor
       - adim hizi: GPT-2'nin cikis katmani {V_GPT2/V_BENIM:.0f} kat buyuk,
         {r_gpt2['token_sn']:.0f} token/sn'ye karsi {r_benim['token_sn']:.0f} token/sn
         -> bu yarisi da benim tokenizer kazaniyor
     Yani bu olcumde iki etki ayni yone bakiyor ve sonuc tek tarafli.
     Beklentim bunlarin TERS yone bakmasi, yani bir takasla
     karsilasmaktı: "buyuk vocab token basina cok metin tasir ama her
     token'i pahaliya isler". Olcum oyle cikmadi - Turkce metinde
     GPT-2'nin buyuk vocab'i ne sikistirmada ne hizda ise yariyor.
     Takas, tokenizer'in VERIYE UYMASI kosuluyla var; uymadiginda
     buyuk vocab iki yarisi da kaybediyor.

  2) LOSS'LARI NASIL KARSILASTIRIRIM
     DOGRUDAN KARSILASTIRAMAM. {r_gpt2['son_loss']:.2f} ile {r_benim['son_loss']:.2f}
     ayni birimde degil: biri 50257 secenekli bir tahminin, oteki 1024
     secenekli bir tahminin zorlugu. Tabanlar bile farkli:
     ln(50257)={r_gpt2['taban']:.2f}, ln(1024)={r_benim['taban']:.2f}.

     Iki durust yol var:
     (i)  TABANA GORE: loss/ln(V). GPT-2 %{100*r_gpt2['taban_orani']:.1f},
          benimki %{100*r_benim['taban_orani']:.1f}. "Hicbir sey bilmemeye gore ne kadar
          ilerledi" sorusunu olcuyor.
     (ii) KARAKTER BASINA BIT: loss'u o tokenizer'in karakter basina
          kac token urettigiyle carp, ln(2)'ye bol. Bu, "metnin bir
          karakterini kodlamak kac bit tutuyor" demek ve tokenizer'dan
          BAGIMSIZ. GPT-2 {r_gpt2['bpc']:.3f} bit/karakter,
          benimki {r_benim['bpc']:.3f} bit/karakter.
          Bu olcege gore daha iyi olan: {kazanan_bpc}.

     Hafta 9'da bits-per-character'i tokenizer'lari karsilastirmak icin
     kullanmistim; burada MODELLERI karsilastirmak icin ayni olcek ise
     yariyor. Ayni sebep: ikisinde de token birimi degisiyor, karakter
     birimi degismiyor.
""")
sonuc['karsilastirma'] = dict(kazanan_metin=kazanan_metin,
                              kazanan_bpc=kazanan_bpc, kat_metin=kat_metin,
                              butce=BUTCE)
kaydet()
print("  -> sonuc_gorev4.json kaydedildi")
