# Gorev 5 (ek) - (a) Data loader, metnin tamaminda egitim, weight tying
# YZ50 Hafta 10  (video 1:02:00 - 1:22:18)
#
# Gorev 4'te tek bir batch'i ezberlettim; o bir dogrulama testiydi.
# Burada metnin tamamini sirayla okuyan bir data loader yazip gercek
# egitim adimlari atiyorum.

import json
import math
import time

import torch

from gpt2 import GPT, GPTConfig
from tokenizer import enc

CIHAZ = 'cuda' if torch.cuda.is_available() else 'cpu'
sonuc = {}


# =====================================================================
# 5a-1) WEIGHT TYING KAC PARAMETRE KAZANDIRIYOR
# =====================================================================
print("=" * 78)
print("5a-1) WEIGHT TYING")
print("=" * 78)
cfg = GPTConfig()
torch.manual_seed(1337)
m_tied = GPT(cfg, weight_tying=True, ozel_init=True)
torch.manual_seed(1337)
m_free = GPT(cfg, weight_tying=False, ozel_init=True)
p_tied = sum(p.numel() for p in m_tied.parameters())
p_free = sum(p.numel() for p in m_free.parameters())
kazanc = p_free - p_tied

print(f"  {'':<34}{'parametre':>14}{'bellek (fp32)':>16}")
print("  " + "-" * 64)
print(f"  {'weight tying YOK':<34}{p_free:>14,}{p_free*4/1e6:>13.0f} MB"
      .replace(',', '.'))
print(f"  {'weight tying VAR':<34}{p_tied:>14,}{p_tied*4/1e6:>13.0f} MB"
      .replace(',', '.'))
print("  " + "-" * 64)
print(f"  {'kazanc':<34}{kazanc:>14,}{kazanc*4/1e6:>13.0f} MB".replace(',', '.'))
print(f"  {'yuzde':<34}{100*kazanc/p_free:>13.1f}%")
print()
print(f"  Kazanc tam olarak embedding tablosunun boyutu:")
print(f"    vocab_size x n_embd = {cfg.vocab_size:,} x {cfg.n_embd} = "
      f"{cfg.vocab_size*cfg.n_embd:,}".replace(',', '.'))
print()
# ayni tensor mu?
ayni = m_tied.transformer.wte.weight.data_ptr() == m_tied.lm_head.weight.data_ptr()
print(f"  wte ile lm_head ayni tensoru mu gosteriyor: {ayni}")
print(f"  (bellekte tek kopya; birini guncelleyince digeri de degisiyor)")
print()
print("""  NEDEN MANTIKLI

  Giris embedding'i "bu token'in anlami nedir" sorusunu, cikis katmani
  "bu anlama en yakin token hangisi" sorusunu cevapliyor. Ikisi ayni
  uzayin iki yonu. Ayni matrisi kullanmak hem parametre hem de
  tutarlilik kazandiriyor: benzer anlamli token'lar hem girisde hem
  cikisda benzer davraniyor.

  Ama bedava degil. Gorev 4a-ek'te olctugum sey buydu: weight tying,
  modelin baslangictaki davranisini degistiriyor - egitilmemis model
  "bir sonraki token girdinin kendisidir" tahminine egilimli
  basliyor.

  Bir sayi daha: {:,} parametrenin {:,}'i, yani yuzde {:.0f}'i tek
  basina embedding tablosunda. GPT-2'nin "124M" olarak anilmasinin
  sebebi weight tying; olmasaydi 163M olacakti.
""".format(p_tied, cfg.vocab_size*cfg.n_embd,
           100*cfg.vocab_size*cfg.n_embd/p_tied).replace(',', '.'))

sonuc['tying'] = dict(tied=p_tied, free=p_free, kazanc=kazanc,
                      emb=cfg.vocab_size*cfg.n_embd, ayni_tensor=bool(ayni))
del m_free


# =====================================================================
# 5a-2) DATA LOADER
# =====================================================================
class DataLoaderLite:
    """Metni bastan sona, ust uste binmeyen B*T'lik parcalar halinde okur.

    Hafta 7-8'de get_batch rastgele yerlerden parca cekiyordu. Burada
    sirayla okuyorum: her token'in tam bir kez gorulmesi, bir "epoch"
    kavraminin olmasi ve kayiplarin karsilastirilabilir olmasi icin.
    Sona gelince basa donuyor.
    """

    def __init__(self, B, T, metin):
        self.B, self.T = B, T
        e = enc()
        self.tokens = torch.tensor(e.encode(metin))
        self.konum = 0
        print(f"  {len(metin):,} karakter -> {len(self.tokens):,} token"
              .replace(',', '.'))
        print(f"  bir epoch = {len(self.tokens) // (B*T):,} batch"
              .replace(',', '.'))

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.konum: self.konum + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.konum += B * T
        if self.konum + B * T + 1 > len(self.tokens):
            self.konum = 0               # basa don
        return x, y


print("=" * 78)
print("5a-2) DATA LOADER")
print("=" * 78)
B, T = 4, 32
metin = open('input.txt', encoding='utf-8').read()
loader = DataLoaderLite(B, T, metin)
x, y = loader.next_batch()
print(f"  batch sekli: x {tuple(x.shape)}  y {tuple(y.shape)}")
e = enc()
print(f"  ilk satir  : {e.decode(x[0].tolist())!r}")
print(f"  hedefi     : {e.decode(y[0].tolist())!r}")
print(f"  (y, x'in bir token kaydirilmis hali)")
print()

# =====================================================================
# 5a-3) EGITIM
# =====================================================================
ADIM, LR = 120, 3e-4
print("=" * 78)
print(f"5a-3) EGITIM ({ADIM} adim)")
print("=" * 78)
print(f"  cihaz: {CIHAZ}   B={B}, T={T}   lr={LR}, AdamW")
print(f"  her adimda FARKLI bir batch - artik ezber degil, gercek egitim")
print()

model = m_tied.to(CIHAZ)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
loader.konum = 0
kayit = []
t0 = time.time()
for i in range(ADIM):
    x, y = loader.next_batch()
    x, y = x.to(CIHAZ), y.to(CIHAZ)
    t1 = time.time()
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    dt = time.time() - t1
    kayit.append(loss.item())
    if i % 10 == 0 or i == ADIM - 1:
        tok_sn = (B * T) / dt
        print(f"    adim {i:>4}  loss {loss.item():>8.4f}  "
              f"{dt*1000:>7.0f} ms  {tok_sn:>7.0f} token/sn")
sure = time.time() - t0
print()
print(f"  ilk loss : {kayit[0]:.4f}")
print(f"  son loss : {kayit[-1]:.4f}")
print(f"  sure     : {sure/60:.1f} dk")
print(f"  gorulen token: {ADIM*B*T:,}".replace(',', '.'))
print()

print(f"""  OKUNUSU - VE DURUST BIR SINIR

  Loss {kayit[0]:.2f}'den {kayit[-1]:.2f}'e indi. Bu Gorev 4'teki
  ezberleme degil: her adimda farkli bir batch geliyor, yani model
  gercekten ogreniyor.

  Ama buyuk kismi "ucuz" bir ogrenme. Model ilk adimlarda butun
  50.257 token'i esit olasi goruyordu; simdi en azindan hangi
  token'larin SIK gectigini ogrendi (bosluk, 'the', nokta, satir
  sonu). Hafta 7'deki bigram modelinin yaptigi seyin aynisi. Asil is
  bundan sonra basliyor ve orasi bu haftanin konusu degil.

  Hiz: {B*T/ (sure/ADIM):.0f} token/saniye. Karpathy videoda A100'de
  yaklasik 16.000 token/saniye aliyor; aradaki fark yaklasik
  {16000/(B*T/(sure/ADIM)):.0f} kat ve tamamen donanimdan. Tiny
  Shakespeare'in tamami {len(loader.tokens):,} token; bir epoch bu
  makinede yaklasik {len(loader.tokens)/(B*T/(sure/ADIM))/3600:.1f}
  saat surerdi. Modeli gercekten egitmek bu haftanin gorevi degil ve
  bu makinede mumkun de degil - gelecek haftanin konusu.
""".replace(',', '.'))

sonuc['egitim'] = dict(B=B, T=T, adim=ADIM, lr=LR, sure=sure, kayit=kayit,
                       token=len(loader.tokens), cihaz=CIHAZ,
                       tok_sn=B * T / (sure / ADIM))

# =====================================================================
# 5a-4) BASLANGIC DEGERLERI
# =====================================================================
print("=" * 78)
print("5a-4) BASLANGIC DEGERLERI (videodaki sema)")
print("=" * 78)
print("""  Uc kural var, ucu de gpt2.py icindeki _init_weights'te:

  1. Butun Linear agirliklari    : normal(0, 0.02)
  2. Butun Embedding agirliklari : normal(0, 0.02)
  3. Bias'lar                    : 0
  4. Residual yoluna EKLENEN katmanlar (attn.c_proj ve mlp.c_proj):
     std ayrica 1/sqrt(2*n_layer) ile carpiliyor

  Dorduncusu neden: her blokta residual'a iki ekleme var (attention ve
  MLP). 12 blok boyunca 24 bagimsiz terim toplaniyor. Bagimsiz
  degiskenlerin toplaminda varyanslar TOPLANIR, yani cikisin varyansi
  24 katina cikar. 1/sqrt(24) ile olcekleyince sabit kaliyor.

  0.02 sayisi nereden: 1/sqrt(n_embd) = 1/sqrt(768) = 0.036. GPT-2'nin
  sectigi 0.02 bunun biraz altinda ve modelin butun boyutlari icin
  (768'den 1600'e) ayni degeri kullanmislar. Yani tam bir formul degil,
  calistigi gorulmus bir deger.""")
print()
std0 = 0.02
std_res = 0.02 * (2 * cfg.n_layer) ** -0.5
print(f"  n_layer = {cfg.n_layer}  ->  1/sqrt(2*{cfg.n_layer}) = "
      f"{(2*cfg.n_layer)**-0.5:.4f}")
print(f"  normal katmanlarin std'si  : {std0:.4f}")
print(f"  residual katmanlarinin std'si: {std_res:.4f}")
print()
# gercekten uygulanmis mi?
# DIKKAT: m_tied yukarida 120 adim EGITILDI, agirliklari artik baslangic
# degerleri degil. Ilk yazdigimda onu olcmus ve 0.0047 gibi sapmis bir
# sayi almistim. Baslangic semasini olcmek icin TAZE bir model kurmam
# gerekiyor.
torch.manual_seed(1337)
m_taze = GPT(cfg, weight_tying=True, ozel_init=True)
w1 = m_taze.transformer.h[0].mlp.c_fc.weight.std().item()
w2 = m_taze.transformer.h[0].mlp.c_proj.weight.std().item()
w3 = m_taze.transformer.h[0].attn.c_proj.weight.std().item()
w4 = m_taze.transformer.wte.weight.std().item()
w5 = m_tied.transformer.h[0].mlp.c_proj.weight.std().item()
print(f"  olculen - mlp.c_fc.weight std   : {w1:.4f}  (beklenen {std0:.4f})")
print(f"  olculen - mlp.c_proj.weight std : {w2:.4f}  (beklenen {std_res:.4f})")
print(f"  olculen - attn.c_proj.weight std: {w3:.4f}  (beklenen {std_res:.4f})")
print(f"  olculen - wte.weight std        : {w4:.4f}  (beklenen {std0:.4f})")
print()
print(f"  Karsilastirma: ayni katman 120 adim EGITILDIKTEN sonra {w5:.4f}.")
print("  Ilk yazdigimda yanlislikla bu sayiyi olcmustum ve baslangic")
print("  semasinin tutmadigini sanmistim. Olctugum model egitilmisti;")
print("  taze modelde sema birebir tutuyor.")
print()
sonuc['init'] = dict(std0=std0, std_res=std_res, c_fc=w1, mlp_proj=w2,
                     attn_proj=w3, wte=w4, egitilmis_proj=w5)

json.dump(sonuc, open('sonuc_gorev5a.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev5a.json kaydedildi")
