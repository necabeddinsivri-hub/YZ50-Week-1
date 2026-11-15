# Gorev 1 - Data loader, weight tying ve baslangic degerleri
# YZ50 Hafta 11  (video 1:02:00 - 1:22:18)
#
# Bu uc konuyu Hafta 10'un ek gorevinde (5a) yapmistim. Mail "gecen
# hafta yaptiysan kisaca gostermen yeterli" diyor. O yuzden burada
# sayilari kisa tutuyorum ve yerine CEVABINI BILMEDIGIM bir soruyu
# olcuyorum: 1/sqrt(2*n_layer) olceklemesi olmasa ne olurdu?
#
# Gerekcesini Hafta 10'da yazmistim ama hic OLCMEMISTIM. Iddiayi
# yazmak ile gostermek arasindaki farki Hafta 10'da ogrendim: dort
# tahminim olcumle yanlislandi. O yuzden bu sefer once olcuyorum.

import json
import math
import time

import torch

from gpt2 import GPT, GPTConfig
from tokenizer import enc
from veri import VeriYukleyici

torch.manual_seed(1337)
sonuc = {}

# =====================================================================
# 1a) DATA LOADER: METNIN TAMAMI
# =====================================================================
print("=" * 78)
print("1a) DATA LOADER - METNIN TAMAMINDA GEZINMEK")
print("=" * 78)
print("""
  Hafta 10'da modele TEK bir batch'i 300 kez verip ezberletmistim.
  Loss 10.96'dan 0.00075'e dusmustu. O bir saglik testiydi: "kod
  ogrenebiliyor mu?" sorusunun cevabi. Egitim degildi, cunku model
  metnin yalnizca 1024 token'ini gormustu.

  Simdi sirayla gezen bir yukleyici yaziyorum (veri.py). Isi basit:
  token dizisini B*T'lik dilimlere bol, sirayla ver, sona gelince
  basa don. Hafta 8'deki yukleyicimden farki RASTGELE degil SIRALI
  olmasi - boylece metnin tamamini gordugumu garanti ediyorum ve
  "bir devir bitti" diyebiliyorum.
""")
B, T = 4, 256
e = enc()
yuk = VeriYukleyici('input.txt', B, T, lambda s: e.encode(s))
print()
x, y = yuk.sonraki()
print(f"  ilk batch: x {tuple(x.shape)}, y {tuple(y.shape)}")
print(f"  x'in ilk 8 token'i: {x[0, :8].tolist()}")
print(f"  y'nin ilk 8 token'i: {y[0, :8].tolist()}")
print(f"  -> y, x'in bir token kaydirilmis hali: "
      f"{x[0, 1:8].tolist() == y[0, :7].tolist()}")
sonuc['loader'] = dict(B=B, T=T, karakter=yuk.karakter,
                       token=len(yuk.tokenler),
                       token_karakter=len(yuk.tokenler) / yuk.karakter,
                       devir_batch=yuk.devir_basi_batch)
print()

# =====================================================================
# 1b) WEIGHT TYING
# =====================================================================
print("=" * 78)
print("1b) WEIGHT TYING - KAC PARAMETRE KAZANDIRIYOR")
print("=" * 78)
cfg = GPTConfig(block_size=T)
m_ayri = GPT(cfg, weight_tying=False)
m_bagli = GPT(cfg, weight_tying=True)
p_ayri = m_ayri.param_sayisi()
p_bagli = m_bagli.param_sayisi()
kazanc = p_ayri - p_bagli
V, C = cfg.vocab_size, cfg.n_embd
print(f"  weight tying YOK : {p_ayri:>12,} parametre".replace(',', '.'))
print(f"  weight tying VAR : {p_bagli:>12,} parametre".replace(',', '.'))
print(f"  kazanc           : {kazanc:>12,} parametre".replace(',', '.')
      + f"  (%{100*kazanc/p_ayri:.1f})")
print(f"  beklenen (V*C)   : {V*C:>12,} = {V:,} x {C}".replace(',', '.'))
print(f"  tutuyor mu       : {kazanc == V * C}")
print("""
  Mantigi: wte giris tarafinda "bu token id'si hangi vektor" diyor,
  lm_head cikis tarafinda "bu vektor hangi token'a ne kadar yakin"
  diyor. Ikisi de token ile vektor arasinda ayni sozlugu kuruyor,
  sadece ters yonde. Ayri ogrenmeleri icin bir sebep yok; ayni
  tensoru paylasiyorlar.

  124M'lik modelde bu 38,6 milyon parametre, yani modelin yaklasik
  ucte biri. Hafta 10'da dosyada lm_head'i BULAMAMIS ve once hata
  sanmistim; sebebi buydu.
""")
sonuc['tying'] = dict(ayri=p_ayri, bagli=p_bagli, kazanc=kazanc,
                      beklenen=V * C, tutuyor=bool(kazanc == V * C))
del m_ayri

# =====================================================================
# 1c) BASLANGIC DEGERLERI
# =====================================================================
print("=" * 78)
print("1c) BASLANGIC DEGERLERI: std 0.02 VE 1/sqrt(2*n_layer)")
print("=" * 78)
m = GPT(cfg, weight_tying=True, ozel_init=True)
olcek = (2 * cfg.n_layer) ** -0.5
print(f"  olcek carpani: 1/sqrt(2*{cfg.n_layer}) = {olcek:.4f}")
print(f"  yani residual katmanlarin std'si 0.02 * {olcek:.4f} = "
      f"{0.02*olcek:.5f}")
print()
olcumler = [
    ('wte  (embedding)', m.transformer.wte.weight, 0.02),
    ('h.0.attn.c_attn  (residual DEGIL)', m.transformer.h[0].attn.c_attn.weight, 0.02),
    ('h.0.mlp.c_fc     (residual DEGIL)', m.transformer.h[0].mlp.c_fc.weight, 0.02),
    ('h.0.attn.c_proj  (residual, OLCEKLI)', m.transformer.h[0].attn.c_proj.weight, 0.02 * olcek),
    ('h.0.mlp.c_proj   (residual, OLCEKLI)', m.transformer.h[0].mlp.c_proj.weight, 0.02 * olcek),
    ('h.11.mlp.c_proj  (residual, OLCEKLI)', m.transformer.h[11].mlp.c_proj.weight, 0.02 * olcek),
]
print(f"  {'tensor':<40}{'olculen std':>13}{'beklenen':>11}")
print("  " + "-" * 66)
init_kayit = []
for ad, t, bekl in olcumler:
    s = t.std().item()
    init_kayit.append(dict(ad=ad, olculen=s, beklenen=bekl))
    print(f"  {ad:<40}{s:>13.5f}{bekl:>11.5f}")
print("  " + "-" * 66)
print("""
  Hafta 10'da bu olcumu YANLIS yapmistim: std'leri EGITILMIS modelde
  olcup 0.0047 bulmus, "0.0041 beklerken 0.0047 cikti" diye not
  dusmustum. Egitim agirliklari degistirdigi icin o sayi baslangic
  semasini hic olcmuyordu. Burada model TAZE, hic adim atmamis.
""")
sonuc['init'] = dict(olcek=olcek, kayit=init_kayit)

# =====================================================================
# 1d) OLCEKLEMEYI NEDEN YAPIYORUZ - IDDIA DEGIL, OLCUM
# =====================================================================
print("=" * 78)
print("1d) OLCEKLEME OLMASA NE OLURDU? (olcum)")
print("=" * 78)
print("""  KENDI CUMLELERIMLE NEDEN OLCEKLIYORUZ

  Residual akisi (x = x + attn(...), x = x + mlp(...)) bir toplam.
  Her blok akisa bir sey EKLIYOR, ustune yazmiyor. Bagimsiz ve
  ortalamasi sifir olan n tane seyi toplarsam varyanslari toplanir,
  standart sapma sqrt(n) kat buyur. 12 blokta blok basina iki ekleme
  var, yani 2*12 = 24 terim. Hicbir sey yapmazsam akisin std'si
  kabaca sqrt(24) = 4.90 kat buyumeli.

  Bu neden kotu: son LayerNorm akisi tekrar normalize eder, yani
  modelin ilk adimdaki ciktisi kurtarilabilir. Ama ara katmanlar
  gittikce buyuyen sayilarla calisir; GELU'nun ve softmax'in
  dogrusal-olmayan bolgelerine savrulur ve gradyanlar ilk bloklara
  zayif gider. Yani "patlamak" degil, "derinlestikce dengesizlesmek".

  Carpani 1/sqrt(2*n_layer) secmek tam bu sqrt(24)'u geri aliyor:
  her terimi 1/sqrt(24) ile kuculttugumde 24 terimin toplami yine
  bir terim mertebesinde kaliyor.

  SIMDI OLCUYORUM. Iki taze model kuruyorum - biri olcekli, biri
  olceksiz - ayni batch'i veriyorum ve her blogun cikisindaki
  akisin std'sine bakiyorum.
""")


def akis_profili(model, x):
    """Her blogun cikisindaki residual akisinin std'sini dondurur."""
    kayit = []
    kancalar = []

    def yap(i):
        def kanca(mod, girdi, cikti):
            kayit.append((i, cikti.detach().std().item()))
        return kanca

    for i, blok in enumerate(model.transformer.h):
        kancalar.append(blok.register_forward_hook(yap(i)))
    with torch.no_grad():
        model(x)
    for k in kancalar:
        k.remove()
    kayit.sort()
    return [s for _, s in kayit]


xb = x[:1]        # tek dizi yeter, olcum icin
torch.manual_seed(1337)
m_olcekli = GPT(cfg, weight_tying=True, ozel_init=True, residual_olcek=True)
torch.manual_seed(1337)
m_olceksiz = GPT(cfg, weight_tying=True, ozel_init=True, residual_olcek=False)

p_olcekli = akis_profili(m_olcekli, xb)
p_olceksiz = akis_profili(m_olceksiz, xb)

print(f"  {'blok':<6}{'OLCEKLI std':>14}{'OLCEKSIZ std':>15}{'oran':>9}")
print("  " + "-" * 46)
for i, (a, b) in enumerate(zip(p_olcekli, p_olceksiz)):
    print(f"  {i:<6}{a:>14.3f}{b:>15.3f}{b/a:>9.2f}x")
print("  " + "-" * 46)
buyume_olcekli = p_olcekli[-1] / p_olcekli[0]
buyume_olceksiz = p_olceksiz[-1] / p_olceksiz[0]
teorik = math.sqrt(2 * cfg.n_layer)
print(f"  ilk bloktan son bloga buyume:")
print(f"    OLCEKLI  : {buyume_olcekli:.2f}x")
print(f"    OLCEKSIZ : {buyume_olceksiz:.2f}x")
print(f"    teorik beklenti (sqrt(2*n_layer)) : {teorik:.2f}x")
print()

oran_son = p_olceksiz[-1] / p_olcekli[-1]
oran_ilk = p_olceksiz[0] / p_olcekli[0]
print("-" * 78)
print("YANLIS SEYI TAHMIN ETMISIM")
print("-" * 78)
print(f"""  Yukaridaki paragrafta "hicbir sey yapmazsam akisin std'si kabaca
  sqrt(24) = {teorik:.2f} kat buyumeli" yazdim ve bunu DERINLIK BOYUNCA
  buyume olarak kurmusum. Olcum bunu desteklemiyor:

    olceksiz modelde derinlik boyunca buyume : {buyume_olceksiz:.2f}x
    olcekli modelde derinlik boyunca buyume  : {buyume_olcekli:.2f}x

  Ikisi de buyuyor ve ikisi de {teorik:.2f} degil. sqrt(24) ile esleşen
  sayi bambaska bir yerde: AYNI DERINLIKTE iki modelin ORANI.

    blok 0'da  olceksiz/olcekli : {oran_ilk:.2f}x
    blok 11'de olceksiz/olcekli : {oran_son:.2f}x      <- sqrt(24)={teorik:.2f}

  Hatami soyle anliyorum. Blok 0'in CIKTISI zaten bir toplam (girdi +
  iki ekleme), yani "tek terim" degil; onu bolen yaptigimda sqrt(n)
  yasasini olcmuyorum, iki farkli toplami oranliyorum. Olceklemenin
  yaptigi sey akisi SABIT tutmak degil - residual akisi tanimi geregi
  birikiyor, {buyume_olcekli:.2f}x hala buyuyor. Yaptigi sey her EKLEMEYI
  1/sqrt(2*n_layer) kadar kuculterek nihai buyuklugu tek bir katmanin
  mertebesinde tutmak.

  Yani "olcekleme akisi sabit tutar" cumlesi yanlisti, dogrusu
  "olcekleme akisin seviyesini {oran_son:.1f} kat asagi ceker ve derinlige
  gore buyume oranini {buyume_olceksiz:.2f}x'ten {buyume_olcekli:.2f}x'e indirir".
  Teorinin dogru oldugu yer de burasi: {oran_son:.2f} ile {teorik:.2f} cok yakin.
""")
sonuc['akis'] = dict(olcekli=p_olcekli, olceksiz=p_olceksiz,
                     buyume_olcekli=buyume_olcekli,
                     buyume_olceksiz=buyume_olceksiz, teorik=teorik,
                     oran_ilk=oran_ilk, oran_son=oran_son)

# =====================================================================
# 1e) METNIN TAMAMINDA KISA BIR EGITIM
# =====================================================================
print("=" * 78)
print("1e) YUKLEYICIYLE EGITIM: METIN ILERLIYOR MU?")
print("=" * 78)
model = GPT(cfg, weight_tying=True, ozel_init=True)
opt, wd = model.configure_optimizers(0.1, 3e-4)
yuk.basa_al()
TABAN = math.log(cfg.vocab_size)
print(f"  egitimsiz taban: ln({cfg.vocab_size}) = {TABAN:.4f}")
print()
print(f"  {'adim':>5}{'loss':>9}{'sure':>8}{'tok/sn':>9}   gorulen metin")
print("  " + "-" * 62)
ADIM = 30
egri = []
t_bas = time.time()
for adim in range(ADIM):
    t0 = time.time()
    xb_, yb_ = yuk.sonraki()
    opt.zero_grad(set_to_none=True)
    _, loss = model(xb_, yb_)
    loss.backward()
    opt.step()
    dt = time.time() - t0
    egri.append(dict(adim=adim, loss=loss.item(), dt=dt,
                     token=yuk.verilen_token))
    if adim < 3 or adim % 10 == 9 or adim == ADIM - 1:
        print(f"  {adim:>5}{loss.item():>9.4f}{dt:>8.2f}s"
              f"{B*T/dt:>9.0f}   {yuk.verilen_token:>7} token, "
              f"%{100*yuk.verilen_token/len(yuk.tokenler):.1f} / devir")
print("  " + "-" * 62)
toplam = time.time() - t_bas
print(f"  {ADIM} adim {toplam:.0f} saniye surdu. "
      f"Loss {egri[0]['loss']:.2f} -> {egri[-1]['loss']:.2f}")
print(f"""
  Dikkat: bu 30 adim bir devrin yalnizca
  %{100*yuk.verilen_token/len(yuk.tokenler):.1f}'i. Metnin tamamini bir kez gormek icin
  {yuk.devir_basi_batch} adim gerekiyor, bu CPU'da yaklasik
  {yuk.devir_basi_batch*toplam/ADIM/60:.0f} dakika. Bu yuzden Gorev 2'de
  once HIZLANDIRIYORUM, Gorev 3'te de GPT-3'un ayarlariyla
  gradient accumulation kuruyorum - amac ayni sure icinde daha cok
  metin gormek.
""")
sonuc['egitim'] = dict(adim=ADIM, egri=egri, toplam_sure=toplam,
                       taban=TABAN, wd=wd)

json.dump(sonuc, open('sonuc_gorev1.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev1.json kaydedildi")
