# Gorev 2 - Transformer blogunu parca parca tamamla
# YZ50 Hafta 8  (video 1:24:25 - 1:37:49)
#
# Sirayla ekliyorum ve her adimda val loss'u yaziyorum:
#   4) feedforward
#   -) 3 blok ama residual YOK          <- teshis icin, derinlesmenin bedeli
#   5) 3 blok + residual
#   6) 3 blok + residual + LayerNorm
#
# Sonra LayerNorm'u Hafta 6'da yazdigim BatchNorm1d ile karsilastiriyorum.

import json
import torch
import torch.nn as nn
from ortak import (veri, V, BLOCK, BATCH, loss_olc, egit, decode,
                   param_sayisi)
from katmanlar import AblasyonModeli, LayerNorm1d, N_EMBD, uret
from batchnorm_h6 import BatchNorm1d

ADIM, LR = 5000, 1e-3
sonuc = json.load(open('sonuclar.json'))

print("=" * 70)
print("GOREV 2 - TRANSFORMER BLOGU, PARCA PARCA")
print("=" * 70)
print()


def kos(anahtar, ad, asama, **kw):
    print("-" * 70)
    print(ad)
    print("-" * 70)
    torch.manual_seed(1337)
    m = AblasyonModeli(asama, **kw)
    p = param_sayisi(m)
    print(f"  parametre: {p:,}".replace(',', '.'))
    sure = egit(m, adim=ADIM, lr=LR, sessiz=True)
    s = loss_olc(m)
    print(f"  train {s['train']:.4f}   val {s['val']:.4f}   ({sure:.0f} sn)")
    sonuc[anahtar] = dict(ad=ad.split(' - ')[0], param=p, train=s['train'],
                          val=s['val'], sure=sure)
    print()
    return m, s


# ---------------------------------------------------------------------
# 4) FEEDFORWARD EKLENDI
# ---------------------------------------------------------------------
print("  Attention iletisimdir: tokenlar birbirine bakar, bilgi toplar.")
print("  Ama topladigi bilgiyi SINDIRECEK bir adim yok - dogrudan logit'e")
print("  ceviriyordum. FeedForward tam o bosluga giriyor: her token kendi")
print("  basina, 32 -> 128 -> 32 bir sinir agindan geciyor.")
print()
kos('4_feedforward', '4) multi-head + feedforward', 'ff', n_head=4)


# ---------------------------------------------------------------------
# ARA ADIM: 3 BLOK, RESIDUAL YOK  (derinlik tek basina ise yaramiyor)
# ---------------------------------------------------------------------
print("  Simdi 3 blogu ust uste koyuyorum - ama residual olmadan.")
print("  Beklenti: derinlesince daha iyi olsun. Bakalim.")
print()
kos('4b_blok_duz', '4b) 3 blok, residual YOK (teshis)', 'blok_duz',
    n_head=4, n_blok=3)


# ---------------------------------------------------------------------
# 5) RESIDUAL EKLENDI
# ---------------------------------------------------------------------
print("  Residual: x = x + alt_katman(x).")
print("  Toplama dugumunun gradyani 1'dir (Hafta 1-2): yani gradyan")
print("  katmanlardan GECMEK zorunda degil, otoyoldan dosdogru")
print("  embedding'e kadar akiyor. Derin agin egitilebilmesinin sebebi bu.")
print()
kos('5_residual', '5) 3 blok + residual', 'residual', n_head=4, n_blok=3)


# ---------------------------------------------------------------------
# 6) LAYERNORM EKLENDI
# ---------------------------------------------------------------------
print("  LayerNorm: her token'in 32 boyutlu vektorunu, kendi icinde")
print("  ortalamasi 0 varyansi 1 olacak sekilde yeniden olcekliyor.")
print("  Blogun ONUNE koyuyorum (pre-norm): x = x + sa(ln(x)).")
print()
ln_model, s_ln = kos('6_layernorm', '6) 3 blok + residual + LayerNorm', 'ln',
                     n_head=4, n_blok=3)


# ---------------------------------------------------------------------
# TEK TABLO  (gorevde istenen)
# ---------------------------------------------------------------------
print("=" * 70)
print("ABLASYON TABLOSU")
print("=" * 70)
sira = ['1_bigram', '2_tek_head', '3_multi_head', '4_feedforward',
        '4b_blok_duz', '5_residual', '6_layernorm']
print(f"  {'#':<3}{'model':<34}{'param':>9}{'train':>9}{'val':>9}{'delta':>9}")
print("  " + "-" * 73)
onceki = None
for i, k in enumerate(sira, 1):
    r = sonuc[k]
    d = '' if onceki is None else f"{onceki - r['val']:+.4f}"
    yildiz = ' *' if k == '4b_blok_duz' else ''
    print(f"  {i:<3}{r['ad'] + yildiz:<34}{r['param']:>9,}"
          f"{r['train']:>9.4f}{r['val']:>9.4f}{d:>9}".replace(',', '.'))
    if k != '4b_blok_duz':        # teshis satiri zinciri bozmasin
        onceki = r['val']
print("  " + "-" * 73)
print("  (* teshis satiri; ana zincirin disinda)")
print(f"  bigram -> LayerNorm toplam kazanc: "
      f"{sonuc['1_bigram']['val'] - sonuc['6_layernorm']['val']:+.4f}")
print()


# =====================================================================
# LAYERNORM vs HAFTA 6'DAKI BATCHNORM1D
# =====================================================================
print("=" * 70)
print("LAYERNORM vs BATCHNORM1D  (Hafta 6'da yazdigim sinif)")
print("=" * 70)
print()

torch.manual_seed(0)
x = torch.randn(4, 6)            # 4 ornek, 6 ozellik - elle takip edilebilir
print("  Test girdisi x: (4 ornek, 6 ozellik)")
for i in range(4):
    print("    " + " ".join(f"{v:>7.3f}" for v in x[i]))
print()

bn = BatchNorm1d(6)
ln = LayerNorm1d(6)
ybn = bn(x)
yln = ln(x)

print("  --- BatchNorm1d cikisi ---")
for i in range(4):
    print("    " + " ".join(f"{v:>7.3f}" for v in ybn[i]))
print(f"    SUTUN ortalamalari : {[round(v, 4) for v in ybn.mean(0).tolist()]}")
print(f"    SUTUN std          : {[round(v, 4) for v in ybn.std(0, unbiased=False).tolist()]}")
print(f"    satir ortalamalari : {[round(v, 4) for v in ybn.mean(1).tolist()]}  <- 0 DEGIL")
print()

print("  --- LayerNorm1d cikisi (kendi yazdigim) ---")
for i in range(4):
    print("    " + " ".join(f"{v:>7.3f}" for v in yln[i]))
print(f"    SATIR ortalamalari : {[round(v, 4) for v in yln.mean(1).tolist()]}")
print(f"    SATIR std          : {[round(v, 4) for v in yln.std(1, unbiased=False).tolist()]}")
print(f"    sutun ortalamalari : {[round(v, 4) for v in yln.mean(0).tolist()]}  <- 0 DEGIL")
print()

# kendi LayerNorm'um PyTorch'un LayerNorm'u ile ayni mi?
tln = nn.LayerNorm(6, elementwise_affine=False)
fark = (yln - tln(x)).abs().max().item()
print(f"  Kendi LayerNorm1d'im ile nn.LayerNorm arasindaki en buyuk fark: "
      f"{fark:.2e}")
print("  (sifira yakin: dogru yazmisim)")
print()
print("  BESSEL TUZAGI: ilk yazdigimda x.var(1) yazmistim. PyTorch'ta")
print("  var() varsayilan olarak n-1'e boler (Bessel duzeltmesi, Hafta 6).")
print("  nn.LayerNorm ise n'e boluyor. Fark:")


class _LN_n_eksi_1(LayerNorm1d):
    def __call__(self, x):
        m = x.mean(1, keepdim=True)
        v = x.var(1, keepdim=True)                  # unbiased=True (varsayilan)
        return self.gamma * (x - m) / torch.sqrt(v + self.eps) + self.beta


y_hatali = _LN_n_eksi_1(6)(x)
print(f"    n-1'e bolen (hatali) surum ile nn.LayerNorm farki: "
      f"{(y_hatali - tln(x)).abs().max().item():.2e}")
print(f"    n'e bolen (dogru)   surum ile nn.LayerNorm farki: {fark:.2e}")
print(f"    hatali surumun satir std'si: {y_hatali.std(1, unbiased=False)[0]:.4f}"
      f"  (sqrt(5/6) = {(5 / 6) ** .5:.4f})")
print()
print("  NEDEN farkli: Hafta 6'da n-1 kullanmamin sebebi, batch'in butun")
print("  verinin bir ORNEGI olmasiydi; ornekten evren varyansini")
print("  kestirirken n-1 gerekiyor. LayerNorm'da normalize ettigim sey tek")
print("  bir token'in butun ozellikleri - kestirim yok, elimdeki veri")
print("  evrenin kendisi. O yuzden n'e bolunuyor. Ayni formul, farkli")
print("  istatistiksel durum.")
print()

print("  --- Kod farki: TEK KARAKTER ---")
print("    BatchNorm1d :  xmean = x.mean(0, keepdim=True)   # sutun")
print("    LayerNorm1d :  xmean = x.mean(1, keepdim=True)   # satir")
print()

# ---------------------------------------------------------------------
# NEDEN RUNNING_MEAN TUTMUYOR?
# ---------------------------------------------------------------------
print("  --- NEDEN LayerNorm running_mean TUTMUYOR? ---")
print()
print("  Deney: ayni ornegi (1) 4'luk bir batch icinde, (2) tek basina ver.")
print()
tek = x[0:1]                                   # (1,6)

bn2 = BatchNorm1d(6)
_ = bn2(x)                                     # batch halinde (egitim kipi)
toplu_cikti = bn2.out[0].clone()
bn2.training = True
tek_cikti_bn = bn2(tek)[0]
print("  BatchNorm (egitim kipi):")
print(f"    batch icinde : {[round(v, 3) for v in toplu_cikti.tolist()]}")
print(f"    tek basina   : {[round(v, 3) for v in tek_cikti_bn.tolist()]}")
print(f"    en buyuk fark: {(toplu_cikti - tek_cikti_bn).abs().max():.4f}")
print("    Ayni ornek, farkli cikti! Cunku ortalama batch'ten geliyor.")
print("    Tek ornekte batch varyansi 0/tanimsiz -> cikti anlamsiz.")
print("    Iste bu yuzden BatchNorm cikarim icin running_mean/var TUTMAK")
print("    ZORUNDA: 'egitim sirasinda gordugum tipik ortalama buydu' diye.")
print()

ln2 = LayerNorm1d(6)
tek_cikti_ln = ln2(tek)[0]
print("  LayerNorm:")
print(f"    batch icinde : {[round(v, 3) for v in yln[0].tolist()]}")
print(f"    tek basina   : {[round(v, 3) for v in tek_cikti_ln.tolist()]}")
print(f"    en buyuk fark: {(yln[0] - tek_cikti_ln).abs().max():.2e}")
print("    Bire bir ayni. Her satir kendi ortalamasini kullaniyor,")
print("    batch'te kac ornek oldugunun hic onemi yok.")
print("    -> biriktirilecek bir istatistik YOK -> running_mean YOK")
print("    -> egitim/cikarim ayrimi YOK -> self.training bayragi bile YOK")
print()

print("""  KENDI CUMLELERIMLE

  Hafta 6'da BatchNorm yazarken en cok ugrastigim sey iki kiplik mesele-
  siydi: egitimde batch'in ortalamasini kullan, cikarimda biriktirdigin
  ortalamayi kullan. Hatta egitimden sonra ayri bir dongu yazip
  istatistikleri kalibre etmistim. Butun o karmasa su yuzdendi:
  BatchNorm bir ornegi normalize ederken YANINDAKI orneklere bakiyor.
  Sinavda yanindakinin kagidina bakmak gibi. Egitimde sinifta 32 kisi
  var, ortalamayi hesaplayabiliyorsun; cikarimda tek bir ornek geliyor,
  yanindaki yok. O yuzden "sinifin ortalamasi genelde su kadardi" notunu
  running_mean'de saklamak zorunda kaliyorsun.

  LayerNorm bu bagi tamamen kopariyor. Ortalamayi satir boyunca, yani
  token'in KENDI 32 boyutu boyunca aliyor. Komsuya hic bakmiyor.
  Dolayisiyla:
    - tek ornek de 1000 ornek de ayni sonucu verir,
    - saklanacak bir sey yok, running_mean/running_var gereksiz,
    - egitim ve cikarim kodu ayni, model.eval() bu katman icin fark etmez.

  Dil modelinde bu ozellikle onemli, cunku cumlelerin uzunlugu degisiyor
  ve uretim sirasinda tek tek token besliyorum. BatchNorm olsa her
  uretim adiminda "batch ortalamasi" diye bir sey uydurmam gerekirdi.

  Bir de Hafta 6'da dustugum tuzak var: 3 boyutlu (B,T,C) girdide
  mean(0) yazmistim, dogrusu mean((0,1)) idi; batch ekseniyle zaman
  ekseninin ikisini de kapsamasi gerekiyordu. LayerNorm'da boyle bir
  tuzak yok, cunku her zaman en son eksende (C) calisiyor - T ve B kac
  olursa olsun fark etmez.
""")

# uretim ornegi
gg = torch.Generator().manual_seed(42)
print("  LayerNorm'lu modelin urettigi 400 karakter:")
ornek = decode(uret(ln_model, torch.zeros((1, 1), dtype=torch.long), 400,
                    g=gg)[0].tolist())
print(ornek)
print()

sonuc['_ornek_shakespeare_kucuk'] = ornek
json.dump(sonuc, open('sonuclar.json', 'w'), indent=1, ensure_ascii=False)
print("  -> sonuclar.json guncellendi")
