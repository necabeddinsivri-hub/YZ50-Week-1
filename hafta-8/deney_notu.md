# Deney notu - Hafta 8

Ne denedim, ne cikti, neyi bekliyordum. Siraya gore.

## 1. Multi-head

**Bekledigim:** tek head'i 4'e bolmek kucuk bir kazanc getirir, cunku
parametre neredeyse hic artmiyor.

**Cikan:** val 2.3998 -> 2.2604, yani
0.1394 kazanc. Parametre sadece
1.056 artti.

**Sasirtan:** bu kazanc, gecen haftanin "bigram -> tek head" kazancinin
(0.0771) 1.81 kati.
Parametre eklemeden gelen bir kazanc.

**Kontrol ettim:** dort head ayni seyi mi ogrendi? Son satirlarin ortalama
mutlak farklari 0.0413-0.1655 arasinda. Sifir olsaydi bolme bosa gitmis
olurdu. Head 2 digerlerinden farkli: kendine degil bir onceki BOSLUGA
bakiyor. Head 3 ise neredeyse tamamen kendine bakiyor (0.998) - attention'i
pratikte kapatmis.

## 2. FeedForward

**Bekledigim:** iyilesme, cunku Hafta 7'de attention ciktisi dogrudan
logit'e gidiyordu - arada hic hesap yoktu.

**Cikan:** val 2.2604 -> 2.1818 (0.0786).
Parametre 8.609 -> 16.961.

## 3. Uc blok, residual yok  <- bu haftanin en ogretici deneyi

**Bekledigim:** bir blok ise yaradiysa uc blok daha iyi olmali.

**Cikan:** val 2.1818 -> 2.2811. Yani
**0.0992 KOTULESTI.** Parametre
2.47 katina cikti, sonuc daha kotu.

**Neden:** gradyan embedding'e varmak icin butun katmanlardan geri gecmek
zorunda; her gecis zincir kuralina bir carpma ekliyor. Hafta 1'de
micrograd'i elle yazarken bunu tek bir zincirde gormustum.

**Not:** bu satiri tablodan silmedim. Teshis satiri olarak biraktim cunku
residual'in neden gerekli oldugunu baska hicbir sey bu kadar iyi
anlatmiyor.

## 4. Residual

**Cikan:** val 2.2811 -> 2.0700. Parametre AYNI
(41.921). Tek fark: `x = sa(x)` yerine `x = x + sa(x)`.

Feedforward'lu tek bloga gore de 0.1118 iyilesme var -
yani derinlik artik gercekten kazanc getiriyor.

## 5. LayerNorm

**Cikan:** val 2.0700 -> 2.0520 (0.0180).

**Bekledigimden kucuk** ve bu beni sasirtmadi: 3 blok sig bir ag,
LayerNorm'un asil faydasi derinlestikce cikiyor. Gorev 3'te 4 bloga
ciktigimda bu parca olmasa egitim kararsizlasirdi.

## 6. LayerNorm vs BatchNorm1d - ve bir hata

Kendi yazdigim `LayerNorm1d` ile `nn.LayerNorm` arasinda **1.66e-01** fark
cikti. Sifira yakin olmasi gerekiyordu.

**Sebep:** `x.var(1)` yazmistim. PyTorch'ta `var()` varsayilan olarak n-1'e
boler (Bessel duzeltmesi - Hafta 6), `nn.LayerNorm` ise n'e boluyor.
Hatali surumun satir std'si 0.9129 = sqrt(5/6) cikiyordu; tam ipucu buydu.

`unbiased=False` ekledim, fark **3.58e-07** oldu.

**Ogrendigim:** Hafta 6'da n-1 kullanmam dogruydu, cunku batch butun verinin
bir ORNEGIYDI. LayerNorm'da normalize ettigim sey tek bir token'in butun
ozellikleri - kestirim yok, elimdeki veri evrenin kendisi. Ayni formul,
farkli istatistiksel durum. Hafta 6'da ezberledigimi sandigim bir
ayrintinin ne ise yaradigini burada anladim.

## 7. Olcekleme

**Yaklasim:** config'i tahmin etmek yerine olctum. Bes aday:

| Config | n_embd | head | blok | block | Parametre | ms/adim | 3000 adim |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (Hafta 7) | 32 | 4 | 3 | 8 | 42.369 | 24.9 | 1.2 dk |
| B | 64 | 4 | 3 | 16 | 158.913 | 39.6 | 2.0 dk |
| C | 96 | 4 | 3 | 32 | 350.465 | 75.0 | 3.7 dk |
| **D (sectigim)** | **128** | **4** | **4** | **64** | **816.705** | **203.6** | **10.2 dk** |
| E (Karpathy, GPU) | 384 | 6 | 6 | 256 | 10.788.929 | 6108.1 | **305.4 dk** |

E, yani videodaki config, bu makinede 5 saat surerdi.

## 8. Dropout - beklentimi vermedi

| | train | val | val-train |
|---|---:|---:|---:|
| 0.0 | 1.4055 | 1.6113 | 0.2058 |
| 0.1 | 1.4552 | 1.6476 | 0.1924 |

**Bekledigim:** train yukselir, val duser.

**Cikan:** train yukseldi, val DA yukseldi. Tek olumlu isaret asiri uyum
farkinin daralmasi (0.2058 -> 0.1924).

**Yorumum:** dropout yakinsamayi yavaslatiyor ve faydasi model asiri uyuma
girdikten sonra cikiyor. 4000 adimda dropout'suz model henuz asiri uyumun
derinine inmemis, yani duzeltilecek cok bir sey yok. Daha uzun egitimde
tablonun tersine donmesini beklerdim - olcemedim.

Hafta 4'te BatchNorm'un dev loss'u biraz kotulestirdigini yazdigimdaki
durumun aynisi. Teslim ettigim config dropout'lu olan (gorev oyle
istiyor) ama hangi sayinin daha iyi oldugunu saklamiyorum.

## 9. Turkce

Kendim ~1 MB metin derledim (`turkce_derle.py`). 87 farkli karakter,
Shakespeare'de 65.

val loss 1.5841 - Shakespeare'in 1.6476'inden DAHA DUSUK.
Ama bu sayiyi yan yana koymak yanlis. Egitimsiz taban log(V):
Shakespeare 4.1744, Turkce 4.4659. Turkce model daha baslamadan
0.2915 handikapli; ustune dilin eklemeli yapisi ve metin turu farki
biniyor.

**Ikinci surpriz:** Turkce modelde val < train (1.5841 <
1.6233). Sebebi veri bolmem: korpusu %90/%10 SIRAYLA boldum ve
son %10 daha homojen cikti. Yeniden yapsam karistirip bolerdim. Hafta 6'da
"bolme bicimi olcumu etkiler" diye not almistim; pratik sonucunu burada
gordum.

## 10. Ek calisma

Tek matmul'lu multi-head: dongulu surumun agirliklarini kopyalayip ayni
girdiyi ikisinden gecirdim. En buyuk fark **0.000e+00** - bit duzeyinde
ayni. Bunu beklemiyordum; sebebi head'leri view+transpose ile bolunce
toplama sirasinin hic degismemesi. Hafta 5'in "EXACT mi approx mi"
olcutuyle EXACT. Hiz: 1.25x-1.85x.

Sekil 1 eslesmesi: sag kulenin cross-attention cikarilmis hali = benim
modelim. Encoder yok cunku okunacak kaynak dizi yok; cross-attention yok
cunku key/value'nun gelecegi yer yok.

## Acik kalanlar

- Head 3'un neredeyse tamamen kendine bakmasi kayip mi, bilincli bypass mi?
- Dropout daha uzun egitimde kazanca donuyor mu? Olcmek isterdim.
- Pre-norm ile post-norm farkini okudum, kendim olcmedim.
- Turkce korpusu karistirarak bolup val olcumunu duzeltmek.
- Karakter yerine BPE kullansam Turkce'nin eklemeli yapisinda ne degisirdi?
