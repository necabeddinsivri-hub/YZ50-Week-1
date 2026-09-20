# Hafta 5 - Deney Notu

## Ne yaptim

Gecen haftaki modelin butun gradient'lerini elle hesapladim.
loss.backward() satirinin arkasinda tam olarak ne oldugunu gordum.

## Yontem

Modeli 20 kucuk adima bolduk. Normalde tek satirda yazdigimiz seyleri
ayri ayri degiskenlere ayirdik:

    logits -> logit_maxes -> norm_logits -> counts -> counts_sum
           -> counts_sum_inv -> probs -> logprobs -> loss

Sonra sondan basa dogru her birinin turevini yazdim ve cmp fonksiyonuyla
PyTorch'un sonucuyla karsilastirdim.

## Gorev 2 sonuclari

26 gradient dogrulandi. 11 tanesi EXACT (tam sayisal esitlik), 15 tanesi
approx (fark 1e-9 civari).

approx cikanlarin hepsi BatchNorm zincirinde ya da ondan sonraki
adimlarda. Sebep hata degil: PyTorch islemleri farkli sirada yapiyor ve
float aritmetiginde a+b+c ile c+b+a mikroskobik farkli cikabiliyor.

## Ogrendigim uc kural

### 1. Broadcasting ileri geciste yayar, geri geciste toplar

    probs = counts * counts_sum_inv
    # counts:         (32, 27)
    # counts_sum_inv: (32,  1)   <- yayiliyor

Ileri geciste counts_sum_inv'in tek sutunu 27 sutuna kopyalaniyor.
Geri geciste o 27 sutundan gelen gradient'lerin hepsi TOPLANMALI:

    dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)

Bunu ilk yazdigimda sum'i unuttum ve sekil hatasi aldim. Iyi ki hata
verdi; sessizce gecseydi bulamazdim.

### 2. Bir degisken iki yerde kullanildiysa gradient'ler toplanir

counts hem probs'ta hem counts_sum'da kullaniliyor:

    dcounts = counts_sum_inv * dprobs              # birinci katki
    dcounts += torch.ones_like(counts) * dcounts_sum  # ikinci katki

Bu Hafta 2'de micrograd'da `self.grad +=` yazmamla birebir ayni kural.
O zaman skalerdeydi, simdi tensor'da.

Ayni durum bndiff'te (bnraw ve bndiff2), hprebn'de (bndiff ve bnmeani)
ve logits'te (norm_logits ve logit_maxes) da var.

### 3. Toplam ve maksimum farkli davranir

    counts_sum = counts.sum(1)
    # toplamda gradient BUTUN elemanlara esit dagilir

    logit_maxes = logits.max(1).values
    # maksimumda gradient SADECE kazanan elemana gider

Ikincisi icin one_hot kullandim:

    dlogits += F.one_hot(logits.max(1).indices, 27) * dlogit_maxes

## En cok zorlandigim turev: bndiff

bndiff iki yerde kullaniliyor ve ikinci katki gec geliyor:

1. bnraw = bndiff * bnvar_inv          -> birinci katki, erken
2. bndiff2 = bndiff ** 2               -> ikinci katki, dort adim sonra

Once sadece birinci katkiyi yazdim, cmp "HATA" dedi. Fark buyuktu
(1e-2 seviyesinde), yani yuvarlama degil gercek bir eksiklik.

Grafi kagitta cizdim ve bndiff'ten cikan IKI ok oldugunu gordum.
Ikinci katkiyi ekleyince duzeldi. Ders: bir degiskenden kac ok
cikiyorsa o kadar katki var.

## Gorev 3 (ek) - tek satira indirmek

### Cross entropy

8 adimlik zincir suna iniyor:

    dlogits = (softmax(logits) - one_hot(Y)) / n

maks fark: 1.86e-09

Anlami cok hosuma gitti: gradient = (modelin tahmini) - (gercek cevap).
Model dogru harfe %100 dediyse gradient sifir, hic duzeltme yok. Yanlis
harfe yuksek olasilik verdiyse o harfin gradient'i pozitif, yani
"bunu azalt" diyor.

### BatchNorm

7 adimlik zincir suna iniyor:

    dhprebn = bnvar_inv/n * (n*dbnraw - dbnraw.sum(0)
                             - n/(n-1) * bnraw * (dbnraw*bnraw).sum(0))

maks fark: 4.66e-10

Uc terim var:
- n*dbnraw : dogrudan etki
- -dbnraw.sum(0) : ortalamayi degistirdigin icin geri odeme
- ucuncu terim : varyansi degistirdigin icin geri odeme

Bu ikinci ve ucuncu terimler BatchNorm'un tuhaf yanini aciklıyor:
bir ornegi oynattiginda ayni batch'teki DIGER orneklerin ciktisi da
degisiyor, cunku ortalama ve varyans degisiyor. Gecen hafta bunu
"tuhaf ama ise yariyor" diye gecmistim, bu hafta matematigini gordum.

### loss.backward() olmadan egitim

20.000 adim, hicbir yerde loss.backward() yok. Butun gradient'ler elle.

| adim | 0 | 4000 | 8000 | 16000 | 19999 |
|---|---|---|---|---|---|
| loss | 3.4437 | 2.9163 | 2.0499 | 1.9306 | 2.2072 |

dev loss = 2.2287

Gecen haftaki autograd'li versiyon 2.1486'ydi. Aradaki fark modelin
biraz kucuk olmasindan (GIZLI=64, gecen hafta 200'du) ve batch
boyutunun 32 olmasindan. Onemli olan calistigini gostermek.

Urettigi isimler: carmah, amille, khirmin, reety, salaysa, ner, kiah

## Zorlandigim yerler

- bndiff'in ikinci katkisini unutmak (yukarida anlattim)
- dC'yi yazarken: ayni harf batch icinde birden fazla gectiyse
  gradient'ler toplanmali. Dongu ile yazdim, vektorlestirmedim.
- BatchNorm tek satir turevini kagitta cikarmak uzun surdu. Ucuncu
  terimin n/(n-1) carpani Bessel duzeltmesinden geliyor, onu
  atlayinca kucuk ama sistematik bir fark cikiyordu.

## Acik sorular

1. dC'yi dongu yerine index_add_ ile yazsam ne kadar hizlanir?
2. Modern kutuphaneler bu tek satir formulleri nasil buluyor,
   sembolik turev mi kullaniyorlar?
3. Cross entropy'nin sadelesmesi tesaduf mu, yoksa softmax ve log'un
   bilincli olarak birbirini goturecek sekilde mi secildigi?
