# Hafta 3 - Deney Notu

## Ne yaptim

Ilk dil modelimi kurdum. Bigram karakter modeli: bir harfe bakip sonraki
harfi tahmin ediyor. Ayni modeli iki farkli yoldan kurdum ve ikisinin
ayni sonuca ciktigini gosterdim.

## Deney 1 - Sayim modeli (Ingilizce)

32.033 isim, 228.146 bigram, 27x27 tablo.

En cok gecen bigram'lar:

| bigram | sayi |
|---|---|
| n -> . | 6763 |
| a -> . | 6640 |
| a -> n | 5438 |
| . -> a | 4410 |
| e -> . | 3983 |

Ilk bakista tuhaf geldi: en cok gecen sey "n'den sonra isim biter".
Sonra mantikli geldi, cunku listedeki isimlerin cogu n veya a ile
bitiyor (ann, john, emma, anna...).

Urettigi isimler: cexze, momasurailezitynn, konimittain, llayn, ka, da

Guzel isimler degil ama tamamen rastgele bir modelin urettigiyle
karsilastirinca fark cok net:

rastgele model: kkjemkmmsidguenkbvgynywftbspmhwcivgbvtahlvsu

Yani model bir sey ogrenmis, sadece cok az sey ogrenmis. Tek bir onceki
harfe bakiyor, daha fazlasini bilmiyor.

## Deney 2 - keepdim tuzagi

Videoda uyarilan yer. Sayimlari olasiliga cevirirken:

| kod | sekil | sonuc |
|---|---|---|
| N.sum(1, keepdim=True) | (27, 1) | satirlara boler - DOGRU |
| N.sum(1) | (27,) | sutunlara boler - YANLIS |

Yanlis versiyonda satir toplami 1 yerine 7.02 cikiyor. Kod hata
vermiyor, sessizce yanlis model uretiyor. Gecen haftaki zero grad
bug'ina benziyor: calisiyor gibi gorunup yanlis sonuc veren tur.

## Deney 3 - Negative Log Likelihood

'emma' ismi uzerinde adim adim:

| bigram | olasilik | log |
|---|---|---|
| . -> e | 0.0478 | -3.0410 |
| e -> m | 0.0377 | -3.2793 |
| m -> m | 0.0253 | -3.6753 |
| m -> a | 0.3885 | -0.9454 |
| a -> . | 0.1958 | -1.6305 |

toplam log olasilik = -12.5716, bu ismin loss'u = 2.5143

Smoothing taramasi (tum veri):

| smoothing | loss |
|---|---|
| 0 | 2.4541 |
| 1 | 2.4544 |
| 10 | 2.4616 |
| 100 | 2.5371 |

Smoothing arttikca loss kotulesiyor cunku model duzlesiyor, yani
"her sey biraz mumkun" demeye basliyor.

Karsilastirma: tamamen rastgele model log(27) = 3.2958. Bizim model
2.4544. Yani belirgin sekilde iyi.

## Deney 4 - Sinir agi

Ayni modeli 27x27 agirlik matrisi, one-hot girdi, softmax ve gradient
descent ile kurdum.

| adim | loss |
|---|---|
| 0 | 3.7686 |
| 20 | 2.5823 |
| 60 | 2.5027 |
| 100 | 2.4900 |
| 199 | 2.4830 |

Sayim modeli: 2.4544. Sinir agi: 2.4830. Neredeyse ayni.

Daha carpicisi: ikisi de AYNI isimleri uretiyor (cexze, konimittain,
llayn...). Ayni tabloya iki farkli yoldan varmisiz.

## Deney 5 - Turkce

Veri: github.com/Stealeristaken/Turkce-Isimler, temizleyip 236.096 isim
biraktim. Alfabe 29 harf (c, g, i, o, s, u dahil) + nokta = 30 sembol.

| model | loss |
|---|---|
| sayim | 2.5830 |
| sinir agi | 2.6118 |

Urettigi isimler: ar, zi, yattayucaz, zurancal, nurd, usulaiy, guym, se

Bazilari gercekten Turkce isme benziyor (nurd, zi, se), bazilari sacma.
Turkce loss Ingilizce'den biraz yuksek. Iki sebebi olabilir: alfabe daha
buyuk (30 vs 27) ve liste daha karisik, icinde farkli kokenlerden cok
isim var.

Not: sinir agi kismini 47.220 isimle egittim, tamami 200 saniye
suruyordu. Sayim modelinde tam listeyi kullandim.

## Deney 6 (ek) - Trigram

Veriyi %80/%10/%10 boldum. Smoothing gucunu dev setine gore ayarladim:

| smoothing | dev loss |
|---|---|
| 0.01 | 2.2318 |
| 0.1 | 2.2225 (en iyi) |
| 1 | 2.2365 |
| 10 | 2.3637 |

Test setinde:

| model | test loss |
|---|---|
| bigram | 2.4584 |
| trigram | 2.2240 |

Trigram 0.23 daha iyi. Urettigi isimler de daha "isim gibi":
moulius, ila, luwan, samiyah, javer

Mantikli: iki harfe bakinca daha fazla baglam var.

## Zorlandigim yerler

- keepdim tuzagini once anlamadim, satir toplamlari 1 cikmayinca
  fark ettim.
- Logits, exp ve softmax'in neden "sayim tablosunun yumusak hali"
  oldugunu oturtmak zaman aldi. Su cumle yardimci oldu: logits'in
  exp'i pozitif sayi uretiyor, tipki sayim gibi; sonra satira bolunce
  olasilik oluyor.
- Turkce listeyi temizlerken buyuk I harfi sorun cikardi, Python'un
  .lower() fonksiyonu I harfini yanlis ceviriyordu, elle harita yazdim.

## Acik sorular

1. Neden loss'a W kareleri ekleyince smoothing gibi davraniyor?
2. Trigram'da tablo 27x27x27 oldu, dortlu bakarsak 531 bin hucre olacak.
   Bu nasil olceklenecek? (Hafta 5'te embedding gelecek galiba)
3. Turkce icin daha temiz bir isim listesi bulsam loss ne kadar duser?
