# Hafta 4 - Deney Notu

## Ne yaptim

Ilk gercek neural network dil modelimi kurdum. Gecen hafta model tek harfe
bakiyordu ve bir tabloyu dolduruyordu. Bu hafta uc harfe bakiyor ve tablo
yerine ogrenilen vektorler kullaniyor.

## Deney 1 - Baglam penceresi ve embedding

Veri seti: her ornek 3 harf indeksi (X) ve hedef harf (Y).
32.033 isimden 228.146 ornek cikti.

Parametre karsilastirmasi ilginc:

| | temsil | parametre |
|---|---|---|
| Hafta 3 | one-hot 27'lik vektor | 27x27 = 729 |
| Hafta 4 | ogrenilen 2'lik vektor | 27x2 = 54 |

Yani daha az parametreyle daha fazla is yapiyoruz, cunku embedding
OGRENILIYOR. Benzer harfler birbirine yakin dusuyor (Deney 4'te gosterdim).

## Deney 2 - Elle loss vs F.cross_entropy

Ikisi ayni sonucu veriyor (fark 3.8e-06). Ama cross_entropy'nin bir
avantaji var: sayisal guvenlik.

Test ettim: logits = [1, 2, 3, 100] verdigimde
- elle hesap: exp(100) = inf -> loss = nan
- F.cross_entropy: 0.000000, sorunsuz

Sebep: cross_entropy icten en buyuk logit'i cikarip normalize ediyor.
Bunu bilmesem uzun egitimlerde neden nan aldigimi anlamazdim.

## Deney 3 - Egitim

Once tek bir 32'lik minibatch'i overfit ettim: loss 18.48 -> 0.27.
Model saglam kurulmus demek. Bu testi ogrenmek faydali oldu; buyuk
egitime baslamadan once kodun dogru oldugunu anlamanin en hizli yolu.

Learning rate taramasi:

| lr | train loss |
|---|---|
| 0.001 | 8.3097 |
| 0.01 | 2.9625 |
| 0.05 | 2.6132 |
| 0.1 | 2.6114 (en iyi) |
| 0.3 | 2.6548 |
| 1.0 | 3.1089 |
| 3.0 | 15.9844 |

Asil egitim: 30.000 adim, ilk 20.000'de lr=0.1, sonra lr=0.01.

train loss 2.3629, dev loss 2.3623. Ikisi neredeyse esit -> model
ezberlemiyor, demek ki daha buyuk model denenebilir.

## Deney 4 - Model buyutme

| emb | gizli | parametre | train | dev |
|---|---|---|---|---|
| 2 | 100 | 3.481 | 2.2665 | 2.2668 |
| 2 | 300 | 10.281 | 2.2451 | 2.2477 |
| 10 | 200 | 11.897 | 2.0964 | 2.1287 |
| 10 | 300 | 17.697 | 2.0775 | 2.1167 |

Embedding boyutunu 2'den 10'a cikarmak, gizli katmani buyutmekten daha
cok fayda sagladi. Mantikli: 2 sayiyla 27 harfi ayirt etmek zor.

Son satirda train (2.0775) ile dev (2.1167) arasinda kucuk bir aralik
acildi -> model hafiften ezberlemeye basliyor.

### Embedding gorsellestirmesi

2 boyutlu embedding'i cizdirdim (embedding.png ve terminalde metin harita).

Sesli harflerin koordinatlari:
a (-0.48, -3.84), e (-1.73, -2.18), i (-2.12, -1.05),
o (-2.79, -2.60), u (-1.89, -1.45)

Sayisal kontrol:
- sesli-sesli ortalama uzaklik: 1.741
- sesli-sessiz ortalama uzaklik: 4.168

Yani model, kimse soylemeden sesli harfleri bir kumede topladi. Bu
haftanin en etkileyici sonucu bence. Hicbir yerde "a, e, i sesli harftir"
diye bir bilgi vermedik; model bunu sadece "bu harfler benzer yerlerde
geciyor" gozleminden cikardi.

### Uretilen isimler

MLP: careah, ami, hari, kimrix, thay, sranden, den, nellara, kaleig, jor
Hafta 3 bigram: cexze, konimittain, llayn, ka, da

Fark cok net. MLP'ninkiler gercek isim gibi duruyor.

## Deney 5 - Init ve tanh doymasi

### Baslangic loss'u

Model hicbir sey bilmiyorken beklenen loss: -log(1/27) = 3.2958.
Kotu init ile gercek: 25.42. Neredeyse 8 kat fazla.

Sebep: son katmanin ciktilari rastgele buyuk. Ilk 5 logit: [20.48, -7.18,
2.48, 7.37, -8.29]. Model "z harfi kesinlikle gelir" gibi guclu ama
tamamen yanlis iddialarda bulunuyor. Ilk adimlar bu iddialari sondurmekle
geciyor, gercek ogrenme sonra basliyor.

Son katmani ~0 ile baslatinca: loss 3.2925. Beklenen degere birebir yakin.

### tanh doymasi

Doymus neuron (|h| > 0.99) orani:

| init | doyma orani |
|---|---|
| W1 olcek = 1.0 | %61.6 |
| Kaiming (olcek = 0.30) | %10.0 |

Neden onemli: tanh'in turevi 1 - h^2. h = 0.99 ise turev 0.02, yani
gradient neredeyse hic gecmiyor. Neuron'un %61'i ogrenmiyor demek.

Bu, Hafta 2'de kendi micrograd'imda yazdigim `1 - t**2` satirinin ta
kendisi. O zaman sadece bir formuldu, simdi ne ise yaradigini goruyorum.

### Egitim sonuclari

| init | ilk loss | train | dev |
|---|---|---|---|
| kotu (hepsi rastgele) | 26.36 | 2.2601 | 2.2657 |
| son katman ~0 | 3.32 | 2.1597 | 2.1889 |
| Kaiming + son katman | 3.32 | 2.1218 | 2.1486 |

Kaiming init dev loss'u 2.2657'den 2.1486'ya indirdi. Tek gizli katmanli
bir agda bile fark var; derin aglarda cok daha kritik olacagini tahmin
ediyorum.

## Deney 6 - BatchNorm

Egitim oncesi doymus neuron:
- BatchNorm yok: %10.0
- BatchNorm var: %0.8

20.000 adim sonrasi:

| model | train | dev | doyma |
|---|---|---|---|
| BatchNorm'suz | 2.1219 | 2.1486 | %19.4 |
| BatchNorm'lu | 2.1494 | 2.1646 | %3.1 |

Dev loss'ta BatchNorm 0.016 GERIDE kaldi. Beklentimin tersi cikti.

Yorumum: bu ag cok sig (tek gizli katman) ve Kaiming init zaten isi
cozuyor. BatchNorm burada ekstra bir sey kazandirmiyor, hafif gurultusu
ise kucuk bir maliyet yaratiyor. Asil degeri derin aglarda olmali:
10 katmanda her katmanin dagilimini elle ayarlamak imkansiz.

Bir de sunu ogrendim: BatchNorm varsa b1 bias'ina gerek yok, cunku
ortalama cikarma islemi bias'i zaten siliyor. Kodda kaldirdim.

## Deney 7 - Turkce

29 harf + nokta = 30 sembol. Hiz icin listenin 1/4'unu kullandim
(59.024 isim, 361.709 ornek).

| model | dev loss |
|---|---|
| Hafta 3 bigram sayim | 2.5830 |
| Hafta 3 bigram sinir agi | 2.6118 |
| Hafta 4 MLP | 2.3183 |

Uretilen isimler: ziye, bizer, semim, pire, gunayse, sen, sal, baz,
belli, elayiz, sadtun, azıklahat, gulfeyiletdin

Bazilari gercek Turkce isim gibi (ziye, pire, sen, sal, belli). Hafta 3'te
uretilenler (ar, zi, nurd, zurancal) ile karsilastirinca fark cok belirgin.

Hala uzun ve sacma olanlar var (gulfeyiletdin), yani 3 harf baglam da
her seyi cozmuyor.

## Deney 8 (ek) - BatchNorm'u katlama

Karpathy'nin Part 3 E02 egzersizi.

Egitim bittikten sonra BatchNorm'u W1 ve b1'e katladim:
- W1_yeni = W1 * (kazanc / std)
- b1_yeni = kaydirma - ort * (kazanc / std)

Sonuc:
- BatchNorm'lu dev loss: 2.181981
- Katlanmis dev loss: 2.181981
- logits'ler arasi maksimum fark: 1.91e-06

Forward pass birebir ayni kaldi ama BatchNorm katmani tahmin zamaninda
tamamen ortadan kalkti.

Mantigi basit: tahmin zamani BatchNorm sabit bir dogrusal donusum, ve
iki dogrusal donusum tek bir dogrusal donusumde birlestirilebilir.

## Zorlandigim yerler

- `emb.view(-1, 6)` satirini anlamak zaman aldi. (N, 3, 2) seklindeki
  tensor'u (N, 6) yapiyor, yani 3 harfin vektorlerini yan yana diziyor.
- BatchNorm'da egitim ve tahmin zamaninda farkli sey yapilmasi ilk basta
  tuhaf geldi. Egitimde batch istatistigi, tahminde calisan ortalama.
- BatchNorm'un dev loss'u iyilestirmemesi kafami karistirdi. Once bug
  sandim, sonra agin sig olmasindan kaynaklandigini dusundum.

## Acik sorular

1. BatchNorm derin aglarda ne kadar fark yaratir? Bu ag cok sig kaldi.
2. Embedding boyutu 10'dan 20'ye ciksa ne olur? Denemedim.
3. Baglam penceresi 3 yerine 5 olsa? Model daha iyi olur mu yoksa
   veri mi yetmez?
