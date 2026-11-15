# Hafta 6 - Deney Notu

Gecen haftanin geri bildirimi: kod dogruydu, en zayif taraf aciklamalardi.
Bu hafta her gorevde "ne yaptim" ile birlikte "neden oyle" sorusunu da
kendi cumlelerimle cevaplamaya calistim.

---

## Gorev 1 - Kodu siniflara toplamak

Hafta 4'te model bes ayri tensor'du (C, W1, b1, W2, b2) ve egitim
dongusu hepsinin adini biliyordu. Bu hafta her katmani bir sinif yaptim.
Hepsi ayni iki seye cevap veriyor:

    katman(x)            -> ciktiyi hesapla
    katman.parameters()  -> ogrenilecek tensor'lari ver

Sequential bu katmanlari sirayla cagiriyor ve parametrelerini tek listede
topluyor. Egitim dongusu artik sadece `model(x)` ve `model.parameters()`
biliyor. Katman ekleyip cikarmak icin donguye dokunmuyorum.

Kontrol: ayni model, ayni ayarlar. Hafta 4'te dev loss 2.1486, bu hafta
siniflarla 2.1508. Fark sadece rastgele baslangictan. Yani yeniden
duzenleme hicbir seyi bozmadi.

**BatchNorm'un egitim/tahmin kipi.** Sinif olarak yazinca bir sey netlesti:
BatchNorm'un iki davranisi var ve hangisinde oldugunu bilmesi gerekiyor.
Bunu `training` bayragiyla cozdum; Sequential'a da bayragi butun
katmanlara ileten bir `egitim_kipi()` metodu ekledim. torch.nn'deki
`model.train()` / `model.eval()` tam olarak bu.

**Loss egrisi.** Ham minibatch loss'u cok gurultulu, egri okunmuyor.
1000 adimlik bloklarin ortalamasini alinca (`view(-1, 1000).mean(1)`)
temiz bir egri cikti. Learning rate'in 0.1'den 0.01'e dustugu an egride
belirgin bir basamak olarak gorunuyor (loss_egrisi.png).

---

## Gorev 2 - Baglami 8'e cikarmak (taban)

| | parametre | dev loss |
|---|---|---|
| baglam 3 | 12.097 | 2.1508 |
| baglam 8 | 22.097 | 2.0787 |
| fark | +10.000 (1.83 kat) | -0.0721 |

Parametre artisinin tamami ilk Linear katmandan geliyor: 3x10=30 giris
yerine 8x10=80 giris, 200 neuron'a baglaniyor. 6.000 agirlik 16.000 oldu.

Kazanc gercek: 0.07 dusus. Ama yontem kaba. Sekiz harfin hepsi ilk
katmanda tek seferde ezilip 200 sayiya sikistiriliyor. Model "bu iki harf
yan yana" bilgisini ayri bir adimda ogrenemiyor.

---

## Gorev 3 - WaveNet ve sekiller

Model:

    Embedding -> [FlattenConsecutive(2) -> Linear -> BatchNorm -> Tanh] x 3 -> Linear

4 orneklik batch ile her katmanin cikti sekli:

| katman | sekil | neden |
|---|---|---|
| Embedding | (4, 8, 10) | 4 isim, 8 harf, her harf 10 sayi |
| Flatten 1 | (4, 4, 20) | komsu harfler ikiser birlesti: 8 harf -> 4 cift |
| Linear 1 | (4, 4, 68) | her cift 20 -> 68. AYNI agirlik 4 cifte de uygulaniyor |
| Flatten 2 | (4, 2, 136) | ciftler birlesti: 4 cift -> 2 dortlu |
| Linear 2 | (4, 2, 68) | her dortlu 136 -> 68 |
| Flatten 3 | (4, 136) | dortluler birlesti, orta eksen 1 oldugu icin atildi |
| Linear 3 | (4, 68) | 8 harfin tamami tek vektorde |
| Linear 4 | (4, 27) | cikis: 27 harf icin skor |

**Kendi cumlelerimle.** Duz modelde 8 harf ilk katmanda birden
karistiriliyordu. WaveNet'te her katman sadece IKI komsu seyi birlestiriyor.
Ilk katman harf ciftlerine ('so', 'ph', 'ia') bakiyor, ikinci katman bu
ciftlerden dortluler kuruyor, ucuncu katman butun baglami goruyor. Bilgi bir
agac gibi asagidan yukariya cikiyor, her seviyede biraz daha ozetleniyor.

**Anlamam en uzun suren sey:** Linear katman 3 boyutlu tensor'a
uygulandiginda ne oluyor? PyTorch'ta `x @ W` sadece son eksen uzerinden
calisiyor, ondeki eksenler "toplu islem" gibi davraniyor. Yani (4, 4, 20)
seklindeki tensor 20 x 68'lik tek bir W ile carpildiginda, 16 ayri cift
(4 isim x 4 cift) AYNI W ile ayri ayri donusuyor. Bu parametre paylasimi:
"so" ciftini isleyen agirlik ile "ia" ciftini isleyen agirlik ayni.

**Parametre sayisi:** 68 gizli boyutla 22.397 parametre, duz baglam-8
modelle (22.097) neredeyse esit. Bilincli olarak esitledim ki iki mimariyi
ayni butce ile karsilastirabileyim.

---

## Gorev 4 - BatchNorm hatasi

Sekilleri yazdirirken BatchNorm'un calisan ortalamasinin sekline de
baktim:

    katman 3:  running_mean = (1, 4, 68)    <- beklenen (1, 1, 68)
    katman 7:  running_mean = (1, 2, 68)
    katman 11: running_mean = (1, 68)       <- bu dogru

**Hata ne.** Hafta 4'te BatchNorm'a hep 2 boyutlu tensor giriyordu:
(batch, kanal). `mean(0)` her kanal icin batch uzerinden ortalama aliyordu.
Dogruydu. WaveNet'te tensor 3 boyutlu oldu: (batch, pozisyon, kanal).
Ayni `mean(0)` simdi her POZISYON icin ayri ortalama aliyor. Yani "birinci
cift", "ikinci cift", "ucuncu cift", "dorduncu cift" ayri ayri
normalize ediliyor.

**Neden yanlis.** Bu dort pozisyon ayni Linear katmandan, ayni
agirliklarla geciyor. 68 kanalin her biri, hangi pozisyonda olursa olsun
ayni seyi olcuyor. Dolayisiyla her kanalin tek bir ortalamasi ve varyansi
olmali. Dogrusu `mean((0, 1))`: hem batch hem pozisyon uzerinden. Boylece
her kanal icin 32 x 4 = 128 ornek uzerinden tek bir istatistik cikiyor.

**Neden sessiz.** Kod hata vermiyor. (1,4,68) sekilli ortalama, (32,4,68)
sekilli tensor'dan broadcasting ile sorunsuz cikariliyor. Model
egitiliyor, loss dusuyor. Tek iz running_mean'in sekli. Yazdirmasaydim
hic gormezdim. Hafta 3'teki keepdim tuzagi ve Hafta 2'deki zero-grad
bug'i ile ayni aileden.

**Sonuc:**

| | train | dev |
|---|---|---|
| hatali, mean(0) | 2.0536 | 2.0890 |
| duzeltilmis, mean((0,1)) | 2.0371 | 2.0779 |
| iyilesme | | 0.0111 |

Kucuk ama olculebilir. Neden kucuk: hatali versiyon da normalize
ediyordu, sadece istatistigi 4 kat az ornekten tahmin ediyordu (32 yerine
128). Duzeltilmis versiyonun tahmini daha istikrarli.

---

## Gorev 5 - Buyutme ve karsilastirma

Embedding 10 -> 24, 60.000 adim.

| model | parametre | train | dev loss |
|---|---|---|---|
| baglam 3, duz MLP | 20.875 | 2.0626 | 2.1030 |
| baglam 8, duz MLP | 44.875 | 1.9157 | 2.0101 |
| baglam 8, WaveNet | 76.579 | 1.8838 | 1.9942 |

- baglami 3'ten 8'e cikarmak: -0.0928
- ayni 8 harfi hiyerarsik birlestirmek: -0.0159
- toplam: -0.1088

Karpathy'nin sonucu 1.993 (200.000 adim). Ben 60.000 adimda 1.9942'ye
geldim.

**Durust yorum.** Kazancin buyuk kismi baglami buyutmekten geliyor,
hiyerarsiden degil. Uc gozlem:

1. Bu tabloda WaveNet'in parametresi duz modelden fazla (76k vs 45k).
   Yani 0.016'lik kazancin bir kismi sadece daha buyuk model olmasindan.
2. Gorev 3-4'te parametreleri ESITLEMISTIM (22.397 vs 22.097). Orada
   duzeltilmis WaveNet 2.0779, duz model 2.0787. Neredeyse ayni.
3. WaveNet'te train-dev acigi 0.110, duz modelde 0.094. Yani WaveNet
   biraz daha fazla ezberliyor.

Sonucum: 8 harflik baglamda, bu kucuk veri setinde hiyerarsinin avantaji
sinirli. Karpathy de videoda benzer bir sey soyluyor: mimarinin gercek
degeri daha uzun baglamlarda ve dilated convolution ile verimlilikte
ortaya cikiyor (Gorev 7'de bunu gosterdim).

---

## Gorev 6 - Turkce

Hafta 4'le adil karsilastirma icin ayni alt kume (her 4. isim, 59.024
isim) ve ayni karistirma tohumu.

| model | parametre | dev loss |
|---|---|---|
| Hafta 3 bigram | 729 | 2.5830 |
| Hafta 4 MLP (baglam 3, emb 10) | 12.530 | 2.3183 |
| Hafta 6 baglam 3 MLP (emb 24) | 21.550 | 2.2974 |
| Hafta 6 baglam 8 WaveNet | 77.038 | 2.1829 |

**Baglami 8'e cikarmak Turkce'de ne kazandirdi:** ayni ayarlarda 0.1145.
Ingilizce'deki toplam kazanctan (0.1088) biraz fazla.

Neden olabilir: Turkce isimler daha uzun ve eklemeli yapida
(-gul, -nur, -han, -can, -ay gibi ekler). "gul" ekini dogru yere koymak
icin ismin basini hatirlamak gerekiyor. Bunu dogrudan test etmedim, bir
hipotez.

Urettigi isimler:
ziyangul, gulfizar, gunmaz, eymet, gunayse, sememan, sindavegul, hurzer

Gulfizar gercek bir Turkce isim. Ziyangul ve sindavegul "-gul" ekini
dogru yerde kullaniyor. Baglam 3 modelinin urettikleri daha kisa ve
daha kopuk: ziye, gul, bezer, pire, vis.

---

## Gorev 7 (ek, secenek a) - Conv1d

**Fikir.** FlattenConsecutive(2) + Linear aslinda kernel boyutu 2 olan bir
konvolusyon. Egitilmis WaveNet'in agirliklarini Conv1d'ye kopyaladim:

    Linear agirligi (2*c_in, c_out)  ->  Conv1d agirligi (c_out, c_in, 2)
    [x_sol, x_sag] @ W = x_sol @ W[:c_in] + x_sag @ W[c_in:]

**1) stride=2:** kendi WaveNet'im ile Conv1d arasinda maksimum fark
1.55e-06. Dev loss ikisinde de 2.0998.

**2) dilated (1, 2, 4), stride=1:** 'sophia' isminin 7 pozisyonunun
tahminini tek geciste hesapladim. Kendi modelimi 7 kez ayri calistirmakla
maksimum fark 1.43e-06.

Islem sayisi: tek tek 49 Linear uygulamasi, dilated 31. Komsu pencereler
ayni ara sonuclari (ornegin 'ph' ciftinin temsilini) tekrar tekrar
hesapliyordu; dilated konvolusyon her birini bir kez hesaplayip
paylasiyor.

---

## Zorlandigim yerler

- 3 boyutlu tensor'a Linear uygulamanin ne demek oldugu. Son eksen
  uzerinden calistigini ve ondeki eksenlerin toplu islem gibi
  davrandigini yazdirarak anladim.
- FlattenConsecutive'de squeeze: son katmanda orta eksen 1 oluyor,
  atmazsam son Linear (B, 1, 68) aliyor ve cikis (B, 1, 27) oluyor.
  cross_entropy bunu kabul etmiyor.
- Turkce egitimde bellek: 361 bin ornegin hepsini tek seferde
  degerlendirmeye calisinca program kapandi. Degerlendirmeyi 20 binlik
  parcalara boldum.

## Acik sorular

1. Esit parametre butcesinde WaveNet'in duz modeli gecmesi icin baglam
   ne kadar uzun olmali?
2. Dilated convolution'da ilk katmandaki nokta dolgusu (padding) sonuclari
   nasil etkiliyor?
3. Turkce'de ek yapisi hipotezimi nasil test edebilirim? Belki sadece
   "-gul" ile biten isimlerde loss'a bakarak.
