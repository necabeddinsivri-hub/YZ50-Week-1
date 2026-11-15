# Deney notu - Hafta 10

Ne denedim, ne bekledim, ne cikti. Bu hafta iki kere kendi olcum hatami
yakaladim; ikisi de burada.

## 1. Modeli kurmak

Hafta 8'deki GPT'den yola ciktim. Mimari olarak degistirdigim sey cok az:
isimler, q/k/v'nin tek katmanda toplanmasi, ReLU yerine GELU, lm_head'de
bias olmamasi, weight tying.

Bunlarin hicbiri "daha iyi mimari" degil. Hepsi, OpenAI'in agirliklarini
yukleyebilmek icin kodumun onlarinkiyle birebir ayni dizilmis olmasi
gerektigi icin. Mimari ayni, olcek 117 kat.

## 2. Ilk loss kontrolu  <- birinci hatam buradan cikti

**Bekledigim:** ln(50257) = 10.8249 civarinda bir sayi.

**Cikan:** 9.99. Yani 0.84 ALTINDA.

Loss'un beklenenden DUSUK cikmasi her zaman supheli. Hafta 7'de tril
maskesini kaldirinca da loss dusuyordu ama model kopya cekiyordu. O
yuzden ustune gittim.

**Buldugum:** testi soyle yazmisim:

```python
x = torch.randint(0, V, (4, 32))
_, loss = model(x, x)       # hedef olarak girdinin kendisi
```

Dil modelinde hedef, girdinin bir token KAYDIRILMIS hali olmali.

**Ama neden loss dustu?** Burasi ilginc kisim. Uc durumu ayri olctum:

| | hedef = girdi | bagimsiz rastgele | gercek metin |
|---|---:|---:|---:|
| weight tying VAR | 9.9881 | 11.0015 | 10.9600 |
| weight tying YOK | 11.0215 | 10.9278 | 11.0154 |

Sebep **weight tying**. Giris embedding'i ile cikis katmani ayni tensor.
Bir token'in logit'i = o token'in embedding vektoru ile son katmanin
ciktisinin ic carpimi. Model egitilmedigi icin ciktisi buyuk olcude girdi
embedding'inin kendisine benziyor, ve bir vektorun kendisiyle ic carpimi
baska bir vektorle ic carpimindan buyuk.

Yani egitilmemis model "bir sonraki token girdinin kendisidir" tahminine
dogal bir egilimle basliyor. Hedefi girdinin kendisi yapinca bu egilim
odullendiriliyor.

Kanit ikinci satirda: weight tying YOKken ayni test 11.02 veriyor,
etki tamamen kayboluyor.

**Iki ders:** (1) ln(V) kontrolu gercekten ise yariyor ve benim durumumda
kendi test kodumdaki hatayi soyledi. (2) Weight tying masum bir parametre
tasarrufu degil; modelin baslangictaki davranisini da degistiriyor.

## 3. Tek batch'i ezberletme

**Bekledigim:** loss sifira inmeli.

**Cikan:** 10.9600 -> 0.0008,
token dogrulugu %100.0
(300 adim, 5.7 dk).

Bu test modelin ne kadar iyi oldugunu olcmuyor. 128
token'i ezberlemek 124M parametreli bir model icin kolay bir is; loss
inmiyorsa ogrenme sorunu degil KOD sorunu var demektir. Hafta 5'te
manuel geri yayilimi PyTorch'la karsilastirmistim - mantik ayni: once
dogrulugu kanitla, sonra performansa bak.

## 4. Weight tying hesabi

| | Parametre |
|---|---:|
| tying YOK | 163.037.184 |
| tying VAR | 124.439.808 |
| kazanc | 38.597.376 |

Kazanc tam olarak embedding tablosu: 50.257 x 768. GPT-2'nin "124M"
olarak anilmasinin sebebi bu; olmasaydi 163M olacakti.

## 5. Data loader ve ilk gercek egitim

120 adim, loss 10.96 ->
5.76. Ezberleme degil - her adimda farkli
batch. Ama buyuk kismi hala "ucuz" ogrenme: model hangi token'larin SIK
gectigini ogrendi (bosluk, 'the', nokta). Hafta 7'deki bigram modelinin
yaptigi seyin aynisi.

Hiz 105 token/sn. Karpathy A100'de ~16.000
aliyor. Bir epoch bu makinede
~0.9 saat surerdi.

## 6. Baslangic degerleri  <- ikinci hatam buradan cikti

**Bekledigim:** mlp.c_proj'un std'si 0.0041
(= 0.02 / sqrt(24)).

**Cikan:** 0.0047. Yuzde
15
sapma. Bir an semanin tutmadigini sandim.

**Buldugum:** olctugum model, yukarida 120 adim
EGITILMIS olan modeldi. Agirliklar artik baslangic degerlerinde degildi.
Taze bir model kurunca sema birebir tutuyor:
0.0041 vs beklenen 0.0041.

Bu haftanin ikinci olcum hatasi ve ikisi de ayni aileden: bir sayiyi
olcerken NEYI olctugunden emin olmak gerekiyor. Hafta 6, 7, 8 ve 9'da
"olcum kosullari" diye yazdigim seyin bu haftaki hali.

## 7. Bellek sinirini bulmak

Gorev 4 ve Gorev 5a'yi ayni anda arka planda calistirdim ve Gorev 4
sessizce oldu. Sebep: makinede 7 GB RAM var, her kosu 124M parametre +
gradyan + AdamW'nin iki momenti = yaklasik 2 GB, ustune aktivasyonlar.
Ikisi birlikte sigmiyor. Tek tek calistirinca sorun yok.

Not aldim: bu olcekte artik "arka planda iki sey birden calistir"
yapilamiyor.

## 8. Agirliklari incelemek  <- ucuncu yanlis tahminim

160 tensor. Dosyada **lm_head yok** - weight tying
yuzunden tek kopya saklanmis.

wpe tablosuna bakmadan once "tablonun SONU basindan daha gurultulu
olmali, cunku metinlerin cogu 1024 token'a ulasmiyor, son pozisyonlar az
egitilmistir" diye tahmin etmistim.

**Olcum:** son bolgede komsu benzerligi 0.999,
ilk bolgede 0.987. Tam tersi.

Gercekte buldugum uc sey:

1. **Pozisyon 0 bir aykiri deger.** Normu 9.88,
   medyan 3.37 - uc kat uzun. 0 ile 1 arasi
   benzerlik 0.524, 7 ile 8 arasi
   0.998. Ilk birkac pozisyon keskin ayrilmis.
2. **Cozunurluk basta yuksek.** Ilk 50 pozisyonda 100 adim otesiyle
   benzerlik 0.204, ortada
   0.542.
3. **Pozisyon 1023 hic egitilmemis.** 1022'nin normu
   3.86, 1023'un 0.12.
   Sebep: hedef girdinin bir token kaydirilmis hali; 1023. pozisyonun
   tahmin edecegi 1025. token yok, o embedding hic gradyan almiyor.

Ucuncusu hosuma gitti: Gorev 4'te kendi test kodumda tam o kaydirmayi
atlamistim. Burada ayni kaydirmanin egitilmis bir modelde biraktigi izi
goruyorum.

## 9. Transpoze

Dort tur katman transpoze edilmeli. Beni dusunduren kisim: attn.c_proj
768x768, yani KARE. Transpoze unutulsa boyut tutar, kod calisir ve model
sessizce sacmalar. mlp.c_fc dikdortgen oldugu icin orada hata alirim -
yani gurultulu hata iyi haber.

Logits karsilastirmasi bunun icin: en buyuk fark 1.53e-04.

## 10. Turkce loss  <- dorduncu yanlis tahminim

| | loss | ppl | tabanin yuzdesi |
|---|---:|---:|---:|
| Ingilizce | 3.6937 | 40 | %34 |
| Turkce (isaretli) | 5.5414 | 255 | %51 |
| Turkce (sade) | 6.9797 | 1075 | %64 |

**Bekledigim:** sade yazim token sayisini dusurur, loss da duser; boylece
tokenizer etkisini ayirt ederim.

**Cikan:** token sayisi dustu (163 ->
126) ama loss YUKSELDI (5.541 ->
6.980). Daha cok token kullanan surum
1.44 nat daha iyi.

**Sebep:** internetteki gercek Turkce isaretli harflerle yazilir. Sade
yazim ne Ingilizce ne duzgun Turkce - model icin hic gormedigi bir sey.

Yani token sayisi loss'u belirleyen sey degil; belirleyici olan modelin o
yaziyi daha once gorup gormedigi. Gecen hafta "tokenizer bir semptom, veri
ise sebep" diye yazmistim - bu olcum o cumleyi bekledigimden daha guclu
destekliyor.

## Acik kalanlar

- GELU'nun tanh yaklasimi ile tam formulu arasindaki fark ne kadar?
  Okudum, olcmedim.
- wpe'deki dalgalar gercekten sinus/kosinus'a mi yakinsamis? Fourier ile
  bakmak isterdim; bu hafta goz kararina dayandim.
- top_k=50 neden 50? 10 ve 200 ile ayni tohumla uretip farki gormek
  isterdim.
- Modeli gercekten egitmek. Gelecek haftanin konusu ve asil merak
  ettigim yer.
