# Hafta 12 video metni

Mailin istedigi uc sey: **loss grafigi** (bolum 2), **uretilen metin** (bolum 3), **HellaSwag sonucu** (bolum 4).

`[EKRAN]` ekran paylasimi, `[KAMERA]` kameraya konusma. Yalniz `>` ile baslayan satirlar soyleniyor.

**Tam metin yaklasik 7 dakika** (951 kelime, 130 kelime/dk). **Kisa surum yaklasik 4 dakika:** bolum 0, 2, 3, 4, 7.

| Bolum | Kelime | Sure | Kisa surumde |
|---|---|---|---|
| 0. Giris | 63 | 0.5 dk | VAR |
| 1. Veri | 110 | 0.8 dk | - |
| 2. Egitim ve loss grafigi | 137 | 1.1 dk | VAR |
| 3. Model ne uretiyor | 72 | 0.6 dk | VAR |
| 4. HellaSwag | 220 | 1.7 dk | VAR |
| 5. Fark nereden geliyor | 80 | 0.6 dk | - |
| 6. Ek gorev: Turkce | 194 | 1.5 dk | - |
| 7. Tahminler ve kapanis | 75 | 0.6 dk | VAR |
| **TOPLAM** | **951** | **7 dk** | **4 dk** |

---

## 0. Giris

**[KAMERA]**

> Merhaba, ben Neco. Bu, YZ50'nin son haftasi.
>
> On iki hafta once tek bir noronun gradyanini elle hesaplayarak basladik.
> Iki hafta once OpenAI'in GPT-2 agirliklarini kendi yazdigim koda yukledim,
> gecen hafta ayni modeli hizlandirdim. Bu hafta eksik kalan tek parca kaldi:
> modeli kendim egitmek ve OpenAI'in modeliyle ayni testte olcmek.
>
> Bu videoda uc sey gosterecegim: egitimin loss grafigi, modelin urettigi
> metin ve HellaSwag sonucu.

## 1. Veri

**[KAMERA]**

> Veri FineWeb-Edu: Hugging Face'in egitim amacli metinlerden sectigi bir veri
> seti. Calistigim ortamdan Hugging Face'e erisim kapali oldugu icin veriyi
> tarayicimdan, resmi satir API'siyle yuzer belgelik parcalar halinde cektim.
> 4.000. belgeden sonra API beni durdurdu. Elimde 4.000 belge,
> yani 4,1 milyon token kaldi.
>
> Mail birkac on milyon token oneriyor. Bende neden bu kadari yetti, onu
> egitim sonucunda gosterecegim; kisa cevap: kisitlayan veri degil, hesap.

**[EKRAN: gorev1_veri.py ciktisi]**

**[KAMERA]**

> Her belgenin basina end-of-text token'i koyuyorum ki model iki belgeyi tek
> metin gibi ogrenmesin. Token'lari uint16 olarak sakliyorum: en buyuk token
> elli bin iki yuz elli alti, uint16 altmis bes bine kadar tutuyor. Sonra
> bes yuz bin token'lik dosyalara boldum; ilk dosya val, modele hic gosterilmiyor.

## 2. Egitim ve loss grafigi

**[KAMERA]**

> Modeli Hafta 11'deki olcum tablosunun en iyi satiriyla egittim: bfloat16,
> flash attention ve torch.compile. Adim basina dort bin doksan alti token,
> learning rate warmup'li cosine. Toplam 500 adim.

**[EKRAN: kayip.png]**

**[KAMERA]**

> Mavi train, kirmizi val. Mailin istedigi uc sayi:
> model 2.048.000 token gordu, saniyede 559 token isledi,
> son val loss 6,80. Baslangicta 10,95 idi; bu da hicbir
> sey bilmeyen modelin beklenen degeri, elli bin iki yuz elli yedinin logaritmasi.
>
> Dikkat cekmek istedigim sey iki egrinin ust uste durmasi. Ilk bakista "model
> genelliyor" diye okunabilir. Ama sebebi daha basit: model egitim verisinin
> yuzde 56'ini gordu, yani hicbir belgeyi iki kez gormedi.
> Ezberleyecegi bir tekrar yok. Metni bir devirden fazla gordugunde ne oldugunu
> birazdan Turkce'de gosterecegim.
>
> Bir de veri sorusuna doneyim: bu hizla bir saatte yaklasik
> 2,0 milyon token isleniyor. On kat veri indirseydim model yine
> ayni sayida token gorecekti. Kisitlayan hesap.

## 3. Model ne uretiyor

**[EKRAN: egitim logundaki ornekler]**

```
adim    0: Hello, I'm a language model, surrounded restraints unsolved AX intimidation Fo rodents */ betting weight Tigers generator dismal Martinezimatedorical19721972ina
adim  250: Hello, I'm a language model, and just have a same history and a study to be the "in in the other than a other own, will take us were found in the other and used
adim  500: Hello, I'm a language model, and if to�dus The C, and as the New York, which is some of his own group-B; this, and in the government
```

**[KAMERA]**

> Her elli adimda ayni baslangic ve ayni tohumla metin urettim, yani degisen tek
> sey model. Adim sifirda tamamen rastgele token'lar. Sonra en sik kelimeler:
> "the the, of". Sonra noktalama ve satir yapisi. En son, kisa ama dilbilgisi
> olarak duzgun parcalar.
>
> Bu sira tanidik geldi: Hafta 2'de bigram modeli isimlerde ayni yolu izliyordu.
> Once frekans, sonra yerel duzen. Anlam en son geliyor ve bu olcekte henuz
> hic gelmedi. HellaSwag tam da bunu soruyor.

## 4. HellaSwag

**[KAMERA]**

> HellaSwag'de her ornekte bir baglam ve dort devam var. Biri dogru; ucu
> dilbilgisi olarak dogru ama anlam olarak yanlis. Rastgele tahmin yuzde yirmi
> bes bulur.
>
> Kucuk bir model soruyu okuyup "C" diyemez. O yuzden dort devami ayri ayri
> baglamin arkasina ekliyorum ve modele soruyorum: bu devam ne kadar olasi?
> Token basina loss'u en dusuk olan, modelin cevabi.

**[EKRAN: hellaswag.png]**

**[KAMERA]**

> Ilk bin ornekte benim modelim yuzde 23,5 buldu, OpenAI'in GPT-2'si
> yuzde 34,0.
>
> Bu sayiyi okumadan once bir sey bilmek lazim: rastgele tahmin eden bir model
> bin ornekte tam iki yuz elli bulmaz. Yuzde doksan bes olasilikla yuzde
> 22,3 ile 27,7 arasinda bir sey bulur.
> Benim sonucum bu araligin icinde. Durust okumasi su: bir saatlik egitim
> HellaSwag'de olculebilir bir sey kazandirmadi.

**[KAMERA]**


> Peki benim modelim rastgele ciktiysa, bunun sebebi model mi, olcum kodum mu?
> Ayirt etmek icin OpenAI'in modelini tam sette, on bin kirk iki ornekte kostum.
> Benim kodum yuzde 29,68 buldu; Karpathy'nin raporladigi sayi yuzde
> yirmi dokuz virgul elli bes. Kod dogru; fark modelden.

**[KAMERA]**

> Ve bu haftanin beni en cok sasirtan sonucu: hic model kullanmayan bir secici
> kurdum, baglamla en cok kelime paylasan devami seciyor. Tam sette
> yuzde 29,57 buluyor; OpenAI'in GPT-2'si yuzde 29,68. Ayni seviye,
> ama cogunlukla farkli orneklerde. Yani kucuk GPT-2'nin HellaSwag skoru
> rastgeleden iyi ama basit bir sezgiselden iyi degil. Karpathy'nin on milyar
> token'lik kosusunun bu sayiyi gecmesi bu yuzden anlamli.

## 5. Fark nereden geliyor

**[EKRAN: olcek.png]**

**[KAMERA]**

> Mimari ayni, parametre sayisi ayni, tokenizer ayni. Iki hafta once OpenAI'in
> agirliklarini bu koda yukleyip ayni logits'i almistim. O halde farkin tamami
> egitimden geliyor.
>
> Benim modelim 2,0 milyon token gordu. Karpathy ayni mimariyle
> GPT-2'yi on milyar token'da geciyor: bu 4.883 kat. Ve bu CPU'da on milyar
> token yaklasik 207 gun surerdi.
>
> Kendi cumlelerimle: ayni bina plani, ayni malzeme. OpenAI binayi bitirdi, ben
> temeli yeni attim. HellaSwag cati katinda ne oldugunu soruyor; benim modelim
> henuz zemin katta, o yuzden cevabi sans.

## 6. Ek gorev: Turkce

**[EKRAN: turkce.png]**

**[KAMERA]**

> Ayni modeli Hafta 8'den beri kullandigim Turkce metinle egittim, bir devirden
> fazla. Tahminim, model metni ikinci kez gorunce val'in yukselmesiydi.
> Tutmadi. Val adim 160'de en dusuk, sonraki olcum ancak 0,02 nat yukarida;
> bu gurultu duzeyinde. Bir virgul alti devir ezber icin yetmedi. Yaniltici bir
> yorum yapmak yerine bunu soyluyorum.
>
> Bir de Turkce loss'un Ingilizce'den dusuk gorunmesi: bu bir birim yanilsamasi.
> Tokenizer Turkce'yi token basina 2,0 karaktere boluyor, kisa parca tahmin etmek kolay.
>
> Urettigi metinden bir ornek: "Türkiye'nin başkentişlar ve azulmaya olsun kendiz, kük eşük. Ar kalkaşüne iştısınkü".
>
> HellaSwag Ingilizce; Turkce modeli nasil olcerim? Iki yol var. Birincisi
> Hafta 9'dan beri kullandigim karakter basina bit: ayrilmis Turkce metinde bir
> karakteri kodlamak kac bit tutuyor. Sagdaki grafik uc modeli iki dilde
> gosteriyor.
> Turkce'de en iyi model: Benim, 3,16 bit.
>
> Ikinci yol bir Turkce HellaSwag. Hugging Face'te hellaswag-tr adinda,
> ayni orneklerin Turkce cevirisi var. Ayni kodla, yalniz veri dosyasini
> degistirerek kostum.
>
> Turkce modelim yuzde 28,2, OpenAI yuzde 33,1.
> Turkce modelim rastgelenin hemen ustunde. Ama OpenAI'in modeli, neredeyse hic
> Turkce gormemisken, daha yuksek. Bunu aciklayamiyorum. Ilk aklima gelen,
> modelin baglamdaki kelimeleri devamda aramasiydi; test ettim, Turkce'de kelime
> ortusmesi tek basina yuzde 25,5 buluyor. Aciklamiyor. Acik soru olarak birakiyorum.

## 7. Tahminler ve kapanis

**[EKRAN: tahminler tablosu, sunum]**

**[KAMERA]**

> Hafta dokuz, on ve on birde sekiz tahminim olcumle yanlislandi. Bu hafta
> bunu kurala cevirdim: tahminleri sonuclardan once, zaman damgasiyla bir
> dosyaya yazdim ve degistirmedim. Hangilerinin tuttugu ekranda.
>
> Son olarak, on iki haftadan kalan seyi soylemek istiyorum. Bir teknik degil,
> bir aliskanlik: bir gerekce yazdigimda yanina "bunu nasil olcerim" ve "bunu
> daha once olcmus muydum" sorularini yazmak. Bu hafta ucuncusunu ekledim:
> tahmini sonuctan once yaz.
>
> Izlediginiz icin tesekkurler, ve bu on iki hafta icin tesekkurler.

