# Video metni - YZ50 Hafta 8

**Sure hedefi:** 15-18 dakika. Ekran paylasimi + sesli anlatim.
**Acik olacak dosyalar:** katmanlar.py, gorev2_blok.py cikisi, grafikler,
turkce.txt, uretilen ornekler.

---

## 0 - Acilis (0:00-0:50)

[KAMERA] Yuz, sakin ton.

> "Merhaba, ben Necabeddin. YZ50'nin sekizinci haftasi. Bu hafta GPT'yi
> bitirdik. Gecen hafta tek bir self-attention head yazmistim; bu hafta
> onun etrafina Transformer blogunun kalan parcalarini tek tek ekledim.
> Multi-head, feedforward, residual, LayerNorm. Her parcadan sonra val
> loss'u olctum ve tek bir tabloya yazdim. O tabloyu bu videonun
> merkezine koyacagim."

> "Bastan soyluyorum: bu haftanin en ilginc sonucu bir iyilesme degil, bir
> KOTULESME. Uc blogu ust uste koydugumda model daha kotu oldu. Niye
> oldugunu ve nasil duzeldigini anlatacagim. Ayrica dropout'un benim
> kosullarimda ise yaramadigini da oldugu gibi gosterecegim."

---

## 1 - Gecen haftadan devir (0:50-2:00)

[EKRAN] ablasyon.png, ilk iki nokta.

> "Hafta 7'de bigram modelinden basladim, val loss 2.4769. Sonra
> tek bir self-attention head ekledim, 2.3998'e indi. Kazanc
> 0.0771."

> "Isi haritasina baktigimda tek head'in YEREL bir desen ogrendigini
> gormustum: token cogunlukla kendine ya da bir oncekine bakiyordu.
> Kazancin mutevazi kalmasinin sebebi buydu."

> "Onemli bir not: bu iki sayiyi bu hafta sifirdan yeniden egittim ve
> Hafta 7 raporumdaki sayilarla bire bir ayni cikti. Yani yeni satirlari
> eskisiyle durustce karsilastirabiliyorum."

---

## 2 - Multi-head (2:00-4:30)

[EKRAN] katmanlar.py, MultiHead sinifi.

> "Birinci gorev: tek head'i multi-head'e cevirmek. Yapilan is tek
> cumleyle su: 32 boyutluk tek head yerine, 8 boyutluk dort head."

> "Dikkat edilecek yer: toplam boyut BUYUMUYOR. 32'yi dorde boluyorum.
> Her head kendi key, query, value matrisine sahip, yani 'kime bakacagim'
> kararini bagimsizca veriyor. Sonra dort ciktiyi yan yana ekliyorum -
> 4 kere 8, yine 32 - ve bir lineer katmandan geciriyorum."

> "Analoji: Hafta 7'de sahaya tek bir gozle bakiyordum. O goz her seyi
> ayni anda izlemek zorundaydi - top kimde, pota alti nasil, savunma
> nasil dizilmis, sure ne. Hepsi tek rapora sigmak zorunda oldugu icin
> rapor bulaniklasiyordu. Multi-head dort ayri scout tutmak gibi. Biri
> top tasiyani izliyor, biri pota altini, biri savunma duzenini, biri
> skor tablosunu. Her biri KISA bir rapor yaziyor. Sonra dort rapor
> birlestiriliyor."

> "Scout sayisini artirdim ama toplam rapor uzunlugunu artirmadim. Bu
> yuzden parametre neredeyse hic degismiyor."

[EKRAN] multihead.png.

> "Sonuc: val 2.3998'ten 2.2604'e. Kazanc
> 0.1394. Parametre sadece 1.056
> artti. Yani kazanc modelin buyumesinden degil, isi bolmesinden geliyor.
> Bu beni sasirtti: gecen haftanin tum kazancinin
> 1.81 kati, tek hamlede."

[EKRAN] head_haritalari.png.

> "Peki dort head gercekten farkli seylere mi bakiyor? Kontrol ettim.
> Head 2 digerlerinden farkli: kendine degil, bir onceki BOSLUGA bakiyor
> - yani 'kelime nerede basladi' bilgisini tasiyor. Head 3 ise neredeyse
> tamamen kendine bakiyor, agirlik 0.998. Attention'i pratikte kapatmis.
> Durust olmak gerekirse bu bir miktar kayip. Ama head'lerin son
> satirlari arasindaki ortalama fark 0.04 ile 0.17 arasinda - sifir
> olsaydi bolme bosa gitmis olurdu. Ogrendikleri farkli."

---

## 3 - FeedForward (4:30-6:00)

[EKRAN] katmanlar.py, FeedForward.

> "Ikinci gorev: blogu parca parca tamamlamak. Ilk parca feedforward."

> "Hafta 7'de bir eksiklik vardi: model bilgi TOPLUYOR ama topladigi
> bilgiyle hic hesap yapmiyor. Attention ciktisi dogrudan skorlara
> gidiyordu. Feedforward tam o bosluga giriyor: her token kendi basina,
> 32'den 128'e cikip ReLU'dan gecip 32'ye donen kucuk bir agdan geciyor."

> "Analoji: attention bir toplanti. Herkes digerlerini dinliyor, not
> aliyor. Feedforward masaya donup o notlar uzerinde calismak. Hafta
> 7'de sadece toplanti vardi - insanlar not aliyor ve notu hic okumadan
> karar veriyordu."

> "Ic boyutun dort kat olmasi makaleden geliyor. Masada calisirken not
> defterinden daha genis bir alan gerekiyor."

> "Sonuc: 2.2604'ten 2.1818'e, 0.0786
> kazanc."

---

## 4 - Derinlik ters tepiyor (6:00-8:00)  <- VIDEONUN KALBI

[EKRAN] Ana tablo, teshis satiri vurgulu.

> "Simdi bu haftanin en ogretici kismi. Bir blok ise yaradiysa, uc blok
> daha iyi olmali diye dusundum. Kurdum, egittim."

> "Val loss 2.1818'ten 2.2811'e CIKTI. Yani
> 0.0992 kotulesti. Parametre
> 2.47 katina cikti ve model daha kotu oldu."

> "Bu bir hata degil, beklenen bir sey. Gradyan en alttaki embedding
> katmanina varmak icin butun katmanlardan geri gecmek zorunda. Her
> gecis zincir kuralina bir carpma daha ekliyor. Carpimlar kuculdukce
> alt katmanlara gelen gradyan soniyor."

> "Bunu Hafta 1'de micrograd'i elle yazarken gormustum. Orada tek bir
> zincirde carpimlarin nasil kuculdugunu adim adim izlemistim. Burada
> ayni sey uc blok boyunca oluyor."

> "Bu satiri tablodan silmedim. Teshis satiri olarak biraktim, cunku
> residual'in neden gerekli oldugunu baska hicbir sey bu kadar iyi
> anlatmiyor."

[EKRAN] Residual kod farki, iki satir yan yana.

> "Cozum sasirtici derecede basit. x = sa(x) yerine x = x + sa(x). Yani
> katmanin ciktisini girdinin YERINE koymak yerine, girdiye EKLEMEK."

> "Neden ise yariyor? Hafta 1'de micrograd'da toplamanin geri yayilimini
> kendim kodlamistim: c = a + b ise, a.grad += c.grad ve b.grad +=
> c.grad. Toplama gradyani BOLMEDEN, OLCEKLEMEDEN iki kola da aynen
> geciriyor. Yani kisa yolun gradyani tam 1. Gradyan uc blogu atlayip
> dogrudan embedding'e ulasabiliyor."

> "Analoji: uc katli bir binada her kattan gecmek zorunda olan bir kurye
> yerine, zemin kata inen bir asansor. Kurye hala her kata ugruyor,
> katmanlar ogrenmeye devam ediyor; ama acil paket asansorle gidiyor."

> "Sonuc: 2.2811'ten 2.0700'e. Parametre AYNI. Tek fark
> bir toplama isareti."

---

## 5 - LayerNorm ve Hafta 6 (8:00-11:00)

[EKRAN] eksen.png.

> "Son parca LayerNorm. Her token'in 32 boyutlu vektorunu, kendi icinde
> ortalamasi 0 varyansi 1 olacak sekilde yeniden olcekliyor. Sonuc:
> 2.0700'ten 2.0520'e, 0.0180. Kucuk
> bir kazanc - 3 blok sig bir ag, LayerNorm'un asil faydasi derinlestikce
> cikiyor."

> "Ama ilginc kisim sonuc degil, Hafta 6 ile karsilastirmasi."

[EKRAN] Iki kod blogu yan yana.

> "Hafta 6'da BatchNorm1d'i elle yazmistim. LayerNorm'u yazarken fark
> ettim ki iki sinif neredeyse birebir ayni. Fark TEK BIR KARAKTER:
> BatchNorm mean sifir aliyor, LayerNorm mean bir. Sutun yerine satir."

[EKRAN] gorev2_blok.py cikisi.

> "Karsilastirmanin durust olmasi icin Hafta 6'daki sinifi tek satirini
> bile degistirmeden kopyaladim ve ikisini ayni girdiyle calistirdim.
> BatchNorm'da sutunlarin ortalamasi tam sifir, satirlarin degil.
> LayerNorm'da tersi. Ayni formul, farkli eksen."

> "Simdi gorevin asil sorusu: LayerNorm neden running_mean tutmuyor?
> Deneyle gosterdim. Ayni ornegi iki kere verdim - bir kere dortluk bir
> batch icinde, bir kere tek basina."

[EKRAN] nan satiri gorunur.

> "BatchNorm'da tek ornek verince cikti NAN. Cunku tek ornegin batch
> varyansi tanimsiz. LayerNorm'da iki cikti bire bir ayni, fark tam
> sifir."

> "Kendi cumlelerimle anlatayim. Hafta 6'da BatchNorm yazarken en cok
> ugrastigim sey iki kiplik meselesiydi: egitimde batch'in ortalamasini
> kullan, cikarimda biriktirdigin ortalamayi kullan. Hatta egitimden
> sonra istatistikleri kalibre etmek icin ayri bir dongu yazmistim.
> Butun o karmasanin sebebi su: BatchNorm bir ornegi normalize ederken
> YANINDAKI orneklere bakiyor. Sinavda yanindakinin kagidina bakmak
> gibi. Egitimde sinifta 32 kisi var, ortalamayi hesaplayabiliyorsun.
> Cikarimda tek bir ornek geliyor, yanindaki yok. O yuzden 'sinifin
> ortalamasi genelde su kadardi' notunu running_mean'de saklamak ZORUNDA
> kaliyorsun."

> "LayerNorm bu bagi tamamen kopariyor. Ortalamayi satir boyunca, yani
> token'in kendi boyutlari boyunca aliyor. Komsuya hic bakmiyor. Yani
> tek ornek de bin ornek de ayni sonucu verir. Saklanacak bir sey yok,
> running_mean gereksiz, egitim ve cikarim kodu ayni - self.training
> bayragina bile gerek kalmiyor."

> "Dil modelinde bu ozellikle onemli, cunku uretim sirasinda tek tek
> token besliyorum. BatchNorm olsa her uretim adiminda bir 'batch
> ortalamasi' uydurmam gerekirdi."

[EKRAN] Bessel tablosu.

> "Bir de yol ustunde bir hata yaptim, onu da anlatmak istiyorum. Kendi
> yazdigim LayerNorm ile PyTorch'un LayerNorm'u arasinda 1.66 carpi 10
> uzeri eksi 1 fark cikti. Sifira yakin olmasi gerekiyordu. Sebep:
> PyTorch'ta var fonksiyonu varsayilan olarak n eksi bire boler - Bessel
> duzeltmesi, Hafta 6'da ogrendigim sey. nn.LayerNorm ise n'e boluyor.
> unbiased esittir False ekledim, fark 3.58 carpi 10 uzeri eksi 7 oldu."

> "Ve burada bir sey yerine oturdu. Hafta 6'da n eksi bir kullanmam
> dogruydu, cunku batch butun verinin bir ORNEGIYDI; ornekten evren
> varyansini kestirirken n eksi bir gerekiyor. LayerNorm'da normalize
> ettigim sey tek bir token'in butun ozellikleri - ortada kestirim yok,
> elimdeki veri evrenin kendisi. Ayni formul, farkli istatistiksel
> durum. Hafta 6'da ezberledigimi sandigim bir ayrintinin ne ise
> yaradigini burada anladim."

---

## 6 - Tek tablo (11:00-12:15)

[EKRAN] Ana tablo, tam ekran.

> "Simdi tabloyu butunuyle gosteriyorum. Yedi satir, hepsi ayni butceyle
> egitildi: block 8, batch 32, n_embd 32, 5000 adim, ayni tohum. Tek
> degisen mimari."

> "bigram 2.4769. Tek head 2.3998. Multi-head
> 2.2604. Feedforward 2.1818. Yildizli satir -
> residual'siz uc blok - 2.2811, yani ters tepen deney. Residual
> 2.0700. LayerNorm 2.0520."

> "Toplam kazanc 0.4248."

> "Tablodan uc sey cikardim. Birincisi: en buyuk tek kazanc multi-head'de
> ve parametre eklemeden geldi. Mimari secimi kapasiteden daha onemli
> olabiliyor. Ikincisi: residual, sifir parametreyle
> 0.2110 kazandirdi - ayni parametre sayisiyla iki
> model, fark sadece bir toplama isareti. Ucuncusu: parametre artisinin
> buyuk kismi residual OLMADAN zarar veriyordu. Yani 'daha cok parametre
> daha iyi' cumlesi yanlis; dogrusu 'daha cok parametre arti gradyanin
> akacagi bir yol'."

---

## 7 - Olcekleme ve dropout (12:15-14:00)

[EKRAN] Donanim olcum tablosu.

> "Ucuncu gorev: modeli olceklemek. Karpathy videoda n_embd 384, alti
> head, alti blok, block 256 ile egitiyor. Bende GPU yok, iki CPU
> cekirdegi var. Tahmin etmek yerine olctum."

> "Karpathy'nin config'i bu makinede 3000 adim icin 305 dakika. Bes
> saat. O yuzden kuculttum: n_embd 128, dort head, dort blok, block 64.
> 816.705 parametre, kosu basina alti ile dokuz dakika arasi."

> "block 64 secimim bilincli. Hafta 7'de 8 karakterdi; 64 karakter
> Shakespeare metninde yaklasik iki satir, yani model artik konusan
> kisinin adini gorup satirin devamini ona gore kurabiliyor. Hafta 6'da
> WaveNet denerken en buyuk kazancin hiyerarsiden degil BAGLAMI
> UZATMAKTAN geldigini olcmustum - ayni dersi burada kullandim."

[EKRAN] Dropout tablosu.

> "Dropout da ekledim, yuzde 10. Analoji: takimi her antrenmanda rastgele
> iki oyuncu eksik cikartirsan, kimse tek bir kisiye bagimli kalmaz. Mac
> gununde takim tam kadro cikar."

> "Ama burada durust olmam gereken bir sonuc var. Dropout'suz: train
> 1.4055, val 1.6113. Dropout'lu: train
> 1.4552, val 1.6476. Yani val loss DUSMEDI, yukseldi.
> Dropout bu kosullarda modeli iyilestirmedi."

> "Tek olumlu isaret asiri uyum farkinda: val eksi train
> 0.2058'ten 0.1924'e daraldi.
> Dropout isini yapiyor, kazanci henuz loss'a yansimamis."

> "Benim yorumum: dropout yakinsamayi yavaslatiyor ve faydasi model asiri
> uyuma girdikten sonra ortaya cikiyor. 4000 adimda dropout'suz model
> henuz asiri uyumun derinine inmemis, yani duzeltilecek cok bir sey yok.
> Daha uzun egitimde tablonun tersine donmesini beklerdim ama olcemedim."

> "Bu, Hafta 4'te BatchNorm'un dev loss'u biraz kotulestirdigini
> yazdigimdaki durumun aynisi. Kitapta 'iyi' diye gecen bir teknik her
> kosulda iyi olmuyor ve bunu olcerek gormek, tekniklerin ne zaman ise
> yaradigini anlamanin tek yolu."

[EKRAN] Uretilen metin, yan yana.

> "Ve ciktilar. Solda kucuk model, sagda olcekli model. Fark olculen loss
> farkindan daha carpici: buyuk model satir yapisini, konusan adini,
> kelime uzunlugu dagilimini yakalamis. Hala anlamli Ingilizce degil -
> olmasini da beklemiyorum, bu bir karakter modeli ve 1 MB veriyle
> egitiliyor."

---

## 8 - Turkce (14:00-16:00)

[EKRAN] turkce.txt acik, sonra veri tablosu.

> "Dorduncu gorev: ayni modeli Turkce metinle egitmek. Kendim yaklasik
> 1 MB metin derledim. Gorevde Vikikaynak oneriliyordu; kodu
> calistirdigim ortamdan wikimedia'ya erisim kapaliydi, o yuzden
> erisebildigim ve ayni isi goren serbest lisansli bir Turkce kaynak
> kullandim - roman, deneme, haber karisimi. Derleme betigi
> turkce_derle.py dosyasinda."

> "Farkli karakter sayisi: Shakespeare'de 65, Turkce'de 87. Fark
> Turkce'nin alti ozel harfinden geliyor - c, g, i, o, s, u uzerine
> isaretli haller - hem kucuk hem buyuk, 12 ek karakter. Ustune
> rakamlar; Shakespeare dosyasinda rakam hic yok."

[EKRAN] Turkce cikti.

> "Model Gorev 3'tekiyle bire bir ayni. val loss 1.5841. Iste
> urettigi metin."

[EKRAN] turkce_vs_sh.png.

> "Simdi onemli soru. Turkce model 1.5841 aldi, Shakespeare
> modeli 1.6476. Yani Turkce model DAHA IYI gorunuyor. Bu iki
> sayiyi yan yana koyup 'Turkce daha kolay bir dil' diyebilir miyim?
> Hayir. Uc sebepten."

> "Birincisi alfabe boyutu. cross_entropy'nin olcegi secenek sayisina
> bagli. Hicbir sey ogrenmemis bir model loss esittir log V alir.
> Shakespeare'de log 65 esittir 4.1744, Turkce'de log 87
> esittir 4.4659. Turkce model daha baslamadan 0.2915
> handikapli. 87 secenek arasindan dogruyu bulmak, 65 arasindan
> bulmaktan tanimi geregi daha zor. Adil olcut ham loss degil, tabandan
> ne kadar uzaklastigi."

> "Ikincisi diller ayni degil. Turkce sondan eklemeli; 'gel' kokunden
> 'gelemeyeceklerini' cikiyor. Karakter seviyesinde kelimenin sonunu
> tahmin etmek kolay ama kelime sinirini ve hangi ekin gelecegini tahmin
> etmek zor."

> "Ucuncusu metinler ayni turden degil. Tiny Shakespeare tek bir yazarin
> tiyatro metni, ayni kalip surekli tekrar ediyor. Benim korpusum roman,
> deneme, haber karisimi."

> "Bir de kendi hatami buldum. Turkce modelde val loss, train loss'tan
> DUSUK cikti - normalde tersi olur. Sebebi: korpusu yuzde 90 yuzde 10
> olarak SIRAYLA boldum ve son yuzde 10 daha homojen cikti, yani tahmin
> etmesi kolay. Yeniden yapsam cumleleri karistirip bolerdim. Hafta 6'da
> 'bolme bicimi olcumu etkiler' diye not almistim; pratik sonucunu
> burada gordum."

> "Bu da Hafta 6 ve 7'de ogrendigim seyin ucuncu kez karsima cikisi: bir
> sayiyi sadece olcum kosullari ayni oldugunda karsilastirabilirsin."

---

## 9 - Ek calisma (16:00-17:15)

[EKRAN] gorev5_ek.py cikisi.

> "Besinci gorev istege bagliydi, iki secenekten biri. Ikisini de yaptim."

> "Birincisi: butun head'leri tek matmul'de hesaplamak, nanoGPT gibi.
> Dongulu surumde dort head carpi uc katman, on iki kucuk matris
> carpimi. nanoGPT tek bir buyuk Linear kullaniyor ve sonucu sonradan
> head'lere boluyor."

> "Denkligi nasil gosterdim: dongulu surumun agirliklarini tek-matmul
> surume kopyaladim ve ayni girdiyi ikisinden gecirdim. En buyuk fark
> TAM SIFIR cikti. Bunu beklemiyordum; float32'de toplama sirasi
> degisince genelde 1e-7 mertebesinde bir fark kalir. Sebebini
> dusununce anladim: head'leri view ve transpose ile bolunce her head'in
> ic carpimi hala ayni sekiz sayiyi ayni sirada topluyor. Yani sadece
> matrisler birlestirildi, toplama sirasi hic degismedi."

> "Hafta 5'te manuel geri yayilim yazarken kurdugum olcut tam buydu:
> EXACT mi, approx mi. Orada bazi gradyanlar EXACT cikiyordu, bazilari
> 1e-9 mertebesinde sapiyordu. Burada EXACT cikti. Hiz kazanci 1.25 ile
> 1.85 kat arasinda; GPU'da daha buyuk olurdu."

> "Ikincisi: makalenin Sekil 1'ini kendi kodumla eslestirmek. Sekil iki
> kule: solda encoder, sagda decoder. Bende sag kulenin cross-attention
> cikarilmis hali var."

> "Encoder bende yok, cunku encoder bir KAYNAK diziyi okumak icin var -
> makale ceviri icin yazilmis. Benim gorevim ceviri degil, metin devam
> ettirme; okunacak ayri bir kaynak dizi yok. Cross-attention da yok,
> cunku query decoder'dan, key ve value encoder'dan gelir; encoder
> olmadigi icin key ve value'nun gelecegi yer yok. Benim blogumda tek
> attention var ve ucu de ayni x'ten geliyor. 'Self' kelimesi tam olarak
> bunu anlatiyor. Mimariye 'decoder-only' denmesinin sebebi de bu."

---

## 10 - Kapanis (17:15-18:00)

[KAMERA] Yuz.

> "Toparliyorum. Bu hafta ogrendigim en onemli sey sayilarda degil,
> sayilarin arasindaki iliskide. Multi-head parametre eklemeden
> 0.1394 kazandirdi. Residual sifir parametreyle
> 0.2110 kazandirdi. Ayni parametreyle egitilmis iki
> model arasinda bir toplama isareti kadar fark var ve o fark her seyi
> degistiriyor."

> "Durust olmak icin soylemem gereken seyler: Karpathy'nin videodaki
> config'ini calistiramadim, donanim yetmedi. Dropout benim butcemde ise
> yaramadi. Turkce tarafta val bolmesini yanlis kurdum. Bunlarin
> hicbirini raporda sakladim degil, hepsini yazdim - cunku bence
> ogrenmek tam olarak bu: neyin ne zaman ise yaradigini olcerek gormek."

> "Hala tam netlesmeyen seyler var. Head 3'un neredeyse tamamen kendine
> bakmasi bir kayip mi, bilincli bir bypass kanali mi, emin degilim.
> Dropout daha uzun egitimde kazanca donuyor mu, olcmek isterdim.
> Karakter yerine BPE kullansam Turkce'nin eklemeli yapisinda ne
> degisirdi, onu merak ediyorum."

> "Kod, rapor ve butun ciktilar repoda. Bu isi gercekten ogrenmek
> istiyorum ve calismaya devam ediyorum. Tesekkurler."

---

## Cekimden once kontrol listesi

- [ ] gorev2_blok.py cikisi terminalde hazir (tablo ve nan satiri gorunur)
- [ ] Grafikler ayri sekmelerde acik: ablasyon.png, multihead.png,
      head_haritalari.png, eksen.png, olcekleme.png, turkce_vs_sh.png,
      blok_semasi.png
- [ ] katmanlar.py acik - MultiHead, FeedForward, LayerNorm1d, Blok
      siniflari gorunur
- [ ] Turkce cikti ve Shakespeare ciktisi yan yana kopyalanmis
- [ ] Yazi boyutu buyutulmus, ekranda okunabilir
- [ ] Mikrofon testi
