# Video metni - YZ50 Hafta 9

**Sure hedefi:** 12-15 dakika. Ekran paylasimi + sesli anlatim.
**Gorevde istenen:** tokenizer'in ilk birlestirmelerini ve Turkce-Ingilizce
karsilastirmasini goster. Ikisi de videonun merkezinde.
**Acik olacak:** gorev2_bpe.py cikisi, gorev4 ve gorev5 ciktilari,
grafikler, bpe.py.

---

## 0 - Acilis (0:00-0:50)

[KAMERA] Yuz.

> "Merhaba, ben Necabeddin. YZ50'nin dokuzuncu haftasi. Sekiz haftadir
> yazdigim modeller metni harf harf okuyordu. Bu hafta onu biraktim ve
> metni parcalara ayiran tokenizer'i sifirdan yazdim - GPT-4'un
> kullandigi yontemle, Byte Pair Encoding ile."

> "Iki sey gosterecegim. Birincisi tokenizer'imin ilk birlestirmeleri -
> ve orada benim beklemedigim bir sey cikti. Ikincisi ayni paragrafin
> Turkce ve Ingilizce halinin kac token ettigi. Arada ciddi bir fark
> var ve sebebini anlatacagim."

---

## 1 - Neden harf seviyesinden ciktim (0:50-2:00)

[EKRAN] Hafta 8'in kodu, stoi sozlugu.

> "Once neden. Hafta 8'de Turkce korpusumdaki farkli karakterleri
> saymistim: 87. Her birine bir numara verdim ve isim bitti. Calisiyordu
> ama uc bedeli vardi."

> "Birincisi: diziler cok uzun. Model her karakter icin bir adim atiyor.
> block_size 64'tu, yani model ayni anda en fazla 64 karakter
> gorebiliyordu - Turkce'de bu yaklasik on kelime. Attention maliyeti
> dizi uzunlugunun karesiyle arttigi icin pencereyi buyutmek pahali."

> "Ikincisi: o 87 sayisi tamamen keyfi. Sinifta ayni gorevde bu sayi
> 89'dan 353'e kadar degismis - dort kat fark. Sebep, metni kimin nasil
> temizledigi. Benim 87'm de o keyfi sayilardan biri."

> "Ucuncusu: modele bir emoji versem KeyError alirdim. Sozlugumde yoktu."

> "BPE ucune birden cevap veriyor. Tabanda 256 byte var - herkeste ayni,
> keyfi degil ve her metni temsil edebiliyor."

---

## 2 - Unicode ve UTF-8 (2:00-3:30)

[EKRAN] gorev1_byte.py cikisi.

> "Ama once bir ara durak: bilgisayar bir harfi nasil tutuyor."

> "Sectigim cumle: 'Cigdem'in agirligi 72 kg, Izmir'de usudu ve sasirdi.'
> Elli iki karakter. Ama altmis sekiz byte. Arada on alti byte fark var."

> "Fark tamamen Turkce'nin isaretli harflerinden geliyor. UTF-8 degisken
> uzunluklu bir kodlama: ASCII harfler bir byte, Turkce'nin c-g-i-o-s-u
> uzerine isaretli harfleri iki byte, Cince uc byte, emoji dort byte.
> Butun korpusumda bu fark yuzde 9.1."

> "Gorevde soru soyle: kod noktalarini dogrudan vocab yapmak neden kotu
> bir fikir? Dort sebep gordum."

> "Birincisi tablo cok buyuk ve cogu bos. Unicode'da yaklasik 150 bin
> atanmis kod noktasi var. Hafta 8'deki modelimde embedding tablosu V
> carpi 128 parametreydi; V 150 bin olsaydi sadece o tablo 19 milyon
> parametre olurdu. Modelin tamami 816 bindi. Yani modelden yirmi uc kat
> buyuk, tamami bos bir tablo."

> "Ikincisi kucultursem bilinmeyen karakter sorunu geri geliyor.
> Ucuncusu Unicode her yil degisiyor, byte'lar degismiyor. Dorduncusu
> ayni gorunen metnin birden fazla yazilisi olabiliyor."

> "Peki neden dogrudan byte kullanmiyoruz? Kullanabiliriz, vocab 256
> olur. Ama o zaman diziler uzar. BPE ikisinin ortasi."

---

## 3 - BPE ve ilk birlestirmeler (3:30-6:30)  <- GOREVDE ISTENEN

[EKRAN] bpe.py, istatistik ve birlestir fonksiyonlari.

> "Algoritma sasirtici derecede basit. Metni byte'lara cevir. Ardisik
> ikilileri say. En sik geceni bul, ona yeni bir numara ver. Metinde o
> ikiliyi tek sayiyla degistir. Vocab dolana kadar tekrarla. Hepsi bu -
> iki kisa fonksiyon."

[EKRAN] Kucuk ornek ciktisi.

> "Once kucuk bir ornekte izleyelim: 'agacagacagac'. On sekiz byte.
> Adim bir: en sik ikili doksan yedi ve yuz doksan alti, uc kere geciyor,
> 256 numarasini aliyor."

> "Burada dikkat cekici bir sey var. Algoritma 'g' harfinin iki byte'ini
> birlikte gormuyor. Once 'a' ile o harfin ILK byte'ini birlestiriyor ve
> ortaya tek basina gecerli UTF-8 olmayan bir token cikiyor - ekranda
> kirik karakter olarak gorunuyor. Ikinci adimda tamamlaniyor."

> "Yani ara adimlarda yarim karakterler olusuyor. Bu bir hata degil,
> byte seviyesinde calismanin dogal sonucu. Birazdan decode yazarken
> bunun bedelini odeyecegim."

[EKRAN] gorev2_bpe.py cikisi, ilk 20 birlestirme tablosu.

> "Simdi gercek egitim. Hafta 8'de kendim derledigim bir megabaytlik
> Turkce metin, vocab 1024, yani 768 birlestirme."

> "Ve iste gorevde istenen sey: ilk yirmi birlestirme."

> "Ben ilk birlestirmenin sik bir harf IKILISI olmasini bekliyordum -
> 'la', 'er' gibi. Cikan sey bu degil."

> "Birinci birlestirme: noktasiz i. TEK BIR HARF. Nasil oluyor? Cunku
> Turkce'nin noktasiz i'si UTF-8'de iki byte ve benim metnimde
> 37.557 kere geciyor. BPE byte seviyesinde
> calistigi icin bu harfi 'iki byte'lik bir ikili' olarak goruyor."

> "Ve bu tek bir ornek degil. Dorduncu birlestirme 'u', sekizinci 's',
> on ucuncu 'c', on besinci 'g'. Yani algoritma, kimse ona Turkce
> ogretmeden, ilk islerden biri olarak TURKCE ALFABESINI yeniden
> kesfediyor."

> "Sonrasinda ne geliyor? 'ar', 'er', 'an', 'in', 'lar', 'il'. Yani
> ekler. Turkce sondan eklemeli oldugu icin ekler cok tekrar ediyor ve
> BPE onlari hizla yakaliyor."

[EKRAN] sikistirma.png

> "Sonuc: sikistirma 2.639 kat. Yani ayni block_size'a
> 2.41 kat daha fazla metin
> sigiyor. Hafta 8'deki en buyuk kisitim buydu."

> "Kazanc dogrusal degil: ilk elli birlestirme sikistirmayi 1.52'ye
> cikariyor, kalan 718 birlestirme ancak 2.639'e tasiyor.
> Mantikli - en sik ikililer once secildigi icin ilk birlestirmeler en
> cok yeri kapatiyor."

---

## 4 - encode, decode ve gecersiz UTF-8 (6:30-8:00)

[EKRAN] gorev3 ciktisi.

> "decode(encode(metin)) metnin kendisini vermeli. Butun orneklerde
> tamam."

> "En cok sevdigim sonuc burada: emoji de calisiyor. Halbuki
> tokenizer'i SADECE Turkce metinde egittim, egitim verimde hic emoji
> yoktu. Sebebi taban vocab'in 256 byte olmasi - her UTF-8 metin
> byte'lara cevrilebiliyor. Emoji dort ayri byte olarak gidiyor,
> verimsiz ama kayipsiz."

> "Hafta 8'deki tokenizer'ima ayni emojiyi versem KeyError alirdim.
> Bastaki 'bilinmeyen karakter' probleminin somut karsiligi bu."

[EKRAN] replace / ignore / strict tablosu.

> "Gorevde soru soyle: bazi token dizileri gecerli UTF-8 degil, decode
> ne yapmali?"

> "Uc secenek var. strict cokuyor, ignore sessizce siliyor, replace
> yerine kirik karakter koyuyor. Ben replace sectim."

> "strict degil, cunku model uretim sirasinda ne urettiginin garantisi
> yok - ortasinda kesilmis bir cok-byte'li karakter uretebilir. Burada
> program cokerse metin uretimi durur, oysa olan sey sadece bir
> karakterin yarim kalmasi."

> "ignore degil, cunku sessizce siliyor ve bir sey ters gittiginde
> haberim olmuyor. replace ile ciktida goruyorum ve ne kadar sik
> oldugunu olcebiliyorum."

---

## 5 - GPT-4'un regex deseni (8:00-10:30)

[EKRAN] desen.

> "Simdi dorduncu gorev. Benim tokenizer'im metni tek bir uzun byte
> dizisi olarak goruyor ve istedigi yerde birlestirebiliyor - kelime
> sinirini gecip boslugu icine alabiliyor. GPT-2 makalesi bunu sorun
> olarak isaretliyor: metni ONCE regex ile bol, BPE'yi her parcada ayri
> calistir."

> "Deseni ekleyip tokenizer'i yeniden egittim. Ilk yirmi birlestirmenin
> sadece 1'i ayni kaldi - desen vocab'in tamamini
> degistiriyor."

[EKRAN] vocab_yapisi.png

> "Ve burada iki tahminim de tutmadi, ikisini de anlatacagim."

> "Birinci tahminim: 'regex'li surumde bir token ya harf dizisi ya
> noktalama dizisi olur.' Sayilara baktim: regex'li surumde basinda
> bosluk olan token sayisi daha FAZLA. 322'e karsi
> 30."

> "Desene tekrar baktim ve sebebini anladim. Kuralda 'basta bir
> harf-disi karakter OLABILIR' diyor. Yani bastaki bosluk bilerek
> serbest birakilmis. GPT ciktilarinda token'larin bosluk-the,
> bosluk-and diye gorunmesinin sebebi bu."

> "Gercek fark sonda: sonu bosluklu token sayisi 243'ten
> 0'a dustu. Neden onemli? Sonunda bosluk olan token,
> kelimeyi kendinden SONRA geleni de iceren bir birim yapiyor; ayni
> kelime cumle sonundayken baska, ortasindayken baska bolunuyor.
> Bastaki bosluk ise her zaman ayni."

> "Ikinci tahminim: 'regex sikistirmayi kotulestirir.' Tersine, biraz
> iyilesti. Sebebi: regex'siz surumde ayni kelime yanindaki noktalamaya
> gore farkli farkli bolunuyordu ve bu parcalanmalar vocab butcesini
> tuketiyordu."

> "Ikisini de rapora yazdim. Raporu dogru gostermek icin degil, ne
> ogrendigimi gostermek icin yaziyorum."

---

## 6 - Desen Turkce'de ne yapiyor (10:30-12:00)  <- EN NET BULGU

[EKRAN] ekler.png

> "Simdi bu haftanin en net bulgusu. Gorevde 'desen Ingilizce icin
> yazilmis, Turkce'de ne yapiyor' diye soruluyor."

> "Benim beklentim 'kesme isareti kendi basina bir parca olur, ek
> kokten kopar' idi. Yanlis cikti ve dogrusu daha ilginc."

> "Yirmi alti Turkce eki tek tek denedim. 12 tanesi
> bozuluyor, 14 tanesi temiz kaliyor."

> "Bakin: 'Turkiye'nin' temiz bolunuyor - iki parca, kok ve ek.
> Ama 'Ankara'dan' uc parcaya ayriliyor: 'Ankara', kesme-d, ve 'an'."

> "Neden? Desende iki kural yarisiyor. Birinci kural Ingilizce
> kisaltmalarini yakaliyor: kesme-s, kesme-d, kesme-m, kesme-t,
> kesme-ll, kesme-ve, kesme-re. Ikinci kural ise harf-disi bir karakter
> arti harf dizisi."

> "Regex alternatifleri SIRAYLA deniyor. Ek d, t, s veya m ile
> basliyorsa birinci kural once esleniyor ve ekin sadece ILK HARFINI
> aliyor. Geri kalan ayri bir parcaya dusuyor."

> "Ve bozulanlara bakin: hepsi d, t, s, m ile basliyor. Tesaduf degil.
> Bunlar Turkce'nin en sik kullanilan ekleri - bulunma hali da-de-ta-te
> ve ayrilma hali dan-den-tan-ten. Yani desen, Turkce'de en cok ihtiyac
> duyulan eklerde en kotu davraniyor."

> "Ortadaki kesme-d parcasi Turkce'de hicbir anlam tasimiyor;
> Ingilizce'de 'would' ya da 'had' kisaltmasi. Model bu parcayi
> gordugunde Ingilizce bir kalip bekliyor olabilir."

> "Sayilar da benzer: kural sayilari en fazla uc basamakta kesiyor.
> '2026'da' dort parcaya ayriliyor. Sinifta konusulan 'LLM'ler neden
> sayilarda hata yapiyor' sorusunun kaynaklarindan biri tam olarak bu."

---

## 7 - Turkce ile Ingilizce (12:00-13:30)  <- GOREVDE ISTENEN

[EKRAN] diller.png ve gorev5 ciktisi.

> "Besinci gorev. Ayni icerigin Turkce ve Ingilizce halini aldim ve
> GPT-4'un tokenizer'iyla, cl100k_base ile token'ladim."

> "Turkce: 47 kelime, 148 token.
> Kelime basina 3.149."

> "Ingilizce: 76 kelime, 84 token.
> Kelime basina 1.105."

> "Yani Turkce kelime basina 2.85
> kat daha pahali. Ama dikkat: toplam token farki sadece
> 1.76 kat. Arada ucurum var ve
> sebebi onemli."

> "Uc sebep ust uste biniyor. Birincisi byte maliyeti: Turkce harfler
> iki byte. Ikincisi egitim verisi dengesi: cl100k_base agirlikli olarak
> Ingilizce veriyle egitilmis, bu bir tasarim hatasi degil, verinin
> dogal sonucu."

> "Ucuncusu eklemeli yapi - ve bu ilk ikisini KISMEN dengeliyor.
> Turkce'de bir kelime, Ingilizce'de bes kelimenin isini goruyor.
> 'evlerimizden' esittir 'from our houses'. Turkce kelime basina daha
> cok token harciyor ama ayni anlami daha AZ kelimeyle anlatiyor. Kelime
> basina 2.85 kat olan
> farkin toplamda 1.76 kata
> inmesinin sebebi bu."

[EKRAN] tokenizerlar.png

> "Son olarak ayni Turkce paragrafi kendi tokenizer'imla token'ladim."

> "Benim vocab'im 1024, GPT-4'unki 100.256. Yuz kat fark. Ben kendi
> tokenizer'imin bariz onde olmasini bekliyordum, cunku butun
> birlestirme butcesini Turkce'ye harcadi."

> "Sonuc: esit. Ikisi de 148 token."

> "Sebebi bence su: GPT-4'un vocab'i yuz kat buyuk oldugu icin
> Turkce'ye ayirdigi KUCUK pay bile mutlak olarak benim butun vocab'im
> kadar. Yuzde olarak dezavantajli, sayi olarak degil."

> "Dogru okuma: bir megabayt Turkce metinle egitilmis 1024 token'lik
> bir tokenizer, terabyte'larca cok dilli veriyle egitilmise Turkce'de
> YETISIYOR. Gecmiyor. Ve takas benim aleyhime: benimki Ingilizce'de
> kotu olurdu."

---

## 8 - Ek calisma (13:30-15:00)

[EKRAN] gorev6a ciktisi.

> "Gorev alti istege bagliydi, iki secenekten biri. Ikisini de yaptim."

> "Birincisi: GPT-4'un birlestirmelerini yukleyip kendi kodumun tiktoken
> ile ayni ciktiyi verdigini gostermek. Iki puruz vardi."

> "Birincisi, tiktoken bana birlestirme AGACINI vermiyor, sadece
> 'token - sira numarasi' veriyor. Agaci geri kurmam gerekiyordu. Ilk
> denememde 'token'i ilk gecerli yerden ikiye bol' dedim. KeyError ile
> coktu. Sebep: bir token birden fazla yerden bolunebiliyor ve dogrusu
> her zaman ilki degil. Dogrusu, her token'i byte'larina ayirip SADECE o
> token'dan once gelmis birlestirmeleri uygulamak. Yine sira meselesi -
> bu hafta ayni dersi iki kere ogrendim."

> "Ikinci puruz: GPT-4 taban byte'larini karistiriyor. Encode'dan once
> cevirip decode'da geri aliyorum."

> "Sonuc: 13 elle secilen ornek, 40.000
> karakterlik gercek metin ve 300 rastgele metin. Hicbirinde fark yok."

> "Bu benim icin onemliydi, cunku kodun dogrulugunu kendi testimle degil
> BAGIMSIZ bir referansla olcmus oldum. Hafta 5'te manuel geri yayilimi
> PyTorch'un gradyanlariyla karsilastirmistim; mantik ayni."

[EKRAN] uretilen metin, yan yana.

> "Ikincisi: Hafta 8'deki GPT'yi kendi tokenizer'imla yeniden egittim.
> Mimari bire bir ayni, tek degisen veri katmani."

> "Solda Hafta 8'in harf modeli, sagda bu haftanin BPE modeli. Fark
> olculen loss farkindan cok daha carpici: BPE modeli gercek Turkce
> kelimeler ve dogru ekler uretiyor. Sebebi basit - artik harf harf
> degil, hece ve ek olarak seciyor."

> "Son soru: bu iki val loss'u dogrudan karsilastirabilir miyim? Hayir.
> Olcu birimi farkli - Hafta 8'de token esittir karakterdi, simdi token
> yaklasik 2.45 karakter. Secenek sayisi
> farkli. Ve veri bolmesi farkli - Hafta 8'de korpusu sirayla bolmustum
> ve val train'den kolay cikmisti, o haftanin raporunda kendi hatam
> olarak yazmistim. Bu hafta duzelttim."

> "Adil olcut ikisini de ayni birime cevirmek: karakter basina bit.
> Hafta 8 2.2854, Hafta 9 2.1784."

> "Bu, Hafta 6, 7 ve 8'de karsima cikan dersin dorduncusu: bir sayiyi
> sadece olcum kosullari ayniysa karsilastirabilirsin."

---

## 9 - Kapanis (15:00-15:45)

[KAMERA] Yuz.

> "Toparliyorum. Bu hafta ogrendigim en onemli sey su: tokenizer modelin
> bir parcasi degil, AYRI bir asama. Sekiz hafta boyunca modelin
> icindeki her seyi yazdim; bu hafta disindaki tek parcayi yazdim. Ve
> modelin tuhaf davranislarinin cogunun kaynagi burasi."

> "Turkce acisindan somut bir sonuc cikti: GPT-4'un deseni, denedigim
> yirmi alti ekin on ikisini bozuyor ve bozulanlar Turkce'nin en sik
> kullanilan ekleri. 'LLM'ler neden Turkce'de daha pahali ve daha kotu
> calisiyor' sorusunun olculebilir bir parcasi."

> "Durust olmak icin: bu hafta iki tahminim olcumle curudu ve bir kere
> de kod hatasi yapip coktum. Ucunu de rapora koydum. Ayrica tiktoken
> normalde birlestirme tablosunu OpenAI'in sunucusundan indiriyor;
> calistirdigim ortamdan o adrese cikis kapaliydi, ayni tabloyu baska
> bir paketten alip elle verdim. Kodda acikca yazdim."

> "Hala merak ettiklerim var: deseni Turkce ekleriyle genisletsem kac
> token kazanirdim, vocab 512 ya da 2048 olsa model nasil olurdu,
> sentencepiece ne yapiyor."

> "Kod, rapor ve butun ciktilar repoda. Bu isi gercekten ogrenmek
> istiyorum ve calismaya devam ediyorum. Tesekkurler."

---

## Cekimden once kontrol listesi

- [ ] gorev2_bpe.py ciktisi terminalde hazir (ilk 20 birlestirme tablosu)
- [ ] gorev4 ciktisi hazir (26 ek testi gorunur)
- [ ] gorev5 ciktisi hazir (TR/EN tablosu)
- [ ] Grafikler acik: ilk20.png, sikistirma.png, vocab_yapisi.png,
      ekler.png, diller.png, tokenizerlar.png, model_bpc.png
- [ ] bpe.py acik - istatistik, birlestir, encode, decode gorunur
- [ ] Hafta 8 ve Hafta 9 ciktilari yan yana kopyalanmis
- [ ] Yazi boyutu buyutulmus
- [ ] Mikrofon testi
