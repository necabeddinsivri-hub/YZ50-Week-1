# Video metni - YZ50 Hafta 10

**Sure hedefi:** 12-14 dakika. Ekran paylasimi + sesli anlatim.
**Gorevde istenen:** kendi modelimin urettigi metni ve tek batch'i
ezberletirken loss'un nasil dustugunu goster. Ikisi de videonun
merkezinde.
**Acik olacak:** gpt2.py, gorev4 ciktisi, grafikler, gorev3 ciktisi.

---

## 0 - Acilis (0:00-0:50)

[KAMERA] Yuz.

> "Merhaba, ben Necabeddin. YZ50'nin onuncu haftasi. Dokuz haftadir kendi
> kucuk modellerimi yazdim; bu hafta OpenAI'in 2019'da yayinladigi
> GPT-2'yi yazdim."

> "Bastan soyleyeyim: Hafta 8'de yazdigim GPT ile GPT-2 arasinda mimari
> fark neredeyse yok. Ayni attention, ayni blok, ayni residual. Fark
> olcekte - 117 kat. Ve bu haftanin asil dersi tam da burada: bu olcekte
> mimariyi yazmak kolay, DOGRU yazdigini kanitlamak zor."

> "Iki kontrol kullandim. Ve ikisi de benim kendi kodumdaki hatalari
> yakaladi. Onlari da anlatacagim."

---

## 1 - Ne degisti (0:50-2:30)

[EKRAN] gpt2.py, GPTConfig ve CausalSelfAttention.

> "Once olcek. Hafta 8'de n_embd 128'di, simdi 768. 4 blok yerine 12.
> block_size 64 yerine 1024. vocab 1024 yerine 50.257. Parametre 1 milyon
> yerine 124 milyon."

> "Mimaride degisen alti sey var ve hicbiri 'daha iyi mimari' degil.
> Hepsi, OpenAI'in agirliklarini yukleyebilmek icin kodumun onlarinkiyle
> birebir ayni dizilmis olmasi gerektigi icin."

> "Bir: isimler. transformer.wte, wpe, h, ln_f. Kozmetik gorunuyor ama
> degil - state_dict anahtarlari bire bir tutmazsa agirliklar
> yuklenmiyor."

> "Iki: q, k ve v tek bir Linear'da hesaplaniyor, c_attn. Bunu gecen
> hafta Gorev 5a'da zaten yazmistim ve dongulu surumle farkin TAM SIFIR
> oldugunu gostermistim. Yani matematik ayni, hesap tek matmul'de
> toplaniyor."

> "Uc: ReLU yerine GELU. ReLU negatif tarafta gradyani tam sifirliyor ve
> noron 'oluyor'; GELU kucuk ama sifirdan farkli bir deger birakiyor."

> "Dort: lm_head'de bias yok. Bes: weight tying - giris embedding'i ile
> cikis katmani ayni tensor. Alti: LayerNorm adlari ln_1 ve ln_2."

---

## 2 - Kontrol 1: ilk loss (2:30-5:30)  <- BIRINCI HATAM

[EKRAN] gorev4 ciktisi, 4a bolumu.

> "Simdi birinci kontrol. Modeli rastgele baslatip ilk loss'u olcuyorum."

> "Beklenen deger ne olmali? Model hicbir sey ogrenmemisse 50.257
> token'in hepsini esit olasi gormeli, yani her birine 1 bolu 50.257
> vermeli. Bunun eksi logaritmasi ln(50257) = 10.8249."

> "Bu cok basit bir sayi ama bir HATA DETEKTORU. Loss cok daha BUYUK
> cikarsa baslangic agirliklari fazla buyuk demektir - model daha ilk
> adimda emin ama yanlis. Hafta 4'te tanh'in doymasini incelerken
> gordugum seyin aynisi. Loss cok daha KUCUK cikarsa bir yerde sizinti
> var demektir - Hafta 7'de tril maskesini kaldirinca oyle olmustu."

> "Benim durumumda ikincisi oldu."

[EKRAN] 4a-ek tablosu.

> "Ilk denememde 9.99 cikti. Beklenenin 0.84
> ALTINDA. Ustune gittim ve sunu buldum: testi yazarken hedef olarak
> girdinin KENDISINI vermisim. Oysa dil modelinde hedef, girdinin bir
> token kaydirilmis hali olmali."

> "Ama burada ilginc bir soru var: neden loss DUSTU? Hedef yanlis olsa
> loss'un yukselmesi gerekmez miydi? Uc durumu ayri olctum."

> "Weight tying varken, hedef girdiyle ayni oldugunda 9.99.
> Bagimsiz rastgele hedefle 11.00. Gercek metinle 10.96."

> "Weight tying YOKKEN ayni test 11.02 veriyor. Etki tamamen
> kayboluyor."

> "Sebep weight tying. Giris embedding'i ile cikis katmani ayni tensor.
> Yani bir token'in logit'i, o token'in embedding vektoru ile son
> katmanin ciktisinin ic carpimi. Model egitilmedigi icin ciktisi buyuk
> olcude girdi embedding'inin kendisine benziyor - ve bir vektorun
> kendisiyle ic carpimi, baska bir vektorle ic carpimindan buyuktur."

> "Yani egitilmemis model 'bir sonraki token girdinin kendisidir'
> tahminine dogal bir egilimle basliyor. Hedefi girdinin kendisi yapinca
> bu egilimi odullendirmis oldum."

> "Iki ders cikardim. Birincisi, ln(V) kontrolu gercekten ise yariyor -
> ve benim durumumda modeldeki degil, kendi test kodumdaki hatayi
> soyledi. Ikincisi, weight tying masum bir parametre tasarrufu degil;
> modelin baslangictaki davranisini da degistiriyor."

[EKRAN] duzeltilmis tablo.

> "Duzeltince uc sema da 10.8'in hemen uzerinde cikiyor. Model dogru
> kurulmus."

---

## 3 - Kontrol 2: tek batch'i ezberletme (5:30-8:00)  <- GOREVDE ISTENEN

[EKRAN] ezber.png ve gorev4 ciktisi.

> "Ikinci kontrol. Tek bir batch'i - 128
> token - 300 kez ust uste gosteriyorum. Hep ayni veri."

> "Model genellemek zorunda degil, sadece ezberlemesi yeterli. 124 milyon
> parametreli bir model icin bu cok kolay bir is. Dolayisiyla loss SIFIRA
> inmek ZORUNDA."

> "Inmiyorsa ortada bir ogrenme sorunu degil, bir KOD sorunu vardir.
> Gradyanlar akmiyordur, bir yerde detach ya da no_grad kalmistir.
> Hedefler bir adim kaymistir. Optimizer parametreleri gormuyordur. Ya da
> learning rate tamamen yanlistir."

> "Yani bu test modelin ne kadar iyi oldugunu olcmuyor; makinenin takili
> olup olmadigini olcuyor. Hafta 5'te manuel geri yayilimi PyTorch'un
> gradyanlariyla karsilastirmistim - mantik ayni: once dogrulugu kanitla,
> sonra performansa bak."

[EKRAN] loss egrisi, once dogrusal sonra log olcek.

> "Sonuc: 10.9600'dan
> 0.0008'e. Soldaki grafik dogrusal olcekte, ilk
> elli adimda dikey bir dusus goruyorsunuz. Sagdaki log olcekte ve orada
> dususun devam ettigi gorunuyor."

> "Ayni batch'te token dogrulugu %100.0. Yani
> 128 token'in hepsini dogru biliyor."

> "Ileri gecis, geri yayilim ve optimizer dogru bagli. Gelecek hafta
> gercek egitimde loss inmezse sebebi kod olmayacak."

---

## 4 - Weight tying ve init (8:00-10:00)

[EKRAN] tying.png

> "Ek gorevin ilk secenegine gectim: data loader, weight tying ve
> baslangic degerleri."

> "Weight tying'in ne kazandirdigini hesapladim. Tying yokken
> 163.037.184 parametre, varken 124.439.808.
> Kazanc 38.597.376."

> "Bu sayi tesaduf degil: tam olarak embedding tablosunun boyutu, 50.257
> carpi 768. GPT-2'nin '124M' olarak anilmasinin sebebi weight tying;
> olmasaydi 163 milyon olacakti."

[EKRAN] init semasi tablosu.

> "Baslangic degerleri. Uc kural basit: butun Linear ve Embedding
> agirliklarina std 0.02, bias'lara sifir. Dorduncusu daha ilginc:
> residual yoluna EKLENEN katmanlarin std'si ayrica 1 bolu karekok
> 2n_layer ile carpiliyor."

> "Neden? Her blokta residual'a iki ekleme var - attention ve MLP. 12
> blok boyunca 24 bagimsiz terim toplaniyor. Bagimsiz degiskenlerin
> toplaminda varyanslar TOPLANIR, yani cikisin varyansi 24 katina cikar.
> 1 bolu karekok 24 ile olcekleyince sabit kaliyor."

> "Bu, Hafta 8'de residual'i eklerken 'toplama dugumunun gradyani 1'
> diye yazdigim seyin ILERI yondeki karsiligi. Orada gradyanin
> bolunmeden gectigini soylemistim; burada ayni toplamanin ileri yonde
> varyansi buyuttugunu goruyorum."

> "Ve burada ikinci hatam cikti. Olcumu ilk yaptigimda mlp.c_proj icin
> 0.0047 aldim, beklenen
> 0.0041 idi. Bir an semanin tutmadigini sandim."

> "Sonra fark ettim: olctugum model, az once 120 adim
> EGITTIGIM modeldi. Agirliklar artik baslangic degerlerinde degildi.
> Taze bir model kurunca sema birebir tutuyor."

> "Bu haftanin ikinci olcum hatasi ve ikisi de ayni aileden: bir sayiyi
> olcerken NEYI olctugunden emin olmak gerekiyor. Hafta 6'dan beri her
> hafta baska bir kilikta karsima cikan ayni ders."

---

## 5 - Data loader ve ilk gercek egitim (10:00-11:00)

[EKRAN] DataLoaderLite kodu ve egitim ciktisi.

> "Data loader'i yazdim: metni sirayla, ust uste binmeyen B carpi T'lik
> parcalar halinde okuyor. Hafta 7-8'de get_batch rastgele yerlerden
> cekiyordu; burada her token'in tam bir kez gorulmesini istiyorum."

> "120 adim egittim, loss
> 10.96'dan 5.76'e
> indi. Bu ezberleme degil - her adimda farkli bir batch geliyor."

> "Ama durust olayim: bu dususun buyuk kismi 'ucuz' bir ogrenme. Model
> ilk adimlarda butun 50.257 token'i esit olasi goruyordu; simdi en
> azindan hangilerinin SIK gectigini ogrendi - bosluk, 'the', nokta,
> satir sonu. Hafta 7'deki bigram modelinin yaptigi seyin aynisi."

> "Hiz: 105 token saniye. Karpathy videoda
> A100'de yaklasik 16.000 aliyor - 152
> kat fark ve tamamen donanimdan. Tiny Shakespeare'in tamami
> 338.025 token; bir epoch bu makinede
> yaklasik 0.9 saat
> surerdi. Modeli gercekten egitmek bu haftanin gorevi degil - gelecek
> haftanin konusu."

> "Bu arada bir sinir daha buldum: Gorev 4 ile Gorev 5a'yi ayni anda arka
> planda calistirdim ve Gorev 4 sessizce oldu. Makinede 7 GB RAM var ve
> her kosu model, gradyan ve AdamW'nin iki momentiyle yaklasik 2 GB
> tutuyor. Bu olcekte artik iki sey birden calistiramiyorum."

---

## 6 - Agirliklari yuklemek (11:00-13:00)

[EKRAN] gorev1 ve gorev2 ciktisi.

> "Simdi hazir agirliklara gelelim. 160 tensor,
> 124.439.808 parametre."

> "Dikkatimi ceken ilk sey: dosyada lm_head YOK. Oysa modelin 768
> boyuttan 50.257 skora cikmasi lazim. Sebep weight tying - tek kopya
> saklanmis."

[EKRAN] wpe.png

> "Pozisyon tablosuna baktim. Hafta 7'de bunu kendim eklemistim ve orada
> sadece 'modelin siralamayi bilmesi lazim' diye dusunuyordum. Burada
> gercekten EGITILMIS bir tabloya bakiyorum."

> "Uc sey buldum ve ucu de beklentimden farkliydi."

> "Bir: pozisyon sifir bir aykiri deger. Normu
> 9.88, medyan 3.37 - uc
> kat uzun. Ve sifir ile bir arasindaki benzerlik
> 0.524 iken yedi ile sekiz arasindaki
> 0.998. Yani ilk birkac pozisyon keskin
> ayrilmis, sonrakiler neredeyse ayni. Mantikli - 'metnin ilk token'i'
> ozel bir bilgi."

> "Iki: cozunurluk basta yuksek, ortada dusuk. Ilk elli pozisyonda yuz
> adim otesiyle benzerlik 0.204, ortada
> 0.542. Model basta kacinci token oldugunu ince
> ayrintisiyla ayirt ediyor, ilerledikce kabalasiyor."

> "Uc ve en carpici olani: son pozisyon hic egitilmemis. Pozisyon
> 1022'nin normu 3.86, pozisyon 1023'un normu
> 0.12. Ucurum gibi."

> "Sebebini dusununce buldum: hedef, girdinin bir token kaydirilmis hali.
> 1024 token'lik bir ornekte 1023. pozisyonun tahmin edecegi sey 1025.
> token olurdu - ama o yok. Yani o embedding hicbir zaman gradyan
> almiyor."

> "Ve burada guzel bir simetri var: Gorev 4'te kendi test kodumda tam o
> kaydirmayi atlamistim. Burada ayni kaydirmanin, egitilmis bir modelde
> biraktigi izi goruyorum."

> "Bu arada buraya da yanlis bir tahminle gelmistim: 'tablonun sonu
> basindan daha gurultuludur' diye dusunuyordum. Olcunce tam tersi cikti."

[EKRAN] transpoze tablosu.

> "Agirliklari yuklerken dort tur katmani TRANSPOZE etmem gerekti.
> Sebep: Hugging Face, OpenAI'in TensorFlow kodunu tasirken nn.Linear
> yerine Conv1D adinda kendi sinifini kullandi. O sinif agirligi (giris,
> cikis) seklinde tutuyor, nn.Linear ise (cikis, giris). Ayni matris,
> devrik saklanmis."

> "Burada tehlikeli bir ayrinti var. mlp.c_fc dikdortgen bir matris -
> transpoze unutulursa boyut tutmaz ve kod hata verir. Bu IYI haber.
> Ama attn.c_proj 768 carpi 768, yani KARE. Transpoze etmeden yuklersem
> boyut tutar, kod calisir ve model sessizce sacmalar. Sessiz hata,
> gurultulu hatadan kotudur."

> "O yuzden kendi modelimin logits'ini Hugging Face'in modelininkiyle
> karsilastirdim: en buyuk fark 1.5e-04. Float32'nin
> yuvarlama hatasi mertebesinde. Hafta 5'te kurdugum 'EXACT mi approx mi'
> olcutuyle burada approx, cunku islem sirasi degisiyor."

---

## 7 - Modeli konusturmak (13:00-14:30)  <- GOREVDE ISTENEN

[EKRAN] gorev3 ciktisi.

> "Sampling dongusu Hafta 7'de yazdigimin aynisi, uzerine top-k ekledim:
> her adimda sadece en yuksek olasilikli 50 token arasindan sec."

> "Neden gerekli? Uzun kuyrukta binlerce cok dusuk olasilikli token var.
> Hepsini toplayinca kucumsenmeyecek bir olasilik ediyorlar ve ara sira
> biri secilip metni bozuyor. top-k o kuyrugu kesiyor."

> "Once Ingilizce: 'Hello, I am a language model,'. Iste ciktilar - akici, dilbilgisi
> dogru, anlamli."

> "Simdi Turkce: 'Merhaba, ben bir dil modeliyim,'. Ve cikti bozuk."

> "Neden? Uc sebep var. Birincisi tokenizer: 'Merhaba' tek kelime ama
> GPT-2 onu uce boluyor. Ayni icerik Turkce'de 2.00 kat token
> tutuyor. Ikincisi baglam penceresi - ayni block_size'a daha az Turkce
> metin siginca model daha az sey goruyor."

> "Ucuncusu ve asil sebep: egitim verisi. GPT-2 WebText uzerinde
> egitildi - Reddit baglantilarindan toplanmis, Ingilizce disi icerigin
> ayiklandigi bir veri kumesi. Model Turkce'yi neredeyse hic GORMEDI."

> "Gecen hafta 'tokenizer bir semptom, veri ise sebep' diye yazmistim."

> "Bu hafta onun sayisal kanitini aldim - ama bekledigim gibi degil."

> "Ayni cumlelerin loss'unu olctum: Ingilizce 3.69, Turkce
> 6.98. Egitimsiz taban 10.82. Yani Turkce
> loss, tabanin %64'i - model
> Turkce'de hicbir sey bilmeyenden sadece
> %36 daha iyi."

> "Sebebi ayirmak icin bir test daha kurdum: ayni cumleleri isaretli
> harflerle de olctum. Mantigim suydu - sade yazim token sayisini
> dusurur, loss da duserse sebep tokenizer demektir."

> "Token sayisi gercekten dustu, 163'ten
> 126'ya. Ama loss YUKSELDI: 5.54'ten
> 6.98'e. Yani daha COK token kullanan surum daha iyi."

> "Beklentimin tam tersi. Sebebini soyle anliyorum: internetteki gercek
> Turkce isaretli harflerle yazilir. GPT-2'nin verisinde Turkce cok az
> ama olan kadari dogru yazilmis Turkce. 'Turkiye'nin baskenti' gibi sade
> yazim ise ne Ingilizce ne duzgun Turkce - model icin hic gormedigi bir
> sey."

> "Yani token sayisi loss'u belirleyen sey degil. Belirleyici olan,
> modelin o yaziyi daha once gorup gormedigi. Bu, 'tokenizer semptom,
> veri sebep' cumlesinin bekledigimden daha guclu bir kaniti."

---

## 8 - Kapanis (14:30-15:15)

[KAMERA] Yuz.

> "Toparliyorum. Bu haftanin en onemli dersi su: bu olcekte mimariyi
> yazmak kolay, dogru yazdigini kanitlamak zor. Hafta 8'de 1 milyon
> parametreli bir model yanlis olsa ciktidan anlardim. 124 milyon
> parametreli bir model yanlis kurulduysa yine makul gorunen bir sey
> uretebilir. Bu haftaki kodun yarisi model, yarisi kontrol."

> "Ve bunun en somut ornegi: iki kontrol de benim kendi hatalarimi
> yakaladi. ln(V) kontrolu test kodumdaki hedef-girdi karisikligini, init
> olcumu ise egitilmis bir modeli olctugumu. Ikisi de kod cokmeden gecen
> hatalardi; sadece sayilar beklenenden saptigi icin yakalandi."

> "Ustune iki tahminim de olcumle curudu: wpe tablosunun sonunun
> gurultulu olacagini sanmistim, degilmis - tek bir satir hic
> egitilmemis. Ve sade Turkce yaziminin loss'u dusurecegini sanmistim,
> tersi cikti. Dordunu de rapora koydum, cunku bu hafta ogrendigim sey
> tam olarak bu: bu olcekte tahmin etmek degil, olcmek gerekiyor."

> "Durust olmak icin bir not daha: calistirdigim ortamdan
> huggingface.co'ya cikis kapali. Tokenizer tablosunu gecen haftaki
> yontemle npm'den aldim; agirliklari elle indirip veri klasorune koydum, dosyalar degistirilmedi ve kodda acikca yazili."

> "Hala merak ettiklerim var. GELU'nun tanh yaklasimi ile tam formulu
> arasindaki fark ne kadar? wpe'deki dalgalar gercekten sinus-kosinus'a mi
> yakinsamis - Fourier ile bakmak isterdim. Ve asil merak ettigim:
> modeli gercekten egitmek. Bu makinede bir epoch
> 0.9 saat surerdi.
> Gelecek haftanin konusu."

> "Kod, rapor ve butun ciktilar repoda. Tesekkurler."

---

## Cekimden once kontrol listesi

- [ ] gorev4 ciktisi terminalde hazir (4a tablosu, 4a-ek tablosu,
      ezberleme adimlari gorunur)
- [ ] ezber.png acik - loss egrisi
- [ ] ilk_loss.png ve tying.png acik
- [ ] gpt2.py acik - GPTConfig, CausalSelfAttention, _init_weights gorunur
- [ ] gorev3 ciktisi hazir - Ingilizce ve Turkce ornekler yan yana
- [ ] wpe.png acik
- [ ] Yazi boyutu buyutulmus
- [ ] Mikrofon testi
