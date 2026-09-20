# Hafta 5 Video Anlatim Metni (6-7 dakika)

Mailde istenen: "en cok zorlandigin turevi ve nasil dogruladigini anlat."
Video bu sorunun etrafinda kuruldu.

Koseli parantezler yonerge, sesli okunmaz.

---

## 1. Acilis (0:00 - 0:40)

[KAMERA]

"Merhaba, ben Neco. YZ50 besinci hafta teslimim.

Bu hafta tek bir sey yaptim: gecen haftaki modelin butun gradient'lerini
loss.backward() kullanmadan elle hesapladim.

Gecen hafta o satiri yazip geciyordum. Calisiyordu, sonuc dogruydu, ama
icinde ne oldugunu bilmiyordum. Bu hafta ogrendim."

[EKRAN PAYLASIMINI BASLAT]

---

## 2. Yontem (0:40 - 1:40)

[EKRAN: gorev1_kurulum.py'nin ileri gecis kismini goster]

"Once modeli parcaladim. Normalde tek satirda yazdigimiz seyleri ayri
ayri degiskenlere ayirdim. Mesela cross entropy normalde tek satir:
F.cross_entropy. Burada sekiz adima bolundu: logits, logit_maxes,
norm_logits, counts, counts_sum, counts_sum_inv, probs, logprobs.

Neden? Cunku her adimin turevini ayri ayri yazip kontrol edebileyim diye."

[EKRAN: ara degiskenler tablosunu goster]

"Toplam 19 ara degisken var. PyTorch bunlarin hepsinin gradient'ini
hesapliyor. Benim isim ayni sayilari elle uretmek."

[EKRAN: cmp fonksiyonunu goster]

"Ve su fonksiyon hakem: benim hesabimla PyTorch'unkini karsilastiriyor.
EXACT derse tam ayni, approx derse fark on uzeri eksi dokuz seviyesinde,
HATA derse yanlis yapmisim."

---

## 3. Uc kural (1:40 - 3:20)

[EKRAN: gorev2_elle.py, probs adimini goster]

"Ilerledikce uc kural ogrendim, hepsi tekrar tekrar karsima cikti.

Birincisi broadcasting. Bakin burada counts'un sekli otuz iki carpi
yirmi yedi, counts_sum_inv'in sekli otuz iki carpi bir. Ileri geciste
o tek sutun yirmi yedi sutuna kopyalaniyor.

Geri geciste ne olmali? O yirmi yedi sutundan gelen gradient'lerin
hepsi toplanmali. Yani sum almam lazim.

Ilk yazdigimda sum'i unuttum ve sekil hatasi aldim. Iyi ki hata verdi,
sessizce gecseydi hic bulamazdim."

[EKRAN: dcounts'un iki katkisini goster, imlecle '+=' isaret et]

"Ikinci kural: bir degisken iki yerde kullanildiysa gradient'ler
toplanir. Bakin counts hem probs'ta hem counts_sum'da geciyor. Birinci
katkiyi yaziyorum, sonra arti esittir ile ikincisini ekliyorum.

Ve bu bana Hafta 2'yi hatirlatti. O zaman micrograd'da self.grad arti
esittir yazmistim, ayni kural. Tek fark o zaman tek bir sayidaydi,
simdi koca bir matriste."

[EKRAN: logit_maxes ve counts_sum satirlarini yan yana goster]

"Ucuncu kural: toplam ve maksimum farkli davraniyor. Toplamda gradient
butun elemanlara esit dagiliyor. Maksimumda ise sadece kazanan elemana
gidiyor, digerlerine sifir. Onun icin one_hot kullandim."

---

## 4. EN COK ZORLANDIGIM TUREV (3:20 - 5:00)

[EKRAN: bndiff satirlarini goster - bu bolum mailde ozellikle isteniyor]

"Simdi en cok zorlandigim yere geliyorum: bndiff.

bndiff, BatchNorm'un icinde bir ara degisken. Ve iki yerde kullaniliyor.
Birincisi hemen bir sonraki satirda: bnraw esittir bndiff carpi
bnvar_inv. Ikincisi ise dort adim ilerde: bndiff2 esittir bndiff kare.

Ben ilk yazdigimda sadece birinci katkiyi yazdim. cmp calistirdim ve
HATA dedi."

[EKRAN: cmp ciktisinda HATA satirini goster - ya da o hali anlat]

"Ve fark buyuktu, on uzeri eksi iki seviyesinde. Bu onemli bir detay:
eger fark on uzeri eksi dokuz olsaydi yuvarlama hatasi derdim ve
gecerdim. Ama on uzeri eksi iki demek gercek bir eksiklik var demek."

[EKRAN: kagida cizilmis grafi goster, ya da anlat]

"Nasil buldum? Grafi kagida cizdim. bndiff'i yazdim ve ondan cikan
oklari takip ettim. Iki ok cikiyordu. Bir degiskenden kac ok cikiyorsa
o kadar katki var, ben sadece birini yazmisim.

Ikinci katkiyi ekleyince cmp approx dedi, fark on uzeri eksi ona indi.

Buradan cikardigim ders su: gradient'i yazarken 'bu degisken nereye
gidiyor' diye sormak yetmez, 'kac yere gidiyor' diye sormak lazim.
Ve ayni hatayi hprebn'de ve logits'te de yapabilirdim, cunku onlar da
iki yerde kullaniliyor."

[EKRAN: cmp ciktisinin tamamini goster]

"Sonunda yirmi alti gradient'in hepsi gecti. On bir tanesi EXACT, on
bes tanesi approx. approx olanlarin hepsi BatchNorm zincirinde. Sebep
hata degil: PyTorch islemleri farkli sirada yapiyor ve float
aritmetiginde a arti b arti c ile c arti b arti a mikroskobik farkli
cikabiliyor."

---

## 5. Ek gorev: tek satira indirmek (5:00 - 6:20)

[EKRAN: gorev3_tek_satir.py, egzersiz 2 kismini goster]

"Ek gorevi de yaptim. Cross entropy'nin sekiz adimlik zincirini kagitta
sadelestirdim ve su tek satira indi:

dlogits esittir softmax logits eksi one_hot Y, bolu n.

Fark on uzeri eksi dokuz, yani ayni.

Ve anlami cok guzel: gradient esittir modelin tahmini eksi gercek cevap.
Model dogru harfe yuzde yuz dediyse gradient sifir, hic duzeltme yok.
Yanlis harfe yuksek olasilik verdiyse o harfin gradient'i pozitif, yani
'bunu azalt' diyor."

[EKRAN: BatchNorm tek satirini goster]

"BatchNorm'un yedi adimlik zinciri de tek satira indi ama daha karisik.
Uc terim var: dogrudan etki, ortalamayi degistirdigin icin geri odeme,
varyansi degistirdigin icin geri odeme.

Ve bu bana gecen haftadan kalan bir soruyu cevapladi. Gecen hafta
'BatchNorm'da bir ornegin ciktisi ayni batch'teki digerlerine bagli,
tuhaf ama ise yariyor' demistim. Iste bu iki ek terim tam olarak o
baglantinin matematigi."

[EKRAN: egzersiz 4, egitim ciktisini goster]

"Son olarak modeli yirmi bin adim egittim, hicbir yerde loss.backward
yok. Butun gradient'ler benim. dev loss 2.23 cikti, gecen haftaki
autograd'li versiyon 2.15'ti. Model biraz daha kucuk oldugu icin
farkli, ama calistigini gosteriyor."

[EKRAN: uretilen isimleri goster]

"Urettigi isimler: carmah, amille, reety, ner, kiah. Gecen haftakiler
gibi makul isimler."

---

## 6. Kapanis (6:20 - 6:50)

[KAMERA]

"Toparlarsam, bu haftanin bende biraktigi sey su:

Hafta 2'de micrograd'da backward yazmistim ama orada her sey tek bir
sayiydi. Bu hafta ayni isi matrislerle yaptim ve fark ettim ki asil
zorluk turev formullerinde degil, sekillerde. Broadcasting nerede
toplaniyor, bir degisken kac yerde kullaniliyor, bunlar.

Ve artik loss.backward yazdigimda arkada ne oldugunu biliyorum, cunku
bir kez kendim yazdim.

En cok zorlandigim yer bndiff'in ikinci katkisiydi, kagida cizerek
buldum. Tesekkurler."
