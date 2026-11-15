# Hafta 7 - Deney Notu

## Gorev 1 - Veri ve bigram tabani

Tiny Shakespeare: 1.115.394 karakter, 65 farkli karakter.
%90 train, %10 val.

**Karakter tokenizer.** Alfabe kucuk (65) ama diziler uzun. GPT gibi
modeller kelime parcasi (BPE) kullaniyor: alfabe ~50.000, diziler
kisa. Takas bu; biz basit olani sectik.

**block_size = 8 ne demek.** Tek bir 8'lik parca aslinda 8 ayri
egitim ornegi iceriyor:

    [18]                      -> 47
    [18, 47]                  -> 56
    [18, 47, 56]              -> 57
    ...
    [18, 47, 56, 57, 58, 1, 15, 47] -> 58

Model 1'den 8'e kadar her uzunlukta baglam gormeyi ogreniyor. Uretim
sirasinda ilk harften baslayacagi icin bu gerekli.

**Batch.** xb sekli (32, 8): 32 bagimsiz parca, her biri 8 karakter.

**Bigram tabani.** nn.Embedding(65, 65), 4.225 parametre.

| | deger |
|---|---|
| egitimsiz beklenen loss | -log(1/65) = 4.1744 |
| olculen ilk loss | 4.7313 |
| 5000 adim sonra (lr=1e-2) | train 2.4573, val 2.4769 |

Ilk loss beklenenden yuksek cikti cunku embedding tablosu rastgele
baslatiliyor; Hafta 4'te ogrendigim "son katmani kucult" numarasi
burada uygulanmadi (Karpathy de uygulamiyor, model zaten kucuk).

## Gorev 2 - Gecmisin ortalamasi, uc yol

Hedef: her t adiminda 0..t arasindaki token'larin ortalamasi.

| yol | nasil |
|---|---|
| 1 | ic ice for dongusu, x[b,:t+1].mean(0) |
| 2 | tril matrisini satir toplamina bolup x ile carpmak |
| 3 | sifirlardan baslayip gelecegi -inf yapmak, softmax almak |

**Dogrulama.**

    torch.allclose(yol1, yol2)            : False
    torch.allclose(yol1, yol3)            : False
    torch.allclose(yol1, yol3, atol=1e-6) : True
    maksimum fark                         : 3.24e-08

Varsayilan allclose'un False donmesi hata degil: varsayilan atol
1e-8 ve farkimiz 3.2e-08. Hafta 5'teki EXACT / approx ayriminin
aynisi, buradaki durum "approx".

**Matris carpimi neden agirlikli ortalama - kendi cumlelerimle.**

wei @ x isleminde sonucun t. satiri soyle hesaplaniyor: wei'nin
t. SATIRINI al, x'in butun satirlariyla carp, topla.

    sonuc[t] = wei[t,0]*x[0] + wei[t,1]*x[1] + ... + wei[t,T-1]*x[T-1]

Bu tam olarak agirlikli ortalamanin tanimi. wei[t,j] = "t. token
j. token'a ne kadar baksin".

tril'in isi: ust ucgen sifir oldugu icin gelecekteki token'larin
agirligi sifir. Satir toplamina bolmek de agirliklarin toplamini 1
yapiyor, boylece toplam degil ortalama oluyor.

Asil nokta: su an butun agirliklar esit (1/3, 1/3, 1/3). Yani
"gecmise bak ama hepsine ayni derecede bak". Self-attention bu
agirliklari VERIYE BAKARAK belirliyor. Softmax versiyonunun onemi
de bu: formul hic degismiyor, sadece sifirlarin yerine ogrenilen
skorlar geliyor.

## Gorev 3 - Tek head

**q, k, v.** Her token uc sey uretiyor: query ("ne ariyorum"),
key ("bende ne var"), value ("bana bakarsan ne veririm").
t'nin j'ye agirligi = q[t] ile k[j]'nin ic carpimi.

Analoji: kutuphanede arama. Elinde arama cumlesi var (query), her
kitabin sirtinda etiket var (key). Etiket aramanla ne kadar
ortusuyorsa o kitaba o kadar agirlik veriyorsun, sonra agirliklara
gore kitaplarin ICERIGINI (value) karistirip tek ozet cikariyorsun.

Sekiller: x (4,8,32) -> q,k,v (4,8,16) -> wei (4,8,8) -> cikti (4,8,16).

**Bir wei satirini okumak.** 4. satir, 8 token'lik bir parcada:

    token 0: 0.148   gecmis
    token 1: 0.203   gecmis
    token 2: 0.166   gecmis
    token 3: 0.146   gecmis
    token 4: 0.337   kendisi
    token 5-7: 0.000 gelecek, tril ile kapali

Satir toplami 1.0000 (softmax). Agirliklar esit degil: duz ortalama
olsaydi hepsi 0.200 olurdu. Bu ornekte head egitilmemis (rastgele
baslatilmis) oldugu icin secim rastgele; anlamli oruntu Gorev 5a'daki
egitilmis modelde goruluyor. Ama mekanizma calisiyor: son uc agirlik
tam sifir cunku onlar gelecek ve -inf ile maskelendi.

**Neden sqrt(head_size)'a boluyoruz.**

| | varyans |
|---|---|
| olceklenmemis wei | 15.574 |
| olceklenmis wei | 0.973 |

Sebep: ic carpim head_size (16) terimin toplami. Bagimsiz birim
varyansli 16 sayiyi toplayinca varyans 16 oluyor.

Neden sorun: softmax'a buyuk sayilar girince dagilim keskinlesiyor.

    skorlar [0.1, -0.2, 0.3, -0.2, 0.5]
      softmax: [0.192, 0.143, 0.235, 0.143, 0.287]
    skorlar x 8
      softmax: [0.033, 0.003, 0.162, 0.003, 0.800]

Ikincisinde model neredeyse tek bir token'i seciyor, agirlikli
ortalama olmaktan cikiyor. Baslangicta bu kotu: model hicbir sey
ogrenmeden rastgele bir token'a kilitleniyor ve gradient
digerlerine akmiyor. Hafta 4'teki tanh doymasiyla ayni mantik.

Gercek wei'nin 4. satirinda olculen:

    bolmeden : maksimum agirlik 0.791
    bolerek  : maksimum agirlik 0.337

**Pozisyon embedding'i.** wei[t,j] sadece q[t] ile k[j]'nin ic
carpimi; j'nin kacinci token oldugu hesaba hic girmiyor. Test
ettim: bir token'in yerini degistirdim, skor aynen kaldi
(-0.8306 -> -0.8306). Yani attention bir torba islemi.

Cozum: x = token_embedding(idx) + position_embedding(arange(T)).

Hafta 6'daki WaveNet'te buna gerek yoktu, cunku orada sira agacin
yapisinda kodluydu.

## Gorev 4 - Head'i modele koy

| model | parametre | train | val |
|---|---|---|---|
| bigram | 4.225 | 2.4573 | 2.4769 |
| tek head | 7.553 | 2.3882 | 2.3998 |
| iyilesme | | | 0.0771 |

**Neden dustu - kendi cumlelerimle.** Bigram sadece bir onceki harfe
bakiyordu. Tek head sekiz harfe birden bakabiliyor VE hangisine ne
kadar bakacagini kendi seciyor.

Hafta 6'daki WaveNet de sekiz harfe bakiyordu ama birlestirme sekli
sabitti: hep ikiser ikiser, hep ayni agacla. Burada agirliklar veriye
gore degisiyor.

Ikinci fark pozisyon embedding'i: model artik "ben bu parcanin
3. karakteriyim" bilgisini de tasiyor.

**Neden son block_size harfe kirpmak zorundayiz.**
position_embedding_table'in tam olarak block_size (8) satiri var.
9 karakterlik girdi verirsem arange(9) tabloda olmayan 8. satiri
arar ve IndexError alirim. Bu yuzden generate icinde
idx[:, -block_size:] yaziyorum.

Bedeli: model 8 karakterden eskisini unutuyor. GPT'nin "context
window" dedigi sinir tam olarak bu.

## Gorev 5a (ek) - Isi haritasi

Egitilmis modelde "First Ci" cumlesinin wei matrisi
(attention_isi.png):

| token | en cok baktigi | agirlik |
|---|---|---|
| 's' (3.) | kendisi | 0.907 |
| 't' (4.) | 's' (bir onceki) | 0.637 |
| ' ' (5.) | kendisi | 0.967 |
| 'C' (6.) | ' ' (bir onceki) | 0.561 |
| 'i' (7.) | kendisi | 0.964 |

**Ne gordum.** Head cogunlukla YEREL bir oruntu ogrenmis: her token
ya kendisine ya bir oncekine bakiyor. Uzak token'larin agirliklari
0.01-0.02 seviyesinde.

Bu, 0.077'lik mutevazi kazanci acikliyor. Tek head ve 8 karakterlik
baglamla model esasen "biraz zenginlestirilmis bigram" ogrenmis.
Gercek kazanc coklu head ve derinlik gelince cikacak (onumuzdeki
hafta).

Yine de bir sey var: 'F' ve 'i' gibi ilk token'larda dagilim daha
duz (0.41 / 0.59), yani baglam kisa oldugunda model gercekten
ortalama aliyor.

## Gorev 5b (ek) - tril maskesi kaldirilirsa

| model | val loss |
|---|---|
| maskeli | 2.3998 |
| maskesiz | 0.3955 |

Val loss 6 kat dustu. Ama uretilen metin:

    SSSSSSSSSSSSSSSSSSSSSSSSSS...
    Seeeeeeeeeeeeeeeeeeeeeeeeee...

**Aciklama.** 7. token'in hedefi 8. karakter ve maskesiz modelde
7. token butun parcayi, yani 8. karakteri de goruyor. Cevabi gormek
kolay, loss dusuyor. Ama bu loss yalan: model sinavda kopya cekiyor.

Uretim zamaninda gelecek diye bir sey yok. Model egitimde alistigi
bilgiden yoksun kalinca cokuyor.

Hafta 6'daki train/dev/test dersinin baska bir hali: olcumu bozan
bir sizinti varsa dusen loss ilerleme anlamina gelmiyor.

"Decoder-only" mimari denmesinin sebebi bu maske. Kaldirilirsa model
dil modeli olmaktan cikip BERT tarzi bir encoder'a donusuyor; o da
uretmek icin degil anlamak icin kullaniliyor.

## Zorlandigim yerler

- wei @ v isleminde seklin neden (B,T,head_size) oldugunu oturtmak.
  (B,T,T) @ (B,T,hs): son iki eksen matris carpimi, bastaki B toplu
  islem. Hafta 6'da ogrendigim seyin aynisi.
- tril'i register_buffer ile kaydetmek gerektigini anlamak. Parametre
  degil ama modelle birlikte tasinmasi gereken bir tensor.
- Maskesiz modelin val loss'unun DUSMESI once beni sasirtti; bug
  sandim. Urettigi metne bakinca anladim.

## Acik sorular

1. Tek head yerel bir oruntu ogrendi. Coklu head'de farkli head'ler
   farkli mesafelere mi bakacak?
2. block_size'i 8'den 32'ye cikarsam isi haritasi degisir mi, uzak
   baglantilar olusur mu?
3. Pozisyon embedding'i ogreniliyor. Sinus/kosinus tabanli sabit
   pozisyon kodlamasiyla fark ne olurdu?
