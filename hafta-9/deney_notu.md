# Deney notu - Hafta 9

Ne denedim, ne bekledim, ne cikti. Siraya gore.

## 1. Unicode ve UTF-8

Beklenti yok, olcum var: sectigim cumle 52 karakter ama 68 byte. Butun
korpusta fark yuzde
9.1.

Ilginc ayrinti: Turk bayragi emojisi TEK karakter degil, iki "bolgesel
gosterge" harfinin yan yana gelmis hali. Ekranda bir sey gorunuyor ama
Python icin iki karakter ve 8 byte. "Karakter" kavraminin gorundugu kadar
net olmadiginin iyi bir ornegi.

## 2. BPE egitimi  <- bu haftanin en guzel suprizi

**Bekledigim:** ilk birlestirme sik bir harf IKILISI olur - 'la', 'er',
'in' gibi.

**Cikan:** `ı` - tek bir harf.

Turkce'nin noktasiz i'si UTF-8'de iki byte (196, 177) ve metnimde
37.557 kere geciyor. BPE byte seviyesinde calistigi
icin bu harfi "iki byte'lik bir ikili" olarak goruyor ve ilk islerden biri
olarak birlestiriyor. Ayni sey u (4.), s (8.), c (13.), g (15.) icin de
oluyor.

Yani algoritma, kimse ona Turkce ogretmeden, once TURKCE ALFABESINI
yeniden kesfediyor. Sonra ekler geliyor: 'ar', 'er', 'an', 'in', 'lar'.

**Kucuk ornekte gordugum ikinci sey:** ara adimlarda yarim karakterler
olusuyor. 'agacagacagac' ornegimde ilk birlestirme 'a' + g'nin ilk
byte'i, yani tek basina gecerli UTF-8 olmayan bir token. Bu, Gorev 3'te
decode'da errors='replace' kullanmamin sebeplerinden biri.

**Sikistirma:** 2.639x. Kazanc dogrusal degil - ilk 50
birlestirme 1.52x'e cikariyor, kalan 718 birlestirme ancak
2.639x'e tasiyor.

## 3. encode / decode

`decode(encode(x)) == x` butun orneklerde tamam. En cok sevdigim sonuc:
emoji de calisiyor, halbuki tokenizer'i sadece Turkce metinde egittim ve
egitim verimde hic emoji yoktu. Taban 256 byte oldugu icin her UTF-8
metin temsil edilebiliyor.

Ters yon calismiyor (146/200 rastgele dizi geri donmedi) ve
donmesi de gerekmiyor.

**Gecersiz UTF-8:** strict / replace / ignore uc secenegi denedim.
replace sectim. strict degil cunku model uretimde yarim karakter
uretebilir ve program cokerse uretim durur. ignore degil cunku sessizce
siliyor - bir sey ters gittiginde haberim olmuyor.

## 4. GPT-4'un regex deseni  <- iki tahminim de tutmadi

**Tahmin 1:** "Regex'li surumde bir token ya harf dizisi ya noktalama
dizisi olur."

**Olcum:** regex'li surumde BASINDA bosluk olan token sayisi daha FAZLA
(322 vs 30). Desene tekrar baktim:
`[^\r\n\p{L}\p{N}]?+\p{L}+` kurali basta bir harf-disi karakter
olmasina BILEREK izin veriyor. GPT ciktilarinda token'larin ' the',
' and' diye gorunmesinin sebebi bu.

Gercek fark sonda: sonu bosluklu token 243 -> 0.
TEMEL surumde '.\nB' gibi cumle sinirini asip sonraki kelimenin ilk
harfini yutan token'lar da vardi.

**Tahmin 2:** "Regex sikistirmayi kotulestirir, cunku faydali
birlestirmeleri yasakliyorum."

**Olcum:** tersine biraz IYILESTI (83.116 ->
82.880 token, yuzde
0.28).
Sebebi: regex'siz surumde ayni kelime yanindaki noktalamaya gore farkli
farkli bolunuyordu ve bu parcalanmalar vocab butcesini tuketiyordu.

Ilk 20 birlestirmenin sadece 1'i ayni kaldi.

## 5. Turkce ekleri  <- bu haftanin en net bulgusu

**Bekledigim:** "kesme isareti kendi basina bir parca olur, ek kokten
kopar."

**Cikan:** daha ilginc bir sey. 26 ek denedim, 12'si
bozuluyor:

```
Ankara'dan  ->  ['Ankara', "'d", 'an']     BOZUK
Ankara'nin  ->  ['Ankara', "'nin"]         TEMIZ
```

Bozulanlarin hepsi d/t/s/m ile basliyor. Sebep: desenin ILK kurali
`'(?i:[sdmt]|ll|ve|re)` Ingilizce kisaltmalarini yakaliyor ve regex
alternatifleri SIRAYLA deniyor. Ek bu harflerden biriyle basliyorsa Kural
1 once eslesip ekin sadece ilk harfini aliyor.

Bozulanlar Turkce'nin en sik ekleri: bulunma (-da/-de/-ta/-te) ve ayrilma
(-dan/-den/-tan/-ten) hali. Yani desen, Turkce'de en cok ihtiyac duyulan
eklerde en kotu davraniyor.

Sayilar da benzer: `\p{N}{1,3}` kurali sayilari 3 basamakta kesiyor.
"2026'da" -> ['202', '6', "'d", 'a'] - dort parca.

## 6. Turkce - Ingilizce

| | Kelime | Token | Token/kelime |
|---|---:|---:|---:|
| Turkce | 47 | 148 | 3.149 |
| Ingilizce | 76 | 84 | 1.105 |

Kelime basina 2.85 kat fark ama
toplamda 1.76 kat. Aradaki ucurum
Turkce'nin eklemeli yapisindan: ayni anlami daha az kelimeyle
anlatiyor, bu da byte ve veri dezavantajini kismen dengeliyor.

**Bekledigim:** kendi tokenizer'im (1024) GPT-4'u (100.256) Turkce'de
rahat gecer.

**Cikan:** esit. Ikisi de 148 token. GPT-4'un
vocab'i yuz kat buyuk oldugu icin Turkce'ye ayirdigi kucuk pay bile benim
butun vocab'im kadar. Dogru okuma: benimki yetisiyor, gecmiyor.

## 7. tiktoken denkligi  <- bir kod hatasi

**Ilk cozumum:** "token'i ilk gecerli yerden ikiye bol". KeyError ile
coktu.

**Sebep:** bir token birden fazla yerden bolunebiliyor ve dogrusu her
zaman ilki degil. BPE'de birlestirme SIRASI belirleyici.

**Dogrusu:** her token'i byte'larina ayirip SADECE o token'dan once gelmis
birlestirmeleri (rank < o token'in rank'i) uygulamak. Geriye tam iki parca
kaliyor.

Duzeltince 13 elle secilen ornek, 40.000
karakterlik gercek metin ve 300 rastgele metin - hicbirinde fark yok.

Ikinci puruz: GPT-4 taban byte'larini karistiriyor (b'!' 0 numarali token,
'a' 64 numarali). byte_shuffle ile cozdum.

## 8. Model, BPE token'lariyla

| | Hafta 8 | Hafta 9 |
|---|---:|---:|
| vocab | 87 | 1024 |
| val loss | 1.5841 | 3.6946 |
| bit/karakter | 2.2854 | 2.1784 |

Ham loss'lar karsilastirilamaz: olcu birimi, secenek sayisi ve veri
bolmesi farkli. Ortak olcege (bit/karakter) cevirince fark
0.1069.

Ama asil fark ciktida gorunuyor. Hafta 8'in harf modeli harf obekleri
uretiyordu; BPE modeli gercek Turkce kelimeler ve dogru ekler uretiyor.
Sebebi basit: artik harf harf degil, hece ve ek olarak seciyor.

**Bir duzeltme:** Hafta 8'de korpusu %90/%10 SIRAYLA bolmustum ve val
loss train'den dusuk cikmisti. O haftanin raporunda kendi hatam olarak
yazmistim; bu hafta dilimleri karistirarak boldum.

## Acik kalanlar

- Deseni Turkce ekleriyle genisletsem kac token kazanirdim? Hesabi basit
  ama olcmedim.
- vocab=1024 secimini model egiterek dogrulamadim. 512 ve 2048 ile ayni
  modeli egitip karsilastirmak isterdim.
- GPT-4 taban byte'larini neden karistirmis, bulamadim.
- sentencepiece'i (videonun son bolumu) henuz denemedim.
