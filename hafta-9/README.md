# YZ50 - Hafta 9: Tokenizer

Andrej Karpathy, *Let's build the GPT Tokenizer*. Sekiz haftadir modelim
metni harf harf okudu; bu hafta byte seviyesinden baslayip sik gecen
gruplari birlestiren bir tokenizer yazdim (Byte Pair Encoding).

**Butun sayilar bu klasordeki kodu calistirarak elde edildi.** Rapor ve
grafikler `sonuc_gorev*.json` dosyalarini okuyor; elle girilmis sayi yok.

## Ozet

| | |
|---|---|
| Sikistirma | 2.639x (byte / token) |
| Ayni block_size'a sigan metin | 2.41 kat (Hafta 8'e gore) |
| Turkce / Ingilizce, token per kelime | 3.149 / 1.105 |
| GPT-4 deseninin bozdugu Turkce ek | 12 / 26 |
| tiktoken ile denklik | bire bir (0 fark) |
| Model, karakter basina bit | 2.2854 -> 2.1784 |

## Gorev 1 - Unicode ve UTF-8

Turkce'nin isaretli harfleri (c g i o s u uzerine) UTF-8'de 2 byte,
ASCII harfler 1 byte. Korpusumda fark yuzde
9.1:
962.254 karakter, 1.050.074 byte.

Kod noktalarini dogrudan vocab yapmanin dort sorunu: (1) 150.000 satirlik
tablonun cogu bos kalir, (2) kucultursem bilinmeyen karakter sorunu cikar,
(3) Unicode her yil degisiyor, (4) ayni gorunen metnin birden fazla
yazilisi var.

## Gorev 2 - BPE egitimi

Veri: Hafta 8'de kendim derledigim `turkce.txt`. vocab_size =
1024, egitim 160.4 saniye.

### Ilk 20 birlestirme

| # | Ikili | Yeni | Metin | Kac kere |
|---|---|---|---|---|
| 1 | (196, 177) | 256 | `ı` | 37.557 |
| 2 | (110, 32) | 257 | `n ` | 17.357 |
| 3 | (97, 114) | 258 | `ar` | 15.176 |
| 4 | (195, 188) | 259 | `ü` | 13.939 |
| 5 | (101, 32) | 260 | `e ` | 13.912 |
| 6 | (97, 32) | 261 | `a ` | 13.229 |
| 7 | (101, 114) | 262 | `er` | 12.896 |
| 8 | (197, 159) | 263 | `ş` | 12.670 |
| 9 | (105, 32) | 264 | `i ` | 12.649 |
| 10 | (46, 10) | 265 | `.\u000a` | 11.446 |
| 11 | (256, 32) | 266 | `ı ` | 8.872 |
| 12 | (105, 114) | 267 | `ir` | 8.786 |
| 13 | (195, 167) | 268 | `ç` | 8.231 |
| 14 | (97, 110) | 269 | `an` | 7.748 |
| 15 | (196, 159) | 270 | `ğ` | 7.459 |
| 16 | (107, 32) | 271 | `k ` | 7.369 |
| 17 | (105, 110) | 272 | `in` | 7.178 |
| 18 | (108, 258) | 273 | `lar` | 7.142 |
| 19 | (105, 108) | 274 | `il` | 7.137 |
| 20 | (44, 32) | 275 | `, ` | 6.950 |

**Ilk birlestirme bir harf ikilisi degil, TEK HARF cikti:** `ı`.
Turkce'nin noktasiz i'si UTF-8'de iki byte ve metnimde
37.557 kere geciyor. Ayni sey u, s, c, g icin de
oluyor. Algoritma kimse soylemeden Turkce alfabesini kesfediyor.

### Sikistirma

| Birlestirme | Kalan token | Oran |
|---:|---:|---:|
| 0 (ham byte) | 1.050.074 | 1.000x |
| 50 | 692.562 | 1.516x |
| 100 | 602.023 | 1.744x |
| 200 | 525.882 | 1.997x |
| 400 | 455.702 | 2.304x |
| 768 | 395.449 | 2.655x |

### Vocab boyutu taramasi

| Vocab | Token | Oran | Egitim |
|---:|---:|---:|---:|
| 256 | 65.771 | 1.000x | 0.0 sn |
| 512 | 31.149 | 2.111x | 19.8 sn |
| 1.024 | 24.293 | 2.707x | 50.8 sn |
| 2.048 | 19.959 | 3.295x | 113.7 sn |
| 4.096 | 16.587 | 3.965x | 233.6 sn |

1024 sectim: 1 MB'lik korpusta 4096 token'in cogu birkac kez gorulur ve
iyi egitilemez.

## Gorev 3 - encode / decode

`decode(encode(metin)) == metin` butun orneklerde tamam, emoji dahil -
tokenizer'i sadece Turkce metinde egitmis olmama ragmen. Sebebi taban
vocab'in 256 byte olmasi.

Ters yon (`encode(decode(ids)) == ids`) garantili degil: 200 rastgele
token dizisinden 146 tanesi geri donmedi. Her token dizisi
dogal bir metne karsilik gelmiyor.

Gecersiz UTF-8'de `errors='replace'` sectim. strict degil cunku model
uretimde yarim karakter uretebilir ve program cokerse uretim durur;
ignore degil cunku sessizce siliyor ve haberim olmuyor.

## Gorev 4 - GPT-4'un regex deseni

Ilk 20 birlestirmenin sadece 1'i ayni kaldi.

| | Sonu bosluk | Icinde noktalama | Basi bosluk |
|---|---:|---:|---:|
| TEMEL (regex yok) | 243 | 61 | 30 |
| REGEX (GPT-4) | 0 | 11 | 322 |

**Iki tahminim de tutmadi.** (1) "Token ya harf ya noktalama olur"
demistim; desen bastaki bosluga bilerek izin veriyor, gercek fark sonda.
(2) "Sikistirma kotulesir" demistim; tersine biraz iyilesti
(83.116 -> 82.880 token).

### Turkce ekleri

26 ekten 12'si ikiye bolunuyor:

- **Bozulan:** 'da 'de 'ta 'te 'dan 'den 'tan 'ten 'si 'sı 'mi 'mı
- **Temiz:** 'nin 'nın 'in 'ın 'ye 'ya 'e 'a 'le 'la 'ler 'lar 'nde 'ndan

Bozulanlarin hepsi d/t/s/m ile basliyor, cunku desenin ILK kurali
Ingilizce kisaltmalari ('d 's 'm 't) yakaliyor ve ekin sadece ilk
harfini yiyor:

```
Ankara'dan  ->  ['Ankara', "'d", 'an']
Turkiye'nin ->  ['Turkiye', "'nin"]      <- temiz, cunku 'n' listede yok
```

Bunlar Turkce'nin en sik ekleri: bulunma (-da/-de/-ta/-te) ve ayrilma
(-dan/-den/-tan/-ten) hali.

## Gorev 5 - Turkce ile Ingilizce

| | Karakter | Kelime | Token | Token/kelime | Byte/token |
|---|---:|---:|---:|---:|---:|
| Turkce | 351 | 47 | 148 | 3.149 | 2.764 |
| Ingilizce | 410 | 76 | 84 | 1.105 | 4.881 |

Kelime basina 2.85 kat fark,
toplamda 1.76 kat. Arada ucurum var cunku
Turkce'nin eklemeli yapisi ilk iki dezavantaji (byte maliyeti ve egitim
verisi dengesi) kismen dengeliyor: ayni anlami daha az kelimeyle
anlatiyor.

### Dort tokenizer yan yana

| Tokenizer | Vocab | Token | Byte/token |
|---|---:|---:|---:|
| harf seviyesi (Hafta 8) | 87 | 351 | 1.165 |
| benim, temel | 1.024 | 150 | 2.727 |
| benim, regex | 1.024 | 148 | 2.764 |
| GPT-4 cl100k_base | 100.256 | 148 | 2.764 |

Kendi tokenizer'im GPT-4'u Turkce'de **gecmiyor, yetisiyor** - ikisi de
148 token. Vocab farki yuz kat oldugu icin
GPT-4'un Turkce'ye ayirdigi kucuk pay bile benim butun vocab'im kadar.

## Gorev 6 (ek) - ikisini de yaptim

### (a) tiktoken ile denklik

Kendi RegexTokenizer'im, GPT-4'un 100.000 birlestirmesiyle
beslendiginde tiktoken ile **bire bir ayni** token'lari uretiyor:
13 elle secilen ornek, 40.000
karakterlik gercek metin ve 300 rastgele metin - hicbirinde fark yok.

Iki puruz vardi: (1) tiktoken birlestirme agacini vermiyor, geri kurmak
gerekiyor - ilk cozumum yanlisti ve KeyError ile coktu; (2) GPT-4 taban
byte'larini karistiriyor.

### (b) Hafta 8 modelini BPE token'lariyla egitmek

| | Hafta 8 (harf) | Hafta 9 (BPE) |
|---|---:|---:|
| vocab | 87 | 1024 |
| parametre | 822.359 | 1.063.168 |
| train loss | 1.6233 | 3.2049 |
| val loss | 1.5841 | 3.6946 |
| log(V) taban | 4.4659 | 6.9315 |
| **bit / karakter** | **2.2854** | **2.1784** |

**Ham val loss'lar karsilastirilamaz.** Olcu birimi farkli (token =
karakter -> token ~ 2.45 karakter), secenek
sayisi farkli, veri bolmesi farkli. Ortak olcek:

```
bpc = (val_loss / ln 2) x (token sayisi / karakter sayisi)
```

Bu olcege gore fark 0.1069 bit/karakter.

## Dosyalar

| Dosya | Ne yapiyor |
|---|---|
| `bpe.py` | Cekirdek: istatistik, birlestir, BasicTokenizer, RegexTokenizer, GPT4Tokenizer |
| `gpt4_yukle.py` | cl100k_base tablosunu cevrimdisi yukler |
| `gorev1_byte.py` | Unicode kod noktalari ve UTF-8 byte'lari |
| `gorev2_bpe.py` | BPE egitimi, ilk 20 birlestirme, sikistirma, vocab taramasi |
| `gorev3_encode_decode.py` | Gidis-donus testi, gecersiz UTF-8, birlestirme sirasi |
| `gorev4_regex.py` | GPT-4 deseni, vocab yapisi, 26 Turkce ek testi |
| `gorev5_karsilastirma.py` | Turkce/Ingilizce ve dort tokenizer |
| `gorev6a_gpt4.py` | tiktoken ile denklik testi |
| `gorev6b_model.py` | Hafta 8 modelini BPE token'lariyla egitmek |
| `model_h8.py` | Hafta 8'deki GPT, token dizisiyle calisacak sekilde |
| `gorseller.py` | Grafikleri sonuc_gorev*.json'dan uretir |
| `veri/cl100k_base.tiktoken` | GPT-4'un birlestirme tablosu |
| `turkce.txt` | Hafta 8'de derledigim ~1 MB Turkce metin |

## Calistirma

```bash
python3 gorev1_byte.py             # Unicode / UTF-8      (aninda)
python3 gorev2_bpe.py              # BPE egitimi          (~12 dk)
python3 gorev3_encode_decode.py    # encode / decode      (~1 dk)
python3 gorev4_regex.py            # GPT-4 deseni         (~6 dk)
python3 gorev5_karsilastirma.py    # TR / EN              (~1 dk)
python3 gorev6a_gpt4.py            # tiktoken denkligi    (~3 dk)
python3 gorev6b_model.py           # model egitimi        (~16 dk)
python3 gorseller.py               # grafikler
```

Gereken: `regex`, `tiktoken`, `torch`, `matplotlib`, `numpy`. GPU gerekmiyor.

**Not:** `tiktoken` normalde birlestirme tablosunu OpenAI'in sunucusundan
indiriyor. Kodu calistirdigim ortamdan o adrese cikis kapaliydi; ayni
tabloyu npm'deki `js-tiktoken` paketinden alip `veri/` klasorune koydum ve
tiktoken'a elle veriyorum (`gpt4_yukle.py`). Tablo bire bir ayni.

## Onceki haftalarla baglanti

- **Hafta 5 (manuel backprop):** Gorev 6(a)'da kodun dogrulugunu kendi
  testimle degil bagimsiz bir referansla (tiktoken) olctum - orada
  PyTorch'un gradyanlariyla yaptigim seyin aynisi.
- **Hafta 6 (BatchNorm):** "egitim ve cikarim ayni seyi yapmali" dersi bu
  hafta iki kere karsima cikti: encode'da birlestirme sirasi, ve GPT-4'un
  agacini kurtarirken yaptigim hata.
- **Hafta 7 (uretim):** kenar durumlarini kodun kendisi karsilamali -
  orada baglam kirpmasi, burada errors='replace'.
- **Hafta 8 (Turkce model):** ayni korpus, ayni model, farkli tokenizer.
  Hafta 8'de val bolmesini sirayla yapma hatami bu hafta duzelttim.
