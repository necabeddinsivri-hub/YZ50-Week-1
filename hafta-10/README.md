# YZ50 - Hafta 10: GPT-2 (124M)

Andrej Karpathy, *Let's reproduce GPT-2 (124M)* - ilk saat. Modeli
kurmak, OpenAI'in egittigi agirliklari yuklemek, ornek uretmek ve ilk
egitim adimlarini atmak.

**Butun sayilar bu klasordeki kodu calistirarak elde edildi.** Rapor ve
grafikler `sonuc_gorev*.json` dosyalarini okuyor.

## Bu haftanin tezi

Hafta 8'de yazdigim GPT ile GPT-2 arasinda mimari fark yok: ayni
attention, ayni blok, ayni residual. Fark olcekte - 117 kat. Bu olcekte
asil is mimariyi yazmak degil, **modelin dogru kuruldugunu kanitlamak**.

Iki kontrol kullandim ve **ikisi de kendi kodumdaki hatalari yakaladi.**

## Hafta 8'deki GPT ile GPT-2 (124M)

| | Hafta 8 | GPT-2 124M | Kat |
|---|---:|---:|---:|
| n_embd | 128 | 768 | 6x |
| n_head | 4 | 12 | 3x |
| n_layer | 4 | 12 | 3x |
| block_size | 64 | 1.024 | 16x |
| vocab_size | 1.024 | 50.257 | 49x |
| parametre | 1.063.168 | 124.439.808 | 117x |

Mimaride degisen alti sey: (1) isimler (transformer.wte/wpe/h/ln_f),
(2) q,k,v tek katmanda (c_attn), (3) ReLU yerine GELU(tanh), (4)
lm_head'de bias yok, (5) weight tying, (6) LayerNorm adlari ln_1/ln_2.

## Gorev 4 - Ilk egitim adimlari

### 4a) Ilk loss ln(V)'ye yakin mi

Beklenen: ln(50257) = **10.8249**

| Baslatma semasi | Parametre | Ilk loss | Fark |
|---|---:|---:|---:|
| ozel init + weight tying (GPT-2 semasi) | 124.439.808 | 10.9600 | 0.1351 |
| ozel init, weight tying YOK | 163.037.184 | 11.0154 | 0.1905 |
| PyTorch varsayilan init + weight tying | 124.439.808 | 10.8889 | 0.0640 |

Model hicbir sey ogrenmemisse 50.257 token'in hepsini esit olasi gormeli,
yani loss = ln(V). Bu basit sayi bir **hata detektoru**: cok buyuk cikarsa
baslangic agirliklari fazla buyuk, cok kucuk cikarsa bir yerde sizinti var.

### 4a-ek) Sanity check kendi hatami yakaladi

Ilk yazdigimda testi soyle kurmustum:

```python
x = torch.randint(0, V, (4, 32))
_, loss = model(x, x)       # <- hedef olarak GIRDININ KENDISI
```

Cikan sayi 9.99'du, yani beklenenin 0.84
ALTINDA. Loss'un beklenenden dusuk cikmasi her zaman supheli (Hafta 7'de
tril maskesini kaldirinca da boyle olmustu), o yuzden ustune gittim.

| | hedef = girdi (hatali) | bagimsiz rastgele | gercek metin |
|---|---:|---:|---:|
| weight tying VAR | 9.9881 | 11.0015 | 10.9600 |
| weight tying YOK | 11.0215 | 10.9278 | 11.0154 |

**Sebep weight tying.** Giris embedding'i ile cikis katmani ayni tensor,
yani bir token'in logit'i o token'in embedding vektoru ile son katmanin
ciktisinin ic carpimi. Model egitilmedigi icin ciktisi buyuk olcude girdi
embedding'ine benziyor - ve bir vektorun kendisiyle ic carpimi en
buyuktur. Yani model "bir sonraki token girdinin kendisidir" egilimiyle
basliyor.

Kanit ikinci satirda: weight tying YOKken ayni test 11.02 veriyor.

### 4b) Tek batch'i ezberletme

| | |
|---|---|
| batch | B=4, T=32 (128 token) |
| adim / lr | 300 / 0.0003 |
| ilk loss | 10.9600 |
| son loss | **0.0008** |
| token dogrulugu | **%100.0** |
| sure | 5.7 dk (cpu) |

Loss **sifira inmek zorunda** - 124M parametreli model icin
128 token'i ezberlemek kolay. Inmiyorsa
ogrenme sorunu degil, KOD sorunu var: gradyan akmiyor, hedefler kaymis,
optimizer parametreleri gormuyor. Bu test modelin ne kadar iyi oldugunu
degil, makinenin takili olup olmadigini olcuyor.

## Gorev 5 (ek a) - Data loader, weight tying, init

### Weight tying

| | Parametre | Bellek (fp32) |
|---|---:|---:|
| tying YOK | 163.037.184 | 652 MB |
| tying VAR | 124.439.808 | 498 MB |
| **kazanc** | **38.597.376** | 154 MB |

Kazanc tam olarak embedding tablosu: 50.257 x 768 = 38.597.376.
GPT-2'nin "124M" olarak anilmasinin sebebi weight tying; olmasaydi
163.037.184 olacakti.

### Egitim

120 adim, loss 10.96 ->
5.76. Hiz **105
token/sn** (cpu). Karpathy A100'de ~16.000 aliyor -
152 kat fark, tamamen donanimdan. Tiny
Shakespeare'in tamami 338.025 token; bir epoch bu
makinede ~0.9 saat
surerdi.

### Baslangic degerleri

| Katman | Beklenen std | Olculen (taze model) |
|---|---:|---:|
| Linear / Embedding (genel) | 0.0200 | 0.0200 |
| mlp.c_proj (residual) | 0.0041 | 0.0041 |
| attn.c_proj (residual) | 0.0041 | 0.0041 |

Residual yoluna eklenen katmanlarin std'si 1/sqrt(2*n_layer) ile
olcekleniyor: her blokta residual'a iki ekleme var, 12 blok boyunca 24
bagimsiz terim toplaniyor ve varyanslar toplanir.

**Ikinci hatam:** ilk olcumu 120 adim EGITILMIS model
uzerinde yapmisim (0.0047 cikmisti).
Taze model kurunca sema birebir tutuyor.

## Gorev 1 - Hazir agirliklari incelemek

160 tensor, 124.439.808 parametre.
Dosyada **lm_head yok** - weight tying yuzunden tek kopya saklanmis.
Toplamin %31'i embedding tablosunda.

### Pozisyon embedding'leri (wpe)

| Olcum | Deger |
|---|---:|
| pozisyon 0'in normu | 9.88 |
| medyan norm | 3.37 |
| pozisyon 1022'nin normu | 3.86 |
| **pozisyon 1023'un normu** | **0.12** |
| benzerlik 0-1 | 0.524 |
| benzerlik 7-8 | 0.998 |

Uc bulgu, ucu de beklentimden farkliydi:

1. **Pozisyon 0 bir aykiri deger.** Normu medyanin uc kati. "Metnin ilk
   token'i" ozel bir bilgi ve model bunu guclu bir sinyal olarak kodlamis.
2. **Cozunurluk basta yuksek, ortada dusuk.** Ilk 50 pozisyonda 100 adim
   otesiyle benzerlik 0.204, ortada (500-800)
   0.542. Model basta "kacinci token" sorusunu
   ince ayrintisiyla ayirt ediyor, ilerledikce kabalasiyor.
3. **Son pozisyon hic egitilmemis.** 1022'nin normu
   3.86, 1023'un 0.12.
   Sebep: hedef girdinin bir token kaydirilmis hali, yani 1023. pozisyonun
   tahmin edecegi 1025. token yok. O embedding hicbir zaman gradyan
   almiyor.

Ilk yazdigimda "tablonun sonu basindan daha gurultulu" diye tahmin
etmistim. Olcunce yanlis cikti: son bolgedeki komsu benzerligi
0.999, ilk bolgede 0.987.
Tam tersi. Tek egitilmemis satir 1023 ve o da gurultulu degil, sifira
yakin.

## Gorev 2 - Yazmak ve yuklemek

Dort tur katman TRANSPOZE edilmeli: attn.c_attn, attn.c_proj, mlp.c_fc,
mlp.c_proj. Sebep: HF, OpenAI'in TF kodunu tasirken nn.Linear yerine
Conv1D kullandi; o sinif agirligi (giris, cikis) seklinde tutuyor.

**Tehlikeli kisim:** mlp.c_fc dikdortgen, transpoze unutulursa kod hata
verir. Ama attn.c_proj 768x768, yani KARE - transpoze unutulursa boyut
tutar, kod calisir ve model sessizce sacmalar.

Logits karsilastirmasi: en buyuk fark
**1.53e-04** (rastgele girdide 9.92e-05),
yani float32 yuvarlama mertebesinde.

## Gorev 3 - Modeli konusturmak

top-k = 50 ile sampling. Ingilizce ciktilar akici, Turkce ciktilar bozuk.
Ayni icerik Turkce'de **2.00 kat** token tutuyor.

Uc sebep: (1) tokenizer - 'Merhaba' uce bolunuyor, (2) baglam penceresi,
(3) **egitim verisi** - GPT-2 WebText'te egitildi ve Ingilizce disi
icerik ayiklandi. Ucuncusu belirleyici.

## Gorev 5 (ek b) - Turkce ve Ingilizce loss

| | Token | Ortalama loss | Perplexity |
|---|---:|---:|---:|
| Ingilizce | 72 | 3.6937 | 40 |
| Turkce (sade) | 126 | 6.9797 | 1075 |
| Turkce (c g i o s u) | 163 | 5.5414 | 255 |
| egitimsiz taban ln(V) | - | 10.8249 | 50257 |

**En carpici sayi:** Turkce loss, egitimsiz tabanin
%64'i. Yani 124M parametreli, 40 GB
metinle egitilmis model, Turkce'de hicbir sey bilmeyenden sadece
%36 daha iyi. Ingilizce'de bu oran
%66.

**Testimi kurarken yanildim.** Isaretli harfli surumu "sade yazim token
sayisini dusurur, boylece tokenizer etkisini ayirt ederim" diye eklemistim.
Token sayisi gercekten dustu (163 -> 126)
ama loss YUKSELDI (5.541 -> 6.980). Yani
daha cok token kullanan surum 1.44 nat
daha iyi.

Sebep: internetteki gercek Turkce isaretli harflerle yazilir. Sade yazim ne
Ingilizce ne duzgun Turkce - model icin hic gormedigi bir sey. Demek ki
token sayisi loss'u belirleyen sey degil; belirleyici olan modelin o yaziyi
daha once gorup gormedigi. Gecen hafta "tokenizer bir semptom, veri ise
sebep" diye yazmistim - bu olcum o cumleyi bekledigimden daha guclu
destekliyor.

Not: Hafta 8 ve 9'da "loss'lari ancak olcum kosullari ayniysa
karsilastirabilirsin" demistim. Burada o uyari gecerli degil - ayni model,
ayni tokenizer, ayni V. Degisen tek sey dil.

## Dosyalar

| Dosya | Ne yapiyor |
|---|---|
| `gpt2.py` | GPT-2 modulu: GPTConfig, CausalSelfAttention, MLP, Block, GPT, from_pretrained |
| `tokenizer.py` | GPT-2 tokenizer'ini cevrimdisi yukler |
| `agirlik.py` | Agirlik dosyasini ve HF modelini yerelden kurar |
| `gorev1_incele.py` | Agirlik listesi, parametre hesabi, wpe analizi |
| `gorev2_yukle.py` | Hafta 8'den farklar, transpoze, logits karsilastirmasi |
| `gorev3_konustur.py` | top-k sampling, TR/EN cikti |
| `gorev4_ilk_egitim.py` | ln(V) kontrolu, sanity-check hatasi, ezberleme |
| `gorev5a_dataloader.py` | Weight tying, data loader, egitim, init semasi |
| `gorev5b_loss.py` | TR/EN loss ve perplexity |
| `gorseller.py` | Grafikleri sonuc_gorev*.json'dan uretir |
| `veri/gpt2.tiktoken` | GPT-2'nin birlestirme tablosu |

## Calistirma

```bash
python3 tokenizer.py               # tokenizer testi
python3 gorev1_incele.py           # agirlik analizi
python3 gorev2_yukle.py            # yukleme + karsilastirma
python3 gorev3_konustur.py         # sampling
python3 gorev4_ilk_egitim.py       # ln(V) + ezberleme      (~8 dk)
python3 gorev5a_dataloader.py      # data loader + egitim   (~5 dk)
python3 gorev5b_loss.py            # TR/EN loss
python3 gorseller.py               # grafikler
```

Gereken: `torch`, `tiktoken`, `regex`, `transformers`, `safetensors`,
`matplotlib`. GPU gerekmiyor (bu kosu 2 CPU cekirdekli bir makinede
yapildi). `veri/` klasorunde `model.safetensors` ve `config.json` olmali
(HF: openai-community/gpt2).

**Ortam notu:** calistirdigim ortamdan huggingface.co'ya ve OpenAI'in
agirlik sunucusuna cikis kapali (proxy 403). Tokenizer tablosunu gecen
haftaki yontemle npm'deki js-tiktoken paketinden aldim; agirliklari elle indirip veri/ klasorune koydum.

## Onceki haftalarla baglanti

- **Hafta 4 (init):** ilk loss'un beklenenden buyuk cikmasi = baslangic
  olcegi yanlis. Orada tanh doymasiyla gormustum.
- **Hafta 5 (manuel backprop):** kodun dogrulugunu kendi testimle degil
  bagimsiz bir referansla olcmek - orada PyTorch gradyanlari, burada HF
  modeli.
- **Hafta 7 (tril maskesi):** loss'un beklenenden DUSUK cikmasi sizinti
  isareti. Bu hafta ayni sinyal kendi test kodumdaki hatayi gosterdi.
- **Hafta 8 (residual):** "toplama dugumunun gradyani 1" - init
  semasindaki 1/sqrt(2*n_layer) carpani bunun ileri yondeki karsiligi.
- **Hafta 9 (tokenizer):** q,k,v'nin tek matmul'de hesaplanmasini orada
  yazip denkligini gostermistim; "tokenizer semptom, veri sebep"
  cumlesinin kanitini bu hafta aldim.
