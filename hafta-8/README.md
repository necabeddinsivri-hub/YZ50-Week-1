# YZ50 - Hafta 8: GPT'yi bitirdim

Andrej Karpathy, *Let's build GPT* - 1:21:59'dan sonu. Hafta 7'de tek bir
self-attention head yazmistim; bu hafta Transformer blogunun kalan
parcalarini tek tek ekledim ve her adimda val loss'u olctum.

**Butun sayilar bu klasordeki kodu calistirarak elde edildi.** Rapor ve
grafikler `sonuclar.json` dosyasini okuyor; elle girilmis hicbir sayi yok.

## Ana tablo

| # | Model | Parametre | train | val | val kazanci |
|---|-------|----------:|------:|----:|------------:|
| 1 | bigram (taban) | 4.225 | 2.4573 | 2.4769 | — |
| 2 | tek head (Hafta 7) | 7.553 | 2.3882 | 2.3998 | 0.0771 |
| 3 | multi-head (4x8) | 8.609 | 2.2562 | 2.2604 | 0.1394 |
| 4 | + feedforward | 16.961 | 2.1597 | 2.1818 | 0.0786 |
| * | 3 blok, residual YOK (teshis) | 41.921 | 2.2672 | 2.2811 | -0.0992 |
| 5 | + residual (3 blok) | 41.921 | 1.9846 | 2.0700 | 0.1118 |
| 6 | + LayerNorm | 42.369 | 1.9774 | 2.0520 | 0.0180 |

Butun satirlar: block=8, batch=32, n_embd=32, 5000 adim, lr=1e-3
(bigram'da 1e-2), tohum 1337. (*) satiri ana zincirin disinda; 4. satirla
karsilastirilmali.

En buyuk tek kazanc multi-head'de (0.1394) ve sadece
1.056 parametre ekleyerek geldi. En ogretici sonuc
teshis satirinda: 3 blogu residual olmadan ust uste koymak val loss'u
0.0992 KOTULESTIRDI.

## Olceklenmis model

| | Parametre | train | val | val-train | Sure |
|---|---:|---:|---:|---:|---:|
| dropout 0.0 | 816.705 | 1.4055 | 1.6113 | 0.2058 | 6.4 dk |
| dropout 0.1 | 816.705 | 1.4552 | 1.6476 | 0.1924 | 9.1 dk |

Config: `n_embd=128`, `n_head=4`, `n_blok=4`,
`block=64`, `batch=32`, 4000 adim, `lr=0.0003`.

Karpathy'nin GPU config'i (n_embd=384, 6 head, 6 blok, block=256) bu
makinede olculdu: 3000 adim icin **305 dakika**. O yuzden kuculttum;
gerekcesi raporda.

**Durust not:** dropout bu butcede val loss'u iyilestirmedi
(1.6113 -> 1.6476). Asiri uyum farki daraldi
(0.2058 -> 0.1924) ama kazanc
loss'a yansimadi. Sebebini raporda tartistim: 4000 adim dropout'un
faydasinin ortaya cikmasi icin kisa bir butce.

## Turkce

| | Shakespeare | Turkce |
|---|---:|---:|
| dosya | 1.115.394 bayt | 1.050.074 bayt |
| karakter | 1.115.394 | 962.254 |
| farkli karakter (V) | 65 | 87 |
| val loss | 1.6476 | 1.5841 |
| log(V) = egitimsiz taban | 4.1744 | 4.4659 |
| val / log(V) | 0.395 | 0.355 |

Turkce model daha DUSUK bir val loss aldi - ama iki sayi
**dogrudan karsilastirilamaz**: alfabe boyutu, dilin yapisi ve metin turu
farkli. Ustune, Turkce korpusu sirayla boldugum icin val bolmesi
train'den kolay cikti (val 1.5841 < train 1.6233).
Gerekcesi raporda ve `gorev4_turkce.py` cikisinda.

## Dosyalar

| Dosya | Ne yapiyor |
|---|---|
| `ortak.py` | Veri sinifi, tokenizer, batch cekme, loss olcumu, egitim dongusu |
| `katmanlar.py` | Head, MultiHead, FeedForward, LayerNorm1d (kendi yazdigim), Blok, model siniflari |
| `batchnorm_h6.py` | Hafta 6'da yazdigim BatchNorm1d, degistirilmeden kopyalandi |
| `gorev1_multihead.py` | Gorev 1 - bigram, tek head, multi-head + head basina isi haritasi |
| `gorev2_blok.py` | Gorev 2 - feedforward, residual, LayerNorm; ana tablo; LayerNorm-BatchNorm karsilastirmasi |
| `gorev3_buyut.py` | Gorev 3 - `olcum` modu donanim olcer, varsayilan mod egitir |
| `gorev4_turkce.py` | Gorev 4 - Turkce egitim ve loss karsilastirilabilirligi |
| `gorev5_ek.py` | Gorev 5 - tek matmul denkligi + Sekil 1 eslesmesi (ikisi de) |
| `turkce_derle.py` | Turkce korpusu derleyen betik |
| `gorseller.py` | Grafikleri `sonuclar.json`'dan uretir |
| `sonuclar.json` | Butun kosularin sayilari |
| `turkce.txt` | Derledigim ~1 MB Turkce metin |
| `input.txt` | Tiny Shakespeare (Hafta 7 ile ayni) |
| `deney_notu.md` | Deney gunlugu |
| `video_metni.md` | Video anlatim metni |

## Calistirma sirasi

```bash
python3 turkce_derle.py            # Turkce korpusu derler
python3 gorev1_multihead.py        # tablo satir 1-3    (~1 dk)
python3 gorev2_blok.py             # tablo satir 4-7    (~6 dk)
python3 gorev3_buyut.py olcum      # donanim olcumu     (~1 dk)
python3 gorev3_buyut.py            # olcekli model      (~15 dk)
python3 gorev4_turkce.py           # Turkce             (~9 dk)
python3 gorev5_ek.py               # ek calisma         (~1 dk)
python3 gorseller.py               # grafikler
```

Gereken: `torch`, `matplotlib`, `numpy`. Donanim: 2 CPU cekirdegi, GPU yok.

## Onceki haftalarla baglanti

- **Hafta 1-2 (micrograd):** residual'in neden ise yaradigi dogrudan oradan
  geliyor - toplama dugumunun gradyani 1, yani gradyan bolunmeden geciyor.
- **Hafta 3 (learning rate):** model buyudukce lr'yi 1e-3'ten 3e-4'e cektim.
- **Hafta 4 (init, BatchNorm):** olceklenmis modelde agirliklari `std=0.02`
  ile baslattim; dropout'un ise yaramamasini da Hafta 4'te BatchNorm'un
  dev loss'u kotulestirmesi gibi durustce yazdim.
- **Hafta 5 (manuel backprop):** tek-matmul denkligini "EXACT mi approx mi"
  olcutuyle kontrol ettim; EXACT cikti.
- **Hafta 6 (BatchNorm1d, WaveNet):** LayerNorm karsilastirmasi o sinifla
  yapildi; block=64 secimi "baglami uzatmak en buyuk kazanci veriyor"
  olcumunden geliyor; train/val bolmesinin olcumu etkiledigini Turkce
  tarafta yeniden gordum.
- **Hafta 7 (self-attention):** bigram ve tek head satirlari bu hafta
  sifirdan yeniden egitildi ve Hafta 7 raporuyla bire bir ayni cikti.
