# YZ50 - Hafta 7: Self-Attention

Karpathy'nin "Let's build GPT" videosunun ilk 1 saat 22 dakikasi.
Tiny Shakespeare uzerinde bigram tabani kurdum, sonra tek bir
self-attention head yazip bigram'i gectim.

## Dosyalar

- `ortak.py`               - tokenizer, train/val ayrimi, get_batch, egitim, loss olcumu
- `model.py`               - Head ve AttentionDilModeli siniflari
- `gorev1_veri_bigram.py`  - veri, block/batch, bigram tabani
- `gorev2_ortalama.py`     - gecmisin ortalamasi uc yoldan (for, tril, softmax)
- `gorev3_head.py`         - q/k/v, wei satiri okuma, sqrt olcekleme, pozisyon embedding
- `gorev4_egitim.py`       - head'li model egitimi + bigram karsilastirmasi
- `gorev5_ek.py`           - (ek) attention isi haritasi + tril maskesiz model
- `gorseller.py`           - diyagramlar
- `input.txt`              - Tiny Shakespeare (1.115.394 karakter)
- `deney_notu.md`          - sonuclar ve aciklamalar

## Calistirma

    pip install torch matplotlib

    python gorev1_veri_bigram.py   # ~60 sn
    python gorev2_ortalama.py      # ~2 sn
    python gorev3_head.py          # ~2 sn
    python gorev4_egitim.py        # ~80 sn
    python gorev5_ek.py            # ~25 sn

## Ana sonuc

| model | parametre | train | val |
|---|---|---|---|
| bigram (1 harf) | 4.225 | 2.4573 | 2.4769 |
| tek head (8 harf, attention) | 7.553 | 2.3882 | 2.3998 |
| iyilesme | | | 0.0771 |

## Ek gorev

(a) Isi haritasi: `attention_isi.png`. Egitilmis head cogunlukla
YEREL bir oruntu ogrenmis: her token ya kendisine ya bir oncekine
bakiyor. Ornegin 't' token'i 's'ye 0.637, kendisine 0.352 agirlik
veriyor.

(b) tril maskesi kaldirilinca:

| model | val loss | uretilen metin |
|---|---|---|
| maskeli | 2.3998 | okunabilir Shakespeare taklidi |
| maskesiz | 0.3955 | "SSSSSS...eeeeee..." |

Val loss 6 kat dustu ama model ise yaramaz hale geldi: egitimde
gelecegi gordugu icin kopya cekiyor, uretimde gelecek olmadigi icin
coküyor.

## Gecen haftalarla baglanti

- H6 WaveNet de 8 harfe bakiyordu ama birlestirme SABITTI. Burada
  agirliklari model veriye gore seciyor.
- H6'da sira bilgisi agacin yapisinda kodluydu; attention sirayi
  bilmedigi icin pozisyon embedding'i eklendi.
- sqrt(head_size) olceklemesi, H4'teki Kaiming init ve tanh doymasi
  ile ayni mantik: cikti uca yapisinca gradient olur.
- Maskesiz deney, H6'daki train/dev/test dersinin baska bir hali:
  olcumu bozan bir sizinti varsa dusen loss ilerleme degil.
