# YZ50 - Hafta 6: Kendi torch.nn'im ve WaveNet

Iki is yaptim. Birincisi: Hafta 4'teki dagınık kodu katman siniflarina
topladim (Linear, BatchNorm1d, Tanh, Embedding, Flatten, Sequential).
Ikincisi: baglami 3 harften 8 harfe cikardim ve 8 harfi tek seferde
duzlestirmek yerine ikiser ikiser birlestiren bir agac kurdum (WaveNet,
DeepMind 2016).

## Dosyalar

- `katmanlar.py`             - kendi katman siniflarim (torch.nn yerine)
- `ortak.py`                 - veri hazirlama, egitim, degerlendirme
- `gorev1_siniflar.py`       - Hafta 4 modeli tek Sequential olarak + loss egrisi
- `gorev2_baglam8.py`        - baglam 3 -> 8, ayni duz model (taban)
- `gorev3_wavenet.py`        - hiyerarsik model, her katmanin sekli ve aciklamasi
- `gorev4_batchnorm_hatasi.py` - 3 boyutlu girdide BatchNorm hatasi, once/sonra
- `gorev5_buyutme.py`        - uc modelli karsilastirma tablosu
- `gorev6_turkce.py`         - ayni WaveNet Turkce isimlerle
- `gorev7_ek_conv1d.py`      - (ek, secenek a) ayni agac Conv1d ile
- `gorseller.py`             - diyagramlar (egitim yok)
- `names.txt`, `isimler.txt` - veri
- `deney_notu.md`            - sonuclar ve aciklamalar

## Calistirma

    pip install torch matplotlib

    python gorev1_siniflar.py          # ~20 sn
    python gorev2_baglam8.py           # ~45 sn
    python gorev3_wavenet.py           # ~40 sn
    python gorev4_batchnorm_hatasi.py  # ~80 sn
    python gorev5_buyutme.py           # ~3 dk
    python gorev6_turkce.py            # ~2 dk
    python gorev7_ek_conv1d.py         # ~30 sn

## Ana sonuc (Gorev 5)

Embedding 24, 60.000 adim:

| model | parametre | dev loss |
|---|---|---|
| baglam 3, duz MLP | 20.875 | 2.1030 |
| baglam 8, duz MLP | 44.875 | 2.0101 |
| baglam 8, WaveNet | 76.579 | 1.9942 |

Karpathy'nin videodaki sonucu 1.993 (200.000 adim).

## BatchNorm hatasi (Gorev 4)

| BatchNorm | running_mean sekli | dev loss |
|---|---|---|
| hatali, mean(0) | (1, 4, 68) | 2.0890 |
| duzeltilmis, mean((0,1)) | (1, 1, 68) | 2.0779 |

## Turkce (Gorev 6)

| model | dev loss |
|---|---|
| Hafta 4 MLP | 2.3183 |
| Hafta 6 baglam 3 MLP | 2.2974 |
| Hafta 6 baglam 8 WaveNet | 2.1829 |

Urettigi isimler: ziyangul, gulfizar, gunmaz, eymet, gunayse, sememan

## Ek gorev (Conv1d)

Egitilmis WaveNet'in agirliklarini Conv1d'ye kopyaladim:
- kernel 2, stride 2 konvolusyon: maks fark 1.55e-06
- dilated konvolusyon (1, 2, 4): bir ismin 7 pozisyonu tek geciste,
  maks fark 1.43e-06, 49 yerine 31 islem
