# YZ50 - Hafta 3: Bigram Karakter Dil Modeli

Karpathy'nin makemore videosunu takip ederek yazdigim ilk dil modelim.
Ayni model iki farkli yoldan kuruldu: once sayarak, sonra tek katmanli
bir sinir agiyla egiterek. Ikisi de ayni sonuca cikiyor.

## Dosyalar

- `gorev1_sayim.py`   - bigram'lari dictionary ve 27x27 tensor ile sayma
- `gorev2_sampling.py`- sayimlari olasiliga cevirme, isim uretme, keepdim tuzagi
- `gorev3_nll.py`     - negative log likelihood, smoothing
- `gorev4_sinirag.py` - ayni model one-hot + softmax + gradient descent ile
- `gorev5_turkce.py`  - iki modeli de Turkce isim listesiyle calistirma
- `gorev6_trigram.py` - (ek) trigram, train/dev/test, smoothing ayari
- `names.txt`         - Karpathy'nin Ingilizce isim listesi (32.033 isim)
- `isimler.txt`       - temizlenmis Turkce isim listesi (236.096 isim)
- `deney_notu.md`     - sonuclar ve gozlemler

## Calistirma

    pip install torch

    python gorev1_sayim.py
    python gorev2_sampling.py
    python gorev3_nll.py
    python gorev4_sinirag.py
    python gorev5_turkce.py
    python gorev6_trigram.py

## Sonuclar

Ingilizce (names.txt, 32.033 isim, 228.146 bigram):

| model | loss |
|---|---|
| tamamen rastgele (log 27) | 3.2958 |
| sayim modeli (smoothing=1) | 2.4544 |
| sinir agi (200 adim) | 2.4830 |

Iki yol ayni yere cikiyor ve ayni isimleri uretiyor.

Turkce (isimler.txt, 236.096 isim, 29 harfli alfabe):

| model | loss |
|---|---|
| sayim modeli | 2.5830 |
| sinir agi | 2.6118 |

Turkce loss'un biraz yuksek olmasi normal: alfabe 27 yerine 30 sembol
ve liste daha karisik (farkli kokenlerden isimler var).

Ek gorev (trigram, test seti):

| model | test loss |
|---|---|
| bigram | 2.4584 |
| trigram | 2.2240 |

## Turkce veri

Kaynak: https://github.com/Stealeristaken/Turkce-Isimler
Ham liste buyuk harfliydi. Kucuk harfe cevirdim (I -> i, I -> i ayrimina
dikkat ederek), 3-12 harf disindakileri ve Turk alfabesi disinda karakter
iceren isimleri attim, tekrarlari temizledim. Kalan: 236.096 isim.

## Gecen haftalarla baglanti

Hafta 2'de micrograd'da kendi backward() metodumu yazmistim. Gorev 4'te
PyTorch'un loss.backward() cagrisi tam olarak ayni isi yapiyor, sadece
tek tek sayilar yerine tensor'larla. Hafta 1'deki gradient descent
dongusu de aynen duruyor.
