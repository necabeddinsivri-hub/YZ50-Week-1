# YZ50 - Hafta 4: MLP Dil Modeli (Bengio 2003)

Karpathy'nin makemore Part 2 ve Part 3 videolarini takip ederek kurdugum
ilk gercek neural network dil modeli. Tablo yok: her harf ogrenilen kucuk
bir vektore (embedding) donusuyor, onceki uc harfin vektorleri yan yana
konup bir MLP'ye veriliyor.

## Dosyalar

- `ortak.py`            - veri yukleme, baglam penceresi, train/dev/test bolme
- `gorev1_veri.py`      - baglam penceresi veri seti + embedding tablosu
- `gorev2_model.py`     - gizli katman, logits, elle loss vs F.cross_entropy
- `gorev3_egitim.py`    - overfit testi, learning rate taramasi, asil egitim
- `gorev4_buyutme.py`   - model buyutme, embedding gorsellestirme, isim uretme
- `gorev5_init.py`      - baslangic loss'u, tanh doymasi, Kaiming init
- `gorev6_batchnorm.py` - BatchNorm katmani ve karsilastirma
- `gorev7_turkce.py`    - ayni modeli Turkce isimlerle egitme
- `gorev8_ek.py`        - (ek) BatchNorm'u Linear katmana katlama (Part 3 E02)
- `names.txt`           - Karpathy'nin Ingilizce listesi (32.033 isim)
- `isimler.txt`         - temizlenmis Turkce liste (236.096 isim)
- `deney_notu.md`       - butun sonuclar ve gozlemler

## Calistirma

    pip install torch matplotlib

    python gorev1_veri.py
    python gorev2_model.py
    python gorev3_egitim.py
    python gorev4_buyutme.py
    python gorev5_init.py
    python gorev6_batchnorm.py
    python gorev7_turkce.py
    python gorev8_ek.py

matplotlib yoksa kod yine calisir, sadece embedding.png uretilmez.
Terminal icinde metin tabanli embedding haritasi zaten basiliyor.

## Ana sonuclar

Ingilizce (dev loss):

| model | dev loss |
|---|---|
| Hafta 3 bigram sayim | 2.4544 |
| Hafta 3 bigram sinir agi | 2.4830 |
| Hafta 4 MLP (emb=2, gizli=100) | 2.2668 |
| Hafta 4 MLP (emb=10, gizli=300) | 2.1167 |
| Hafta 4 MLP + Kaiming init | 2.1486 |

Turkce (dev loss):

| model | dev loss |
|---|---|
| Hafta 3 bigram sayim | 2.5830 |
| Hafta 4 MLP | 2.3183 |

Init karsilastirmasi (Gorev 5):

| init | ilk loss | dev loss | doymus neuron |
|---|---|---|---|
| kotu (hepsi rastgele) | 26.36 | 2.2657 | %61.6 |
| son katman ~0 | 3.32 | 2.1889 | %61.6 |
| Kaiming + son katman ~0 | 3.32 | 2.1486 | %10.0 |

Beklenen baslangic loss'u -log(1/27) = 3.2958.

## Uretilen isimler

Ingilizce MLP: careah, ami, hari, thay, sranden, den, nellara, kaleig, jor

Turkce MLP: ziye, bizer, semim, pire, gunayse, sen, sal, baz, belli, elayiz

Hafta 3 bigram (karsilastirma): cexze, konimittain, llayn, ka, da

## Gecen haftalarla baglanti

- Hafta 1: egitim dongusunun bes adimi aynen duruyor.
- Hafta 2: micrograd'da yazdigim `1 - t**2` satiri, bu haftaki tanh
  doymasinin tam sebebi. Doymus neuron'un turevi sifira gidiyor.
- Hafta 3: bigram tablosu 27x27 idi; trigram'da 19.683 hucreye cikiyordu.
  Bu hafta tablo yok, embedding var: 3 harf baglam yalnizca 12.530
  parametre ile calisiyor ve dev loss belirgin sekilde daha iyi.
