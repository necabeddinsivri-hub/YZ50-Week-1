# YZ50 - Hafta 5: Backprop Ninja

Gecen haftaki MLP + BatchNorm modelinin butun gradient'lerini
loss.backward() kullanmadan, elle hesapladim. Sonra ayni modeli
sadece kendi gradient'lerimle egittim.

## Dosyalar

- `gorev1_kurulum.py`   - modeli 20 kucuk adima bolup ara degiskenlerin
                          gradient'lerini PyTorch'tan alma
- `gorev2_elle.py`      - 26 gradient'i elle yazma ve cmp ile dogrulama
- `gorev3_tek_satir.py` - (ek) cross entropy ve BatchNorm turevlerini tek
                          ifadeye indirme + loss.backward() olmadan egitim
- `names.txt`           - Ingilizce isim listesi
- `isimler.txt`         - Turkce isim listesi (Hafta 3'ten)

## Calistirma

    pip install torch

    python gorev1_kurulum.py     # ~5 sn
    python gorev2_elle.py        # ~10 sn
    python gorev3_tek_satir.py   # ~6 dk (egzersiz 4 egitim iceriyor)

## Sonuclar

Gorev 2: 26 gradient'in tamami dogrulandi.

| grup | durum |
|---|---|
| logprobs, probs, counts, counts_sum, counts_sum_inv | EXACT |
| norm_logits, logit_maxes, logits | EXACT |
| h, W2, b2 | EXACT |
| hpreact, bngain, bnraw, bnbias | approx (<2e-09) |
| bnvar_inv, bnvar, bndiff2, bndiff, bnmeani, hprebn | approx (<4e-09) |
| embcat, W1, b1, emb, C | approx (<6e-09) |

EXACT olanlar tam sayisal esitlik. approx olanlarda fark 1e-9 civari,
yani float64 hassasiyeti seviyesinde; islem sirasi farkindan kaynaklaniyor.

Gorev 3 (ek):

| egzersiz | sonuc |
|---|---|
| E2: cross entropy tek satir | maks fark 1.86e-09 |
| E3: BatchNorm tek satir | maks fark 4.66e-10 |
| E4: elle gradient'lerle egitim | dev loss 2.2287 |

Karsilastirma:

| model | dev loss |
|---|---|
| Hafta 3 bigram | 2.4544 |
| Hafta 4 MLP (autograd) | 2.1486 |
| Hafta 5 MLP (elle gradient) | 2.2287 |

Elle egitilen modelin urettigi isimler:
carmah, amille, khirmin, reety, salaysa, ner, kiah, maiiv

## Gecen haftalarla baglanti

- Hafta 2: micrograd'da skaler Value'lar icin backward() yazmistim.
  Bu hafta aynisini tensor'lar icin yaptim. Fark: artik sekiller ve
  broadcasting var.
- Hafta 4: `1 - h**2` satiri (tanh turevi) bu hafta yine karsima cikti,
  tek satir olarak: dhpreact = (1.0 - h**2) * dh
- Hafta 4'te BatchNorm'u kullandim ama icini bilmiyordum. Bu hafta
  turevini cikardim ve neden bir ornegin digerine bagli oldugunu
  matematiksel olarak gordum.
