# Hafta 2 - Deney Notu

## Ne yaptim

Gecen hafta gradient'i sayisal olarak buluyordum: parametreyi cok az
oynatip loss'un ne kadar degistigine bakiyordum. Bu hafta ayni degeri
tek bir geri gecisde hesaplayan Value sinifini yazdim.

## Ogrendigim uc sey

1. Toplama, gradient'i degistirmeden iki girdiye de kopyaliyor.
2. Carpma, gradient'i digerinin degeriyle carpiyor.
3. tanh, gradient'i her zaman kisarak geciriyor (1 - t^2 hep 1'den kucuk).

## Gradient sonuclari

Ornek 1 (L = (a*b + c) * f):

| dugum | grad | elle hesapladigim |
|---|---|---|
| a | 6.0 | 6.0 |
| b | -4.0 | -4.0 |
| c | -2.0 | -2.0 |
| e | -2.0 | -2.0 |
| d | -2.0 | -2.0 |
| f | 4.0 | 4.0 |

Ornek 2 (tek neuron):

| dugum | grad | elle hesapladigim |
|---|---|---|
| n | 0.5 | 0.5 |
| x1 | -1.5 | -1.5 |
| w1 | 1.0 | 1.0 |
| x2 | 0.5 | 0.5 |
| w2 | 0.0 | 0.0 |

w2'nin gradient'i sifir cikti. Sebebi x2 = 0 olmasi, yani w2 ne olursa
olsun cikti degismiyor. Bunu ilk basta hata sandim, sonra anladim.

## Dogrulama

- tanh'i parcalara ayirdim (exp, toplama, bolme ile). Fark: 4.4e-16.
- Numerical yontemle karsilastirdim. Fark: 3.2e-06.

PyTorch ile de karsilastirmak istedim ama kuramadim, onu bu hafta
yapamadim.

## Egitim

MLP(3, [4,4,1]), 41 parametre, 100 adim, learning rate 0.05.

| adim | 0 | 10 | 20 | 50 | 99 |
|---|---|---|---|---|---|
| loss | 6.005 | 0.506 | 0.054 | 0.012 | 0.005 |

Loss tam sifira inmedi. Once hata sandim ama tanh cikisi +-1'e tam
ulasamadigi icin normalmis.

## Zorlandigim yerler

- Gradient'leri '+=' ile toplamak gerektigini once anlamamistim,
  '=' yazmistim ve b = a + a testinde 2 yerine 1 cikiyordu.
- backward() icindeki siralamayi anlamak zaman aldi.
- Zamanim yetmedigi icin zero grad'i kaldirip deneyemedim, learning
  rate denemelerini de yapamadim. Bunlari onumuzdeki gunlerde
  deneyecegim.
