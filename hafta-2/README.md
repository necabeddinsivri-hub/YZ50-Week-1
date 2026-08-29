# YZ50 - Hafta 2

Karpathy'nin backpropagation videosunu takip ederek yazdigim mini autograd motoru.

## Dosyalar

- `value.py` - Value sinifi, toplama/carpma/tanh/exp/pow ve backward()
- `ornekler.py` - videodaki iki ornegin gradient'lerini elle hesaplayip kodla karsilastirdim
- `egitim.py` - Neuron, Layer, MLP ve egitim dongusu
- `deney_notu.md` - sonuclar ve gozlemlerim

## Calistirma

    python ornekler.py
    python egitim.py

Ek paket gerekmiyor, sadece Python.

## Sonuclar

Ornek 1 ve 2'de elle hesapladigim gradient'lerin hepsi kodla tutuyor.
tanh'i parcalara ayirdigimda ayni gradient'i aldim (fark 4.4e-16).
Numerical yontemle fark 3.2e-06.

MLP(3, [4,4,1]) = 41 parametre. 100 adim sonunda loss 6.005 -> 0.0051.
Tahminler [0.959, -0.992, -0.96, 0.958], hedefler [1, -1, -1, 1].

## Notum

Bu hafta benim icin zor gecti, gecen haftaki numerical yontemden
buraya gecis biraz zamanimi aldi. Kodu calisir hale getirdim ama
bazi yerleri hala tam oturtmus degilim, uzerinde calismaya devam
ediyorum.
