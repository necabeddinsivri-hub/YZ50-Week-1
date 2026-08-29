# Gorev 5 - Neuron, Layer, MLP ve egitim dongusu
# YZ50 Hafta 2

import random
from value import Value

random.seed(1337)   # her calistirmada ayni sonuc ciksin diye


class Neuron:

    def __init__(self, girdi_sayisi):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(girdi_sayisi)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        toplam = self.b
        for wi, xi in zip(self.w, x):
            toplam = toplam + wi * xi
        return toplam.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, girdi_sayisi, neuron_sayisi):
        self.neurons = [Neuron(girdi_sayisi) for _ in range(neuron_sayisi)]

    def __call__(self, x):
        ciktilar = [n(x) for n in self.neurons]
        if len(ciktilar) == 1:
            return ciktilar[0]
        return ciktilar

    def parameters(self):
        hepsi = []
        for n in self.neurons:
            hepsi = hepsi + n.parameters()
        return hepsi


class MLP:

    def __init__(self, girdi_sayisi, katmanlar):
        boyutlar = [girdi_sayisi] + katmanlar
        self.layers = []
        for i in range(len(katmanlar)):
            self.layers.append(Layer(boyutlar[i], boyutlar[i + 1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        hepsi = []
        for layer in self.layers:
            hepsi = hepsi + layer.parameters()
        return hepsi


# --- veri (videodaki kucuk set) ---
xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]


model = MLP(3, [4, 4, 1])
print("parametre sayisi:", len(model.parameters()), " (41 bekliyordum)")
print()

for adim in range(100):

    # 1) ileri gecis
    tahminler = [model(x) for x in xs]

    # 2) loss (kare fark)
    loss = Value(0.0)
    for hedef, tahmin in zip(ys, tahminler):
        loss = loss + (tahmin - hedef) ** 2

    # 3) gradient'leri sifirla (videodaki meshur bug)
    for p in model.parameters():
        p.grad = 0.0

    # 4) geri gecis
    loss.backward()

    # 5) guncelle
    for p in model.parameters():
        p.data = p.data - 0.05 * p.grad

    if adim % 10 == 0:
        print("adim", adim, " loss =", round(loss.data, 6))

print("adim 99  loss =", round(loss.data, 6))
print()
print("tahminler:", [round(model(x).data, 3) for x in xs])
print("hedefler :", ys)
