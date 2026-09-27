# katmanlar.py - torch.nn'in yaptigi isi kendim yaziyorum
# YZ50 Hafta 6
#
# Her katman ayni sozlesmeye uyuyor:
#   katman(x)            -> ciktiyi hesapla
#   katman.parameters()  -> ogrenilebilir tensor'larin listesi
# Egitim dongusu katmanlarin adini bilmiyor, sadece bu sozlesmeyi biliyor.

import torch


class Linear:
    def __init__(self, fan_in, fan_out, bias=True, g=None):
        # Kaiming benzeri olcek (Hafta 4)
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1, duzeltilmis=True):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.duzeltilmis = duzeltilmis   # Gorev 4 icin: False = hatali hali
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                # DOGRU: hem batch hem zaman ekseninde ortalama -> (0, 1)
                # HATALI: sadece batch ekseninde -> 0
                dim = (0, 1) if self.duzeltilmis else 0
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim, g=None):
        self.weight = torch.randn((num_embeddings, embedding_dim), generator=g)

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten:
    """Butun baglami tek vektore duzlestirir (duz MLP icin)."""
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class FlattenConsecutive:
    """Ardisik n elemani birlestirir (WaveNet icin).
    (B, T, C) -> (B, T//n, C*n)   ve T//n == 1 ise (B, C*n)
    """
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def egitim_kipi(self, acik):
        """training bayragini, bayragi olan butun katmanlara iletir."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = acik
