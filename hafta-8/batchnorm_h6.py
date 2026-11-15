# batchnorm_h6.py - Hafta 6'da yazdigim BatchNorm1d, AYNEN
# YZ50 Hafta 8, Gorev 2'de LayerNorm ile karsilastirmak icin buraya kopyaladim.
# Tek satirini degistirmedim ki karsilastirma durust olsun.

import torch


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1, duzeltilmis=True):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.duzeltilmis = duzeltilmis
        # ogrenilen parametreler
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # cikarim zamani icin biriktirilen istatistikler
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                # Hafta 6'daki hata: 3 boyutlu girdide mean(0) almak
                # (B,T,C)'de B boyunca ortalama alir, T'yi birakir.
                # Dogrusu (0,1): hem batch hem zaman boyunca.
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
                self.running_mean = ((1 - self.momentum) * self.running_mean
                                     + self.momentum * xmean.squeeze())
                self.running_var = ((1 - self.momentum) * self.running_var
                                    + self.momentum * xvar.squeeze())
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
