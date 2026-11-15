# veri.py - dosyalara (shard) bolunmus token'lari okuyan data loader
# YZ50 Hafta 12
#
# Hafta 11'deki VeriYukleyici butun metni bellege alip sirayla geziyordu.
# Bu hafta veri birden cok dosyada duruyor; yukleyici bir dosyayi
# bitirince digerine geciyor, hepsi bitince basa donuyor. Karpathy'nin
# videodaki DataLoaderLite'inin yaptigi da bu.

import os

import numpy as np
import torch

KLASOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'veri', 'shardlar')


def dosya_yukle(yol):
    # uint16 diskte yer kazandiriyor ama embedding indeksi icin long lazim
    return torch.tensor(np.load(yol).astype(np.int64), dtype=torch.long)


class ShardYukleyici:
    def __init__(self, B, T, split, klasor=KLASOR, onek='edufineweb'):
        assert split in ('train', 'val')
        self.B, self.T = B, T
        self.dosyalar = sorted(
            os.path.join(klasor, f) for f in os.listdir(klasor)
            if f.startswith(f'{onek}_{split}_') and f.endswith('.npy'))
        assert self.dosyalar, f'{split} icin dosya yok: {klasor}'
        self.toplam_token = sum(len(np.load(f, mmap_mode='r'))
                                for f in self.dosyalar)
        self.basa_al()

    def basa_al(self):
        self.dosya_no = 0
        self.tokenler = dosya_yukle(self.dosyalar[0])
        self.pozisyon = 0
        self.verilen_token = 0
        self.devir = 0

    def sonraki(self):
        B, T = self.B, self.T
        parca = self.tokenler[self.pozisyon: self.pozisyon + B * T + 1]
        x = parca[:-1].view(B, T)
        y = parca[1:].view(B, T)          # bir token kaydirilmis hedef
        self.pozisyon += B * T
        self.verilen_token += B * T
        # Bu dosyada bir batch'lik (+1) token kalmadiysa sonraki dosyaya gec
        if self.pozisyon + B * T + 1 > len(self.tokenler):
            self.dosya_no += 1
            if self.dosya_no == len(self.dosyalar):
                self.dosya_no = 0
                self.devir += 1
            self.tokenler = dosya_yukle(self.dosyalar[self.dosya_no])
            self.pozisyon = 0
        return x, y
