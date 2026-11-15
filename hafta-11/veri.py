# veri.py - metni batch'lere bolen data loader
# YZ50 Hafta 11  (video 1:02:00 - 1:22:18)
#
# Hafta 10'da modeli TEK bir batch'e ezberletmistim: ayni x, y ciftini
# 300 kez verip loss'u 10.96'dan 0.00075'e dusurmustum. O bir saglik
# testiydi, egitim degil. Bu hafta metnin TAMAMINI gormesi gerekiyor,
# yani bir siradaki batch'i veren bir seye ihtiyacim var.
#
# Isi su kadar: token dizisini B*T'lik dilimler halinde sirayla ver,
# sona gelince basa don.

import os

import torch


class VeriYukleyici:
    """Metni bir kez token'lara cevirir, sonra sirayla batch verir.

    Hafta 8'deki Veri sinifindan farki: orada her cagrida RASTGELE
    yerlerden batch cekiyordum (torch.randint ile). Burada SIRAYLA
    gidiyorum. Ikisi de calisir ama sirali olanin iki avantaji var:
      - metnin tamamini gordugumu garanti ediyorum (rastgele cekimde
        bazi parcalari hic gormeme ihtimali var)
      - "bir devir (epoch) bitti" diyebiliyorum, yani ne kadar veri
        gorduğumu sayabiliyorum
    """

    def __init__(self, metin_yolu, B, T, kodlayici, sessiz=False,
                 onbellek=None):
        self.B, self.T = B, T
        with open(metin_yolu, 'r', encoding='utf-8') as f:
            metin = f.read()
        self.karakter = len(metin)

        # Token'lara cevirmek pahali bir is; bir kez yapip diske
        # yaziyorum. Hafta 9'da yazdigim tokenizer'la 1 MB metni
        # kodlamak dakikalar aliyor, her kosuda tekrar etmek anlamsiz.
        if onbellek and os.path.exists(onbellek):
            tokenler = torch.load(onbellek)
            if not sessiz:
                print(f"  onbellekten okundu: {onbellek}")
        else:
            tokenler = torch.tensor(kodlayici(metin), dtype=torch.long)
            if onbellek:
                torch.save(tokenler, onbellek)
        self.tokenler = tokenler

        self.pozisyon = 0
        self.devir = 0
        self.verilen_token = 0
        self.batch_basi = B * T
        # +1 cunku y, x'in bir token kaydirilmis hali; son batch icin
        # bir token fazla okumam lazim
        self.devir_basi_batch = (len(self.tokenler) - 1) // self.batch_basi

        if not sessiz:
            from bicim import bin as _b
            print(f"  metin   : {metin_yolu}  ({_b(self.karakter)} karakter)")
            print(f"  token   : {_b(len(self.tokenler))}"
                  f"  ({len(self.tokenler)/self.karakter:.3f} token/karakter)")
            print(f"  batch   : B={B}, T={T} -> {self.batch_basi} token/batch")
            print(f"  bir devirde {self.devir_basi_batch} batch var")

    def sonraki(self):
        """Siradaki (x, y) ciftini dondurur."""
        B, T = self.B, self.T
        parca = self.tokenler[self.pozisyon: self.pozisyon + B * T + 1]
        x = parca[:-1].view(B, T)       # girdi
        y = parca[1:].view(B, T)        # hedef: girdinin bir token sonrasi
        self.pozisyon += B * T
        self.verilen_token += B * T
        # Sona gelmeye az kaldiysa basa don. "+1" kontrolu onemli:
        # son dilimde y icin bir token daha gerekiyor.
        if self.pozisyon + (B * T + 1) > len(self.tokenler):
            self.pozisyon = 0
            self.devir += 1
        return x, y

    def basa_al(self):
        self.pozisyon = 0
        self.devir = 0
        self.verilen_token = 0

    @property
    def gorulen_karakter(self):
        """Simdiye kadar kac KARAKTER metin gordum.

        Gorev 4'te iki tokenizer'i karsilastirirken bu lazim: token
        sayisi tokenizer'a gore degisiyor, karakter sayisi degismiyor.
        Ortak olcek karakter.
        """
        return self.verilen_token * self.karakter / len(self.tokenler)
