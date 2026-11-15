# Gorev 3 - Modeli olcekle ve dropout ekle
# YZ50 Hafta 8  (video 1:37:49 - 1:42:39)
#
# Karpathy videoda GPU'da n_embd=384, 6 head, 6 blok, block=256, batch=64,
# 5000 adim egitiyor. Bende GPU yok: 2 CPU cekirdegi var. O config bu
# makinede saatler surer. O yuzden once OLCUYORUM, sonra seciyorum.
#
# Calistirma:
#   python3 gorev3_buyut.py olcum     -> aday config'lerin hizini olcer
#   python3 gorev3_buyut.py egit      -> secilen config'i egitir

import json
import sys
import time
import torch
from ortak import veri, V, loss_olc, egit, decode, param_sayisi
from katmanlar import GPTDilModeli, uret

# ---------------------------------------------------------------------
# SECILEN CONFIG  (olcum sonucuna gore dolduruldu - rapora bak)
# ---------------------------------------------------------------------
CONFIG = dict(n_embd=128, n_head=4, n_blok=4, block=64, dropout=0.1)
BATCH = 32
ADIM = 4000
LR = 3e-4            # model buyudu -> daha kucuk lr (Hafta 3'teki sweep dersi)

ADAYLAR = [
    dict(ad='A (kucuk, hafta 7 ayari)', n_embd=32, n_head=4, n_blok=3, block=8),
    dict(ad='B', n_embd=64, n_head=4, n_blok=3, block=16),
    dict(ad='C', n_embd=96, n_head=4, n_blok=3, block=32),
    dict(ad='D (secilen)', n_embd=128, n_head=4, n_blok=4, block=64),
    dict(ad='E (Karpathy GPU ayari)', n_embd=384, n_head=6, n_blok=6, block=256),
]


def olcum():
    print("=" * 74)
    print("DONANIM OLCUMU - hangi config bu makinede kaliyor?")
    print("=" * 74)
    import os
    print(f"  CPU cekirdegi : {os.cpu_count()}")
    print(f"  CUDA          : {torch.cuda.is_available()}")
    print()
    print(f"  {'config':<26}{'param':>10}{'ms/adim':>10}{'3000 adim':>12}")
    print("  " + "-" * 58)
    for a in ADAYLAR:
        kw = {k: v for k, v in a.items() if k != 'ad'}
        torch.manual_seed(1337)
        m = GPTDilModeli(dropout=0.1, **kw)
        p = param_sayisi(m)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        g = torch.Generator().manual_seed(1)
        n = 12 if a['n_embd'] < 200 else 3
        # isinma
        xb, yb = veri.get_batch('train', block=kw['block'], batch=BATCH, g=g)
        _, l = m(xb, yb); opt.zero_grad(); l.backward(); opt.step()
        t0 = time.time()
        for _ in range(n):
            xb, yb = veri.get_batch('train', block=kw['block'], batch=BATCH, g=g)
            _, l = m(xb, yb)
            opt.zero_grad(set_to_none=True); l.backward(); opt.step()
        ms = (time.time() - t0) / n * 1000
        tah = ms * 3000 / 1000 / 60
        print(f"  {a['ad']:<26}{p:>10,}{ms:>10.1f}{tah:>10.1f} dk"
              .replace(',', '.'))
        del m, opt
    print("  " + "-" * 58)
    print()
    print("  Karar: D  (n_embd=128, 4 head, 4 blok, block=64).")
    print()
    print("  Neden D:")
    print("   - 4000 adim ~14 dk. Uc kosu (dropout yok / dropout 0.1 /")
    print("     Turkce) toplam ~45 dk. Bir oturumda bitiyor, sonuclari")
    print("     rapora koyabiliyorum. C daha hizli ama kapasitesi bosa")
    print("     gidiyordu; D sinirin hemen altinda.")
    print("   - block=64: hafta 7'de 8 karakterdi. 64 karakter Shakespeare")
    print("     metninde yaklasik iki satir, yani model artik konusan")
    print("     kisinin adini gorup satirin devamini ona gore kurabiliyor.")
    print("     Hafta 6'daki dersin aynisi: baglami uzatmak tek basina")
    print("     en buyuk kazanci veriyordu.")
    print("   - n_embd=128, 4 head -> head basina 32 boyut; hafta 7'deki")
    print("     tek head ile tam ayni buyuklukte. Yani head basina")
    print("     kapasiteyi sabit tutup sayisini 4'e cikardim.")
    print("   - n_blok=4: residual + LayerNorm oldugu icin 4 blok hala")
    print("     rahat egitiliyor (Gorev 2'de residual olmadan 3 blogun")
    print("     bozuldugunu gormustum).")
    print("   - dropout=0.1: makaledeki deger. Model 816 bin parametreye")
    print("     cikinca 1 MB'lik veri icin fazla buyuk; asiri uyumu")
    print("     olcup gostermek icin ikisini de egitiyorum.")
    print()
    print("  Neden E degil: Karpathy'nin ayari (10,8 milyon parametre)")
    print("  bu makinede 3000 adim icin ~305 dakika, yani 5 saat. GPU'su")
    print("  olmayan bir makinede o config calistirilamaz. Kuculturken")
    print("  neyi feda ettigimi biliyorum: onun val loss'u ~1,48'e kadar")
    print("  iniyor, benimki inmeyecek. O yuzden benim sayilarimi sadece")
    print("  kendi aralarinda karsilastiriyorum.")
    print()


def egitim():
    sonuc = json.load(open('sonuclar.json'))
    print("=" * 74)
    print("GOREV 3 - OLCEKLENMIS MODEL")
    print("=" * 74)
    print(f"  n_embd   : {CONFIG['n_embd']}")
    print(f"  n_head   : {CONFIG['n_head']}  (head basina "
          f"{CONFIG['n_embd'] // CONFIG['n_head']} boyut)")
    print(f"  n_blok   : {CONFIG['n_blok']}")
    print(f"  block    : {CONFIG['block']}")
    print(f"  dropout  : {CONFIG['dropout']}")
    print(f"  batch    : {BATCH}")
    print(f"  adim     : {ADIM}   lr: {LR}")
    print()

    # --- dropout'suz ve dropout'lu, ayni config ---
    for etiket, dp in [('dropout YOK', 0.0), ('dropout 0.1', CONFIG['dropout'])]:
        kw = dict(CONFIG); kw['dropout'] = dp
        torch.manual_seed(1337)
        m = GPTDilModeli(**kw)
        p = param_sayisi(m)
        print("-" * 74)
        print(f"  {etiket}  ({p:,} parametre)".replace(',', '.'))
        print("-" * 74)
        sure = egit(m, adim=ADIM, lr=LR, block=CONFIG['block'], batch=BATCH,
                    her=500)
        s = loss_olc(m, block=CONFIG['block'], batch=BATCH)
        print(f"  SONUC: train {s['train']:.4f}  val {s['val']:.4f}  "
              f"asiri uyum farki {s['val'] - s['train']:+.4f}  "
              f"({sure / 60:.1f} dk)")
        print()
        anahtar = '7_olcekli_dropoutsuz' if dp == 0.0 else '8_olcekli_dropout'
        sonuc[anahtar] = dict(
            ad=f"olcekli ({etiket})", param=p, train=s['train'], val=s['val'],
            sure=sure, config=dict(kw, batch=BATCH, adim=ADIM, lr=LR))
        if dp != 0.0:
            gg = torch.Generator().manual_seed(42)
            ornek = decode(uret(m, torch.zeros((1, 1), dtype=torch.long), 1000,
                                g=gg)[0].tolist())
            print("  --- 1000 karakter ornek ---")
            print(ornek)
            print()
            sonuc['_ornek_shakespeare_buyuk'] = ornek
            torch.save(m.state_dict(), 'gpt_shakespeare.pt')
        # her kosudan sonra kaydet: uzun egitimler yarida kalirsa
        # elde edilen sayilar kaybolmasin
        json.dump(sonuc, open('sonuclar.json', 'w'), indent=1,
                  ensure_ascii=False)

    a = sonuc['7_olcekli_dropoutsuz']
    b = sonuc['8_olcekli_dropout']
    print("=" * 74)
    print("DROPOUT NE YAPTI?")
    print("=" * 74)
    print(f"  {'':<16}{'train':>10}{'val':>10}{'val-train':>12}")
    print(f"  {'dropout YOK':<16}{a['train']:>10.4f}{a['val']:>10.4f}"
          f"{a['val'] - a['train']:>12.4f}")
    print(f"  {'dropout 0.1':<16}{b['train']:>10.4f}{b['val']:>10.4f}"
          f"{b['val'] - b['train']:>12.4f}")
    print()
    print(f"  val farki: {a['val'] - b['val']:+.4f}")
    print()
    print("""  Dropout egitim sirasinda noronlarin bir kismini rastgele kapatiyor.
  Her adimda model aslinda biraz farkli bir ag oluyor; sonunda bu
  aglarin ortalamasi gibi davraniyor. Beklenti: train loss biraz
  YUKSELIR (modelin isi zorlasti) ama val-train farki DARALIR.

  Hafta 4'te BatchNorm'un dev loss'u biraz kotulestirdigini durustce
  yazmistim. Burada da ayni olcutle bakiyorum: val loss gercekten
  dustu mu, yoksa sadece train loss mu yukseldi.""")

    sonuc['_config'] = dict(CONFIG, batch=BATCH, adim=ADIM, lr=LR)
    json.dump(sonuc, open('sonuclar.json', 'w'), indent=1, ensure_ascii=False)
    print("  -> sonuclar.json guncellendi")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'olcum':
        olcum()
    else:
        egitim()
