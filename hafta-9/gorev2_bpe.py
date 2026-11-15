# Gorev 2 - BPE'yi sifirdan yaz ve kendi Turkce metninde egit
# YZ50 Hafta 9  (video 0:23:50 - 0:39:20)
#
# Veri: Hafta 8'de kendim derledigim turkce.txt (1.050.074 bayt).

import json
import time

from bpe import BasicTokenizer, goster, istatistik, birlestir

VOCAB = 1024          # 256 taban byte + 768 birlestirme

metin = open('turkce.txt', encoding='utf-8').read()
ham = metin.encode('utf-8')

print("=" * 74)
print("2a) VERI")
print("=" * 74)
print(f"  dosya           : turkce.txt (Hafta 8'de kendim derledim)")
print(f"  karakter sayisi : {len(metin):,}".replace(',', '.'))
print(f"  byte sayisi     : {len(ham):,}".replace(',', '.'))
print(f"  farkli karakter : {len(set(metin))}   (Hafta 8'deki alfabem)")
print(f"  farkli byte     : {len(set(ham))}   (BPE'nin tabani)")
print()
print("  Dikkat: karakter sayisi ile byte sayisi arasinda "
      f"{len(ham) - len(metin):,} fark var,".replace(',', '.'))
print(f"  yani yuzde {100*(len(ham)-len(metin))/len(metin):.1f}. Bu fark tamamen Turkce'nin")
print("  isaretli harflerinden geliyor (Gorev 1).")
print()


# ---------------------------------------------------------------------
# 2b) ALGORITMAYI KUCUK BIR ORNEKTE ELLE IZLE
# ---------------------------------------------------------------------
print("=" * 74)
print("2b) ALGORITMA, KUCUK BIR ORNEKTE")
print("=" * 74)
kucuk = "ağaçağaçağaç"
ids = list(kucuk.encode('utf-8'))
print(f"  metin: {kucuk!r}")
print(f"  byte : {ids}")
print(f"  uzunluk: {len(ids)}")
print()
vocab_k = {i: bytes([i]) for i in range(256)}
for adim in range(4):
    sayac = istatistik(ids)
    ikili = max(sayac, key=sayac.get)
    yeni = 256 + adim
    vocab_k[yeni] = vocab_k[ikili[0]] + vocab_k[ikili[1]]
    print(f"  adim {adim+1}: en sik ikili {ikili} ({sayac[ikili]} kere) "
          f"-> yeni token {yeni} = {goster(vocab_k[yeni])!r}")
    ids = birlestir(ids, ikili, yeni)
    print(f"          yeni dizi ({len(ids)} eleman): {ids}")
print()
print("""  IKI SEY GORUNUYOR

  1. Algoritma byte seviyesinde calistigi icin 'ğ' harfinin iki
     byte'ini (196, 159) birlikte gormuyor. Once 'a' ile 196'yi
     birlestiriyor ve ortaya TEK BASINA GECERLI UTF-8 OLMAYAN bir
     token cikiyor (yukarida kirik karakter olarak gorunuyor).
     Ikinci adimda 159 da eklenince 'ag' yumusak g'li hali
     tamamlaniyor.

  2. Yani ara adimlarda yarim karakterler olusuyor. Bu bir hata degil,
     byte seviyesinde calismanin dogal sonucu - ve Gorev 3'te decode
     yazarken errors='replace' kullanmamin sebeplerinden biri bu.

  Kimse algoritmaya Turkce ogretmedi; sadece sik gecen ikilileri
  birlestirdi ve uc adimda bir heceye ulasti.""")
print()


# ---------------------------------------------------------------------
# 2c) GERCEK EGITIM
# ---------------------------------------------------------------------
print("=" * 74)
print(f"2c) EGITIM  (vocab_size = {VOCAB})")
print("=" * 74)
kayit = []
tok = BasicTokenizer()
t0 = time.time()
tok.train(metin, VOCAB, kayit=kayit)
sure = time.time() - t0
print(f"  {VOCAB - 256} birlestirme, {sure:.1f} saniye")
print()

print("  ILK 20 BIRLESTIRME")
print(f"  {'#':<4}{'ikili':<14}{'->':<4}{'yeni':<7}{'metin':<16}"
      f"{'kac kere':>10}")
print("  " + "-" * 60)
for r in kayit[:20]:
    a, b = r['ikili']
    metin_hali = goster(tok.vocab[r['yeni']])
    adet = f"{r['adet']:,}".replace(',', '.')
    print(f"  {r['sira']:<4}{str(r['ikili']):<14}{'->':<4}{r['yeni']:<7}"
          f"{metin_hali!r:<16}{adet:>10}")
print()
print("  Okunusu: parantez icindeki sayilar token numaralari. 256'nin")
print("  altindakiler ham byte, ustundekiler daha once ogrenilmis")
print("  birlestirmeler. 'metin' sutunu o token'in karsiligi.")
print()

# ilk birlestirmelerin anlamli gruplari
print("  NE BIRLESIYOR?")
ilk40 = [goster(tok.vocab[r['yeni']]) for r in kayit[:40]]
print(f"  Ilk 40 token: {ilk40}")
print()
print("""  Beklentim 'en sik gecen harf ikilileri' idi; cikan sey daha
  ilginc. Ilk birlesenler arasinda Turkce'nin 2 byte'lik harfleri var:
  'ı', 'ş', 'ğ', 'ü' gibi harfler aslinda iki byte ve BPE ilk islerden
  biri olarak onlari TEK token haline getiriyor. Yani algoritma,
  kimse soylemeden once 'Turkce alfabesini' yeniden kesfediyor.

  Sonra sik ekler ve heceler geliyor: 'la', 'in', 'ar', 'ler', 'lar'
  gibi. Turkce sondan eklemeli oldugu icin ekler cok tekrar ediyor ve
  BPE onlari hizla yakaliyor.
""")


# ---------------------------------------------------------------------
# 2d) SIKISTIRMA ORANI
# ---------------------------------------------------------------------
print("=" * 74)
print("2d) SIKISTIRMA ORANI")
print("=" * 74)
# tum metni encode etmek uzun surer; temsili bir dilim aliyorum
dilim = metin[:200_000]
d_byte = len(dilim.encode('utf-8'))
ids = tok.encode(dilim)
print(f"  olculen dilim   : ilk {len(dilim):,} karakter".replace(',', '.'))
print(f"  byte sayisi     : {d_byte:,}".replace(',', '.'))
print(f"  token sayisi    : {len(ids):,}".replace(',', '.'))
print(f"  SIKISTIRMA ORANI: {d_byte/len(ids):.3f}x  (byte / token)")
print()
print(f"  Karakter basina token: {len(ids)/len(dilim):.3f}")
print(f"  Hafta 8'de bu sayi 1.000 idi (her karakter bir token).")
print(f"  Yani ayni block_size'a simdi {len(dilim)/len(ids):.2f} kat daha fazla")
print("  metin sigiyor.")
print()

# egitim boyunca kalan uzunluk
print("  EGITIM BOYUNCA DIZI KISALMASI")
print(f"  {'birlestirme':<14}{'kalan token':>14}{'oran':>10}")
print("  " + "-" * 40)
basl = len(ham)
print(f"  {'0 (ham byte)':<14}{basl:>14,}{'1.000x':>10}".replace(',', '.'))
for n in [50, 100, 200, 400, 768]:
    if n <= len(kayit):
        k = kayit[n - 1]['kalan']
        print(f"  {n:<14}{k:>14,}{basl/k:>9.3f}x".replace(',', '.'))
print()
print("  Kazanc dogrusal degil: ilk 50 birlestirme en buyuk sicramayi")
print("  yapiyor, sonra yavasliyor. Mantikli - en sik gecen ikililer")
print("  once secildigi icin ilk birlestirmeler en cok yeri kapatiyor.")
print()


# ---------------------------------------------------------------------
# 2e) VOCAB BOYUTU TARAMASI
# ---------------------------------------------------------------------
print("=" * 74)
print("2e) VOCAB BOYUTU NE OLMALI?")
print("=" * 74)
tarama = []
orn = metin[:60_000]
o_byte = len(orn.encode('utf-8'))
for v in [256, 512, 1024, 2048, 4096]:
    if v == 256:
        n_tok = o_byte
        s = 0.0
    else:
        t2 = BasicTokenizer()
        t1 = time.time()
        t2.train(metin[:300_000], v)
        s = time.time() - t1
        n_tok = len(t2.encode(orn))
    tarama.append((v, n_tok, o_byte / n_tok, s))
    print(f"  vocab {v:>5}: {n_tok:>7,} token   oran {o_byte/n_tok:>6.3f}x   "
          f"egitim {s:>5.1f} sn".replace(',', '.'))
print()
print("""  Takas net: vocab buyudukce diziler kisaliyor ama embedding
  tablosu buyuyor. Hafta 8'deki modelimde embedding tablosu
  V x 128 parametreydi; V=87 iken 11 bin, V=1024 iken 131 bin,
  V=4096 iken 524 bin parametre. Modelin tamami 816 bindi.

  Ben 1024 sectim: 1 MB'lik bir korpus icin 4096 token'in cogu birkac
  kez gorulur ve iyi egitilemez. GPT-4'un 100 bin token'i var ama onun
  verisi de terabyte olculerinde.
""")

json.dump({
    'vocab': VOCAB,
    'egitim_sn': sure,
    'metin_karakter': len(metin),
    'metin_byte': len(ham),
    'farkli_karakter': len(set(metin)),
    'ilk20': [{'sira': r['sira'], 'ikili': list(r['ikili']), 'yeni': r['yeni'],
               'adet': r['adet'],
               'metin': goster(tok.vocab[r['yeni']]),
               'sol': goster(tok.vocab[r['ikili'][0]]),
               'sag': goster(tok.vocab[r['ikili'][1]])} for r in kayit[:20]],
    'ilk40_metin': ilk40,
    'dilim_karakter': len(dilim), 'dilim_byte': d_byte,
    'dilim_token': len(ids), 'oran': d_byte / len(ids),
    'kisalma': [(0, basl)] + [(n, kayit[n-1]['kalan'])
                              for n in [50, 100, 200, 400, 768]
                              if n <= len(kayit)],
    'tarama': tarama,
}, open('sonuc_gorev2.json', 'w'), ensure_ascii=False, indent=1)

# tokenizer'i kaydet (sonraki gorevler kullaniyor)
json.dump({'merges': [[list(k), v] for k, v in tok.merges.items()]},
          open('tokenizer_temel.json', 'w'))
print("  -> sonuc_gorev2.json ve tokenizer_temel.json kaydedildi")
