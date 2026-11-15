# Gorev 5 - Turkce ile Ingilizce'yi karsilastir
# YZ50 Hafta 9
#
# Ayni paragrafin iki dildeki halini GPT-4'un tokenizer'iyla (cl100k_base)
# token'liyorum. Sonra ayni Turkce paragrafi kendi tokenizer'imla.

import json

from bpe import BasicTokenizer, RegexTokenizer, goster
from gpt4_yukle import encoding

# Ayni icerik, iki dil. Teknik olmayan, gundelik bir metin sectim ki
# sonuc ozel terimlerden etkilenmesin.
TR = ("Geçen hafta küçük bir dil modeli eğittim. Model önce metni harf "
      "harf okuyordu, şimdi kelime parçaları halinde okuyor. Çalıştırdığım "
      "bilgisayarda ekran kartı olmadığı için modeli küçülttüm. Yine de "
      "ürettiği cümlelerin yapısı şaşırtıcı derecede düzgün çıktı. "
      "Öğrendiğim en önemli şey, bir sayıyı ancak ölçüm koşulları aynıysa "
      "karşılaştırabileceğim oldu.")

EN = ("Last week I trained a small language model. The model used to read "
      "the text letter by letter, and now it reads it in word pieces. "
      "Because the computer I ran it on has no graphics card, I made the "
      "model smaller. Even so, the structure of the sentences it produced "
      "turned out surprisingly clean. The most important thing I learned "
      "is that I can only compare a number when the measurement conditions "
      "are the same.")

enc = encoding()
sonuc = {}


def olc(ad, metin, tokenle):
    ids = tokenle(metin)
    kelime = len(metin.split())
    byte = len(metin.encode('utf-8'))
    return dict(ad=ad, karakter=len(metin), byte=byte, kelime=kelime,
                token=len(ids), token_kelime=len(ids) / kelime,
                byte_token=byte / len(ids), ids=ids)


print("=" * 78)
print("5a) GPT-4 TOKENIZER (cl100k_base) - TURKCE vs INGILIZCE")
print("=" * 78)
tr = olc('Turkce', TR, enc.encode)
en = olc('Ingilizce', EN, enc.encode)
print(f"  {'':<14}{'karakter':>10}{'byte':>8}{'kelime':>8}{'token':>8}"
      f"{'token/kelime':>14}{'byte/token':>12}")
print("  " + "-" * 76)
for r in (tr, en):
    print(f"  {r['ad']:<14}{r['karakter']:>10}{r['byte']:>8}{r['kelime']:>8}"
          f"{r['token']:>8}{r['token_kelime']:>14.3f}{r['byte_token']:>12.3f}")
print("  " + "-" * 76)
print(f"  Turkce, kelime basina Ingilizce'nin "
      f"{tr['token_kelime']/en['token_kelime']:.2f} katı kadar token harciyor.")
print(f"  Ayni icerik icin toplam token farki: "
      f"{tr['token']} / {en['token']} = {tr['token']/en['token']:.2f}x")
print()

print("  TURKCE CUMLE NASIL BOLUNUYOR (ilk 70 token)")
parcalar = [enc.decode([i]) for i in tr['ids'][:70]]
print("  " + ' | '.join(repr(p) for p in parcalar))
print()
print("  INGILIZCE CUMLE NASIL BOLUNUYOR (ilk 70 token)")
parcalar_en = [enc.decode([i]) for i in en['ids'][:70]]
print("  " + ' | '.join(repr(p) for p in parcalar_en))
print()

# kelime kelime en kotuler
print("  EN COK TOKEN HARCAYAN KELIMELER")
print(f"  {'Turkce':<22}{'token':>7}   {'Ingilizce':<22}{'token':>7}")
print("  " + "-" * 62)
tr_k = sorted({w: len(enc.encode(w)) for w in set(TR.split())}.items(),
              key=lambda kv: -kv[1])[:8]
en_k = sorted({w: len(enc.encode(w)) for w in set(EN.split())}.items(),
              key=lambda kv: -kv[1])[:8]
for i in range(8):
    a = f"{tr_k[i][0]:<22}{tr_k[i][1]:>7}" if i < len(tr_k) else ' ' * 29
    b = f"{en_k[i][0]:<22}{en_k[i][1]:>7}" if i < len(en_k) else ''
    print(f"  {a}   {b}")
print()

# tek tek ornek kelimeler
print("  ORNEK KELIMELER, TEK TEK")
ciftler = [("ev", "house"), ("evim", "my house"), ("evimde", "in my house"),
           ("evlerimizden", "from our houses"),
           ("çalıştırdığım", "that I ran"),
           ("karşılaştırabileceğim", "that I will be able to compare"),
           ("Çiğdem", "Margaret"), ("İstanbul'da", "in Istanbul")]
print(f"  {'Turkce':<24}{'tk':>4}  {'parcalar':<42}{'Ingilizce':<22}{'tk':>4}")
print("  " + "-" * 100)
for t, e in ciftler:
    it, ie = enc.encode(t), enc.encode(e)
    pt = ''.join(enc.decode([i]) + '|' for i in it)
    print(f"  {t:<24}{len(it):>4}  {pt:<42}{e:<22}{len(ie):>4}")
print()

print("""  NEDEN BOYLE

  Uc sebep ust uste biniyor.

  1) BYTE MALIYETI (Gorev 1). Turkce'nin isaretli harfleri 2 byte.
     cl100k_base'de bu harflerin cogu tek basina bir token bile degil;
     'ğ' ve 'ş' gibi harfler kendi baslarina token olarak gorunuyor
     cunku Ingilizce agirlikli egitim verisinde Turkce kelimelerin
     icindeki bu harfler yeterince sik gecmemis.

  2) EGITIM VERISI DENGESI. BPE hangi dilde cok veri gorduyse o dilin
     kelimelerini tek token yapiyor. cl100k_base'in verisi agirlikli
     olarak Ingilizce; 'house' tek token, 'ev' bile bazen bolunuyor.
     Bu bir tasarim hatasi degil, verinin dogal sonucu.

  3) EKLEMELI YAPI. Turkce'de bir kelime, Ingilizce'de bes kelimenin
     isini goruyor: 'evlerimizden' = 'from our houses'. Ingilizce'de
     bu bes kelimenin her biri sik gectigi icin tek token olmus.
     Turkce'de ise 'evlerimizden' butun olarak nadir, o yuzden
     heceleniyor.

     ILGINC OLAN: ucuncu sebep ilk ikisini KISMEN dengeliyor. Turkce
     kelime basina daha cok token harciyor ama ayni anlami daha az
     kelimeyle anlatiyor. Yukaridaki tabloda toplam token farkinin,
     kelime basina farktan daha kucuk olmasinin sebebi bu.
""")

# ---------------------------------------------------------------------
# 5b) KENDI TOKENIZER'IMLA
# ---------------------------------------------------------------------
print("=" * 78)
print("5b) AYNI TURKCE PARAGRAF, KENDI TOKENIZER'IMLA")
print("=" * 78)
d = json.load(open('tokenizer_temel.json'))
bt = BasicTokenizer()
bt.merges = {tuple(k): v for k, v in d['merges']}
bt._vocab_kur()
d2 = json.load(open('tokenizer_regex.json'))
rt = RegexTokenizer()
rt.merges = {tuple(k): v for k, v in d2['merges']}
rt._vocab_kur()

benim = olc('benim (temel)', TR, bt.encode)
benim_r = olc('benim (regex)', TR, rt.encode)
harf = dict(ad='harf (Hafta 8)', karakter=len(TR), byte=len(TR.encode('utf-8')),
            kelime=len(TR.split()), token=len(TR),
            token_kelime=len(TR) / len(TR.split()),
            byte_token=len(TR.encode('utf-8')) / len(TR), ids=[])

print(f"  {'tokenizer':<20}{'vocab':>8}{'token':>8}{'token/kelime':>14}"
      f"{'byte/token':>12}")
print("  " + "-" * 64)
for r, v in [(harf, 87), (benim, 1024), (benim_r, 1024), (tr, 100256)]:
    print(f"  {r['ad']:<20}{v:>8,}{r['token']:>8}{r['token_kelime']:>14.3f}"
          f"{r['byte_token']:>12.3f}".replace(',', '.'))
print("  " + "-" * 64)
print()
print("  BENIM TOKENIZER'IM BU CUMLEYI NASIL BOLUYOR (ilk 60 token)")
print("  " + ' | '.join(repr(goster(bt.vocab[i])) for i in benim['ids'][:60]))
print()

print(f"""  OKUNUSU — BEKLEDIGIMDEN FARKLI BIR SONUC

  Kendi tokenizer'im 1024 token'lik bir vocab'a sahip; GPT-4'unki
  100.256. Aradaki fark yuz kat. Buna ragmen bu Turkce paragrafta
  IKISI DE {benim_r['token']} token uretti - yani esit ciktilar.

  Ben kendi tokenizer'imin bariz onde olmasini bekliyordum: sonucta
  butun birlestirme butcesini Turkce'ye harcadi, GPT-4 ayni butceyi
  yuzlerce dile bolusturmek zorunda kaldi. Esitlik cikmasinin sebebi
  bence su: GPT-4'un vocab'i benimkinden yuz kat buyuk oldugu icin,
  Turkce'ye ayirdigi KUCUK pay bile mutlak olarak benim butun
  vocab'im kadar. Yani yuzde olarak dezavantajli, sayi olarak degil.

  Dogru okuma: 1 MB Turkce metinle egitilmis 1024 token'lik bir
  tokenizer, terabyte'larca cok dilli veriyle egitilmis 100 bin
  token'lik bir tokenizer'a Turkce'de YETISIYOR. Gecmiyor.

  Takas hala var ve benim aleyhime: benim tokenizer'im Ingilizce'de
  ya da Cince'de kotu olurdu, GPT-4'unki her dilde idare ediyor.
  Ozel amacli bir model icin kendi tokenizer'ini egitmek mantikli
  ama kazanc benim bekledigim kadar buyuk degil.

  Hafta 8'e gore kazanc ise net: harf seviyesinde bu paragraf
  {harf['token']} token'di, simdi {benim_r['token']} token. Ayni
  block_size'a {harf['token']/benim_r['token']:.2f} kat daha fazla
  metin sigiyor. Hafta 8'de block=64 ile yaklasik iki satir
  goruyordum; ayni pencere simdi bu paragrafin neredeyse yarisini
  goruyor.
""")

sonuc = {}


def olc(ad, metin, tokenle):
    ids = tokenle(metin)
    kelime = len(metin.split())
    byte = len(metin.encode('utf-8'))
    return dict(ad=ad, karakter=len(metin), byte=byte, kelime=kelime,
                token=len(ids), token_kelime=len(ids) / kelime,
                byte_token=byte / len(ids), ids=ids)


print("=" * 78)
print("5a) GPT-4 TOKENIZER (cl100k_base) - TURKCE vs INGILIZCE")
print("=" * 78)
tr = olc('Turkce', TR, enc.encode)
en = olc('Ingilizce', EN, enc.encode)
print(f"  {'':<14}{'karakter':>10}{'byte':>8}{'kelime':>8}{'token':>8}"
      f"{'token/kelime':>14}{'byte/token':>12}")
print("  " + "-" * 76)
for r in (tr, en):
    print(f"  {r['ad']:<14}{r['karakter']:>10}{r['byte']:>8}{r['kelime']:>8}"
          f"{r['token']:>8}{r['token_kelime']:>14.3f}{r['byte_token']:>12.3f}")
print("  " + "-" * 76)
print(f"  Turkce, kelime basina Ingilizce'nin "
      f"{tr['token_kelime']/en['token_kelime']:.2f} katı kadar token harciyor.")
print(f"  Ayni icerik icin toplam token farki: "
      f"{tr['token']} / {en['token']} = {tr['token']/en['token']:.2f}x")
print()

print("  TURKCE CUMLE NASIL BOLUNUYOR (ilk 70 token)")
parcalar = [enc.decode([i]) for i in tr['ids'][:70]]
print("  " + ' | '.join(repr(p) for p in parcalar))
print()
print("  INGILIZCE CUMLE NASIL BOLUNUYOR (ilk 70 token)")
parcalar_en = [enc.decode([i]) for i in en['ids'][:70]]
print("  " + ' | '.join(repr(p) for p in parcalar_en))
print()

# kelime kelime en kotuler
print("  EN COK TOKEN HARCAYAN KELIMELER")
print(f"  {'Turkce':<22}{'token':>7}   {'Ingilizce':<22}{'token':>7}")
print("  " + "-" * 62)
tr_k = sorted({w: len(enc.encode(w)) for w in set(TR.split())}.items(),
              key=lambda kv: -kv[1])[:8]
en_k = sorted({w: len(enc.encode(w)) for w in set(EN.split())}.items(),
              key=lambda kv: -kv[1])[:8]
for i in range(8):
    a = f"{tr_k[i][0]:<22}{tr_k[i][1]:>7}" if i < len(tr_k) else ' ' * 29
    b = f"{en_k[i][0]:<22}{en_k[i][1]:>7}" if i < len(en_k) else ''
    print(f"  {a}   {b}")
print()

# tek tek ornek kelimeler
print("  ORNEK KELIMELER, TEK TEK")
ciftler = [("ev", "house"), ("evim", "my house"), ("evimde", "in my house"),
           ("evlerimizden", "from our houses"),
           ("çalıştırdığım", "that I ran"),
           ("karşılaştırabileceğim", "that I will be able to compare"),
           ("Çiğdem", "Margaret"), ("İstanbul'da", "in Istanbul")]
print(f"  {'Turkce':<24}{'tk':>4}  {'parcalar':<42}{'Ingilizce':<22}{'tk':>4}")
print("  " + "-" * 100)
for t, e in ciftler:
    it, ie = enc.encode(t), enc.encode(e)
    pt = ''.join(enc.decode([i]) + '|' for i in it)
    print(f"  {t:<24}{len(it):>4}  {pt:<42}{e:<22}{len(ie):>4}")
print()

print("""  NEDEN BOYLE

  Uc sebep ust uste biniyor.

  1) BYTE MALIYETI (Gorev 1). Turkce'nin isaretli harfleri 2 byte.
     cl100k_base'de bu harflerin cogu tek basina bir token bile degil;
     'ğ' ve 'ş' gibi harfler kendi baslarina token olarak gorunuyor
     cunku Ingilizce agirlikli egitim verisinde Turkce kelimelerin
     icindeki bu harfler yeterince sik gecmemis.

  2) EGITIM VERISI DENGESI. BPE hangi dilde cok veri gorduyse o dilin
     kelimelerini tek token yapiyor. cl100k_base'in verisi agirlikli
     olarak Ingilizce; 'house' tek token, 'ev' bile bazen bolunuyor.
     Bu bir tasarim hatasi degil, verinin dogal sonucu.

  3) EKLEMELI YAPI. Turkce'de bir kelime, Ingilizce'de bes kelimenin
     isini goruyor: 'evlerimizden' = 'from our houses'. Ingilizce'de
     bu bes kelimenin her biri sik gectigi icin tek token olmus.
     Turkce'de ise 'evlerimizden' butun olarak nadir, o yuzden
     heceleniyor.

     ILGINC OLAN: ucuncu sebep ilk ikisini KISMEN dengeliyor. Turkce
     kelime basina daha cok token harciyor ama ayni anlami daha az
     kelimeyle anlatiyor. Yukaridaki tabloda toplam token farkinin,
     kelime basina farktan daha kucuk olmasinin sebebi bu.
""")

# ---------------------------------------------------------------------
# 5b) KENDI TOKENIZER'IMLA
# ---------------------------------------------------------------------
print("=" * 78)
print("5b) AYNI TURKCE PARAGRAF, KENDI TOKENIZER'IMLA")
print("=" * 78)
d = json.load(open('tokenizer_temel.json'))
bt = BasicTokenizer()
bt.merges = {tuple(k): v for k, v in d['merges']}
bt._vocab_kur()
d2 = json.load(open('tokenizer_regex.json'))
rt = RegexTokenizer()
rt.merges = {tuple(k): v for k, v in d2['merges']}
rt._vocab_kur()

benim = olc('benim (temel)', TR, bt.encode)
benim_r = olc('benim (regex)', TR, rt.encode)
harf = dict(ad='harf (Hafta 8)', karakter=len(TR), byte=len(TR.encode('utf-8')),
            kelime=len(TR.split()), token=len(TR),
            token_kelime=len(TR) / len(TR.split()),
            byte_token=len(TR.encode('utf-8')) / len(TR), ids=[])

print(f"  {'tokenizer':<20}{'vocab':>8}{'token':>8}{'token/kelime':>14}"
      f"{'byte/token':>12}")
print("  " + "-" * 64)
for r, v in [(harf, 87), (benim, 1024), (benim_r, 1024), (tr, 100256)]:
    print(f"  {r['ad']:<20}{v:>8,}{r['token']:>8}{r['token_kelime']:>14.3f}"
          f"{r['byte_token']:>12.3f}".replace(',', '.'))
print("  " + "-" * 64)
print()
print("  BENIM TOKENIZER'IM BU CUMLEYI NASIL BOLUYOR (ilk 60 token)")
print("  " + ' | '.join(repr(goster(bt.vocab[i])) for i in benim['ids'][:60]))
print()

print("""  OKUNUSU

  Kendi tokenizer'im, 1024 token'lik vocab'iyla, 100.256 token'lik
  GPT-4 tokenizer'indan Turkce'de DAHA IYI sikistiriyor. Sasirtici
  gibi duruyor ama sebebi basit: benim tokenizer'im sadece Turkce
  gordu. Butun birlestirme butcesini Turkce eklere ve hecelere
  harcadi. GPT-4 ise ayni butceyi yuzlerce dile bolusturmek zorunda.

  Takas tabii ki var: benim tokenizer'im Ingilizce'de kotu olur,
  GPT-4'unki her dilde idare eder. Ozel amacli bir model icin kendi
  tokenizer'ini egitmek mantikli; genel amacli bir model icin degil.

  Hafta 8'e gore kazanc da burada gorunuyor: harf seviyesinde bu
  paragraf {} token'di, simdi {} token. Yani ayni block_size'a
  {:.2f} kat daha fazla metin sigiyor. Hafta 8'de block=64 ile
  yaklasik iki satir goruyordum; ayni pencere simdi bir paragrafin
  yarisini goruyor.
""".format(harf['token'], benim['token'], harf['token'] / benim['token']))

sonuc = {
    'TR': TR, 'EN': EN,
    'gpt4_tr': {k: v for k, v in tr.items() if k != 'ids'},
    'gpt4_en': {k: v for k, v in en.items() if k != 'ids'},
    'benim_temel': {k: v for k, v in benim.items() if k != 'ids'},
    'benim_regex': {k: v for k, v in benim_r.items() if k != 'ids'},
    'harf': {k: v for k, v in harf.items() if k != 'ids'},
    'tr_parcalar': parcalar, 'en_parcalar': parcalar_en,
    'benim_parcalar': [goster(bt.vocab[i]) for i in benim['ids'][:60]],
    'ciftler': [[t, len(enc.encode(t)), e, len(enc.encode(e))]
                for t, e in ciftler],
    'tr_pahali': tr_k, 'en_pahali': en_k,
}
json.dump(sonuc, open('sonuc_gorev5.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev5.json kaydedildi")
