# Gorev 4 - GPT-4'un regex desenini ekle
# YZ50 Hafta 9  (video 0:57:36 - 1:11:38)
#
# Gorev 2'deki tokenizer metni tek bir uzun byte dizisi olarak goruyor.
# BPE istedigi yerde birlestirebiliyor - kelime sinirini gecip noktalamayi
# ya da boslugu icine alabiliyor. GPT-2 makalesi bunun sorun oldugunu
# soyluyor ve cozum olarak once regex ile bolmeyi oneriyor.

import json
import time

import regex as re

from bpe import RegexTokenizer, BasicTokenizer, goster, GPT4_DESEN, GPT2_DESEN

VOCAB = 1024
metin = open('turkce.txt', encoding='utf-8').read()
sonuc = {}

# ---------------------------------------------------------------------
# 4a) DESEN NE YAPIYOR
# ---------------------------------------------------------------------
print("=" * 74)
print("4a) GPT-4'UN DESENI")
print("=" * 74)
print(f"  {GPT4_DESEN}")
print()
print("""  Parca parca:
    '(?i:[sdmt]|ll|ve|re)        Ingilizce kisaltmalar: 's 'd 'm 't 'll 've 're
    [^\\r\\n\\p{L}\\p{N}]?+\\p{L}+    (varsa bir harf-disi karakter +) HARF dizisi
    \\p{N}{1,3}                   en fazla 3 basamakli sayi
     ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*   (varsa bosluk +) noktalama dizisi
    \\s*[\\r\\n] | \\s+(?!\\S) | \\s+  satir sonlari ve bosluklar

  Onemli nokta: parcalar birbirine asla KARISMIYOR. BPE her parcanin
  icinde ayri calisiyor, yani bir token asla parca sinirini gecemiyor.
""")

ornek = "Ankara'dan geldim, 2026'da İstanbul'da olacağım. Tamam mı?"
desen4 = re.compile(GPT4_DESEN)
desen2 = re.compile(GPT2_DESEN)
print(f"  Ornek: {ornek!r}")
print(f"  GPT-4 : {desen4.findall(ornek)}")
print(f"  GPT-2 : {desen2.findall(ornek)}")
print()

# ---------------------------------------------------------------------
# 4b) YENIDEN EGIT VE KARSILASTIR
# ---------------------------------------------------------------------
print("=" * 74)
print(f"4b) REGEX'LI TOKENIZER, YENIDEN EGITIM (vocab {VOCAB})")
print("=" * 74)
kayit_r = []
rt = RegexTokenizer()
t0 = time.time()
rt.train(metin, VOCAB, kayit=kayit_r)
sure_r = time.time() - t0
print(f"  {VOCAB-256} birlestirme, {sure_r:.1f} saniye")
print()

# Gorev 2'deki temel tokenizer'i geri yukle
d = json.load(open('tokenizer_temel.json'))
bt = BasicTokenizer()
bt.merges = {tuple(k): v for k, v in d['merges']}
bt._vocab_kur()

print("  ILK 20 BIRLESTIRME - YAN YANA")
print(f"  {'#':<4}{'TEMEL (regex yok)':<26}{'REGEX (GPT-4 deseni)':<26}")
print("  " + "-" * 60)
t_ilk = [goster(bt.vocab[256 + i]) for i in range(20)]
r_ilk = [goster(rt.vocab[256 + i]) for i in range(20)]
for i in range(20):
    isaret = '  ' if t_ilk[i] == r_ilk[i] else ' *'
    print(f"  {i+1:<4}{t_ilk[i]!r:<26}{r_ilk[i]!r:<26}{isaret}")
print("  " + "-" * 60)
ayni = sum(1 for i in range(20) if t_ilk[i] == r_ilk[i])
print(f"  ilk 20'de ayni olan: {ayni}/20   (* = farkli)")
print()

# vocab'in yapisi: token'lar neye benziyor
def olc_vocab(tokenizer, n=VOCAB):
    """sonek: sonu bosluk | noktalamali | onek: basi bosluk."""
    sonek, noktalama, onek = [], [], []
    for i in range(256, n):
        t = tokenizer.vocab[i].decode('utf-8', errors='replace')
        if not any(c.isalpha() for c in t):
            continue
        if t[-1].isspace():
            sonek.append(t)
        if any((not c.isalnum()) and (not c.isspace()) for c in t):
            noktalama.append(t)
        if t[0] == ' ' and not t[-1].isspace():
            onek.append(t)
    return sonek, noktalama, onek


t_sonek, t_nokt, t_onek = olc_vocab(bt)
r_sonek, r_nokt, r_onek = olc_vocab(rt)
print("  VOCAB'IN YAPISI  (768 birlestirme icinde, harf iceren token'lar)")
print(f"  {'':<22}{'sonu bosluk':>13}{'noktalamali':>13}{'basi bosluk':>13}")
print("  " + "-" * 62)
print(f"  {'TEMEL (regex yok)':<22}{len(t_sonek):>13}{len(t_nokt):>13}"
      f"{len(t_onek):>13}")
print(f"  {'REGEX (GPT-4)':<22}{len(r_sonek):>13}{len(r_nokt):>13}"
      f"{len(r_onek):>13}")
print()
print(f"  TEMEL, sonu bosluklu : {t_sonek[:8]}")
print(f"  TEMEL, noktalamali   : {t_nokt[:6]}")
print(f"  REGEX, basi bosluklu : {r_onek[:8]}")
print(f"  REGEX, noktalamali   : {r_nokt[:8]}")
print()
print("""  ILK YORUMUM YANLISTI, DUZELTIYORUM

  Once "regex'li surumde bir token ya harf ya noktalama olur" diye
  dusunmustum. Sayilara bakinca yanlis oldugunu gordum: regex'li
  surumde BASINDA bosluk olan token SAYISI cok daha fazla.

  Desene tekrar baktim:

      [^\\r\\n\\p{L}\\p{N}]?+\\p{L}+

  Basta bir harf-disi karakter OLABILIR, sonra harfler gelir. Yani
  bastaki bosluk bilerek serbest birakilmis. GPT ciktilarinda
  token'larin ' the', ' and' diye gorunmesinin sebebi bu.

  Gercek fark SONDA: TEMEL tokenizer'da sonunda yapisik bosluk olan
  yuzlerce token var ('n ', 'bir '), regex'li surumde SIFIR. Ayrica
  TEMEL'de '.\\nB' gibi cumle sinirini asip sonraki kelimenin ilk
  harfini yutan token'lar olusmus.

  Neden onemli: sonunda bosluk olan token, kelimeyi kendinden SONRA
  geleni de iceren bir birim yapiyor; ayni kelime cumle sonundayken
  baska, ortasindayken baska bolunuyor. Bastaki bosluk ise her zaman
  ayni - kelimenin solunda bosluk vardir. Tutarli olan ikincisi.

  Regex'li surumde kalan 'noktalamali' token'larin neredeyse hepsi
  kesme isaretli Turkce ekler ("'nin", "'ın"). Onlari 4c'de ayrica
  inceliyorum.
""")

# sikistirma karsilastirmasi
dilim = metin[:200_000]
d_byte = len(dilim.encode('utf-8'))
t_tok = len(bt.encode(dilim))
r_tok = len(rt.encode(dilim))
print("  SIKISTIRMA")
print(f"  {'tokenizer':<22}{'token':>10}{'oran':>10}")
print("  " + "-" * 44)
print(f"  {'TEMEL':<22}{t_tok:>10,}{d_byte/t_tok:>9.3f}x".replace(',', '.'))
print(f"  {'REGEX':<22}{r_tok:>10,}{d_byte/r_tok:>9.3f}x".replace(',', '.'))
print()
print(f"""  BEKLENTIM YINE TUTMADI.

  "Regex koyunca sikistirma kotulesir" diye dusunmustum, cunku bazi
  faydali birlestirmeleri (sonu bosluklu token'lar) yasakliyorum.
  Olcunce tersi cikti: regex'li surum biraz DAHA IYI sikistirdi
  ({t_tok:,} -> {r_tok:,} token).

  Sebebini soyle anliyorum: regex'siz surumde ayni kelime, yanindaki
  noktalama ve bosluga gore farkli farkli bolunuyordu. Yani ayni
  kelimenin birkac ayri parcalanisi vocab butcesini tuketiyordu.
  Regex bu dagilimi engelleyince butce tek bir dogru parcalanisa
  yogunlasti ve sonuc daha verimli cikti.

  Fark kucuk (yuzde {100*(t_tok-r_tok)/t_tok:.2f}), yani asil kazanc
  sikistirmada degil TUTARLILIKTA. Ama "kotu olacak" dedigim yerde
  "biraz iyi" cikmasi, tahminde bulunmak yerine olcmenin neden
  gerektiginin bir ornegi daha.
""".replace(',', '.'))

# ---------------------------------------------------------------------
# 4c) KESME ISARETLI TURKCE KELIMELER
# ---------------------------------------------------------------------
print("=" * 74)
print("4c) DESEN TURKCE'DE NE YAPIYOR?")
print("=" * 74)
kelimeler = ["Ankara'dan", "İstanbul'da", "Türkiye'nin", "2026'da",
             "Ahmet'le", "YZ50'nin", "ABD'ye", "Çiğdem'in"]
print(f"  {'kelime':<16}{'GPT-4 deseni':<42}{'parca'}")
print("  " + "-" * 72)
for k in kelimeler:
    p = desen4.findall(k)
    print(f"  {k:<16}{str(p):<42}{len(p)}")
print()
print("  Karsilastirma - Ingilizce:")
for k in ["Ankara's", "don't", "I'll", "we've"]:
    p = desen4.findall(k)
    print(f"  {k:<16}{str(p):<42}{len(p)}")
print()

# ---- 26 Turkce ek uzerinde sistematik test ----
EKLER = ["'da", "'de", "'ta", "'te", "'dan", "'den", "'tan", "'ten",
         "'nin", "'nın", "'in", "'ın", "'ye", "'ya", "'e", "'a",
         "'le", "'la", "'ler", "'lar", "'si", "'sı", "'nde", "'ndan",
         "'mi", "'mı"]
bozuk, temiz = [], []
for ek in EKLER:
    p = desen4.findall("Ankara" + ek)
    (bozuk if len(p) > 2 else temiz).append((ek, p))

print("=" * 74)
print("  26 TURKCE EK UZERINDE SISTEMATIK TEST")
print("=" * 74)
print(f"  BOZULAN ({len(bozuk)}/{len(EKLER)}) - ek ikiye bolunuyor:")
for ek, p in bozuk:
    print(f"    Ankara{ek:<7} -> {p}")
print()
print(f"  TEMIZ ({len(temiz)}/{len(EKLER)}) - ek tek parca kaliyor:")
for ek, p in temiz:
    print(f"    Ankara{ek:<7} -> {p}")
print()

print("""  KENDI CUMLELERIMLE

  Desen Ingilizce icin yazilmis ve Turkce'de ne yaptigi bu tabloda
  cok net gorunuyor. Beklemedigim bir sonuc cikti.

  ONCE BEKLENTIM: "kesme isareti kendi basina bir parca olur, ek
  kokten kopar." YANLIS cikti.

  GERCEKTE OLAN: desenin IKI kurali kesme isaretli kelimelerde
  yarisiyor ve hangisinin kazandigi EKIN ILK HARFINE bagli.

  Kural 1 (ilk siradaki):  '(?i:[sdmt]|ll|ve|re)
      Ingilizce kisaltmalari yakaliyor: 's 'd 'm 't 'll 've 're
  Kural 2:  [^\\r\\n\\p{L}\\p{N}]?+\\p{L}+
      (bir harf-disi karakter +) harf dizisi

  Regex alternatifleri SIRAYLA deniyor. Ek d, t, s veya m ile
  basliyorsa Kural 1 once eslesiyor ve sadece "'d" kadarini aliyor;
  geri kalan ("an", "a", "en") ayri bir parcaya dusuyor. Ek baska bir
  harfle basliyorsa Kural 1 eslesmiyor, Kural 2 devreye giriyor ve
  kesme isareti + ekin tamami tek parca kaliyor.

  Sonuc: 26 ekten 12'si bozuluyor ve bozulanlar tesaduf degil -
  hepsi d/t/s/m ile baslayanlar. Bunlar Turkce'nin EN SIK kullanilan
  ekleri: bulunma hali (-da/-de/-ta/-te) ve ayrilma hali
  (-dan/-den/-tan/-ten). Yani desen, Turkce'de en cok ihtiyac
  duyulan eklerde en kotu davraniyor.

  "Ankara'dan" -> ['Ankara', "'d", 'an']
  Ortadaki "'d" parcasi Turkce'de hicbir anlam tasimiyor; Ingilizce'de
  "would/had" kisaltmasi. Model bu parcayi gordugunde Ingilizce bir
  kalip bekliyor olabilir.

  BIR SEY DAHA - SAYILAR: kural \\p{N}{1,3} sayilari en fazla 3
  basamakta kesiyor. "2026'da" -> ['202', '6', "'d", 'a'] gibi dort
  parcaya ayriliyor. Bu Ingilizce'de de ayni; sinifta konusulan
  "LLM'ler neden sayilarda hata yapiyor" sorusunun kaynaklarindan
  biri tam olarak bu.

  BUNU NASIL DUZELTIRDIM: desenin ilk kuralini Turkce ekleriyle
  genisletirdim, ornegin
      '(?i:[sdmt]|ll|ve|re|d[ae]n?|t[ae]n?|n[iıuü]n|l[ae]r)
  ya da daha basiti, o kurali tamamen kaldirirdim - cunku Kural 2
  zaten kesme isareti + eki dogru topluyor. Denemedim, cunku bu
  haftanin gorevi GPT-4'un desenini anlamak; ama olcup gormek
  isterdim.
""")

sonuc.update({
    'temel_ilk20': t_ilk, 'regex_ilk20': r_ilk, 'ilk20_ayni': ayni,
    'temel_sonek': len(t_sonek), 'regex_sonek': len(r_sonek),
    'temel_nokt': len(t_nokt), 'regex_nokt': len(r_nokt),
    'temel_onek': len(t_onek), 'regex_onek': len(r_onek),
    'temel_sonek_ornek': t_sonek[:8], 'temel_nokt_ornek': t_nokt[:6],
    'regex_onek_ornek': r_onek[:8], 'regex_nokt_ornek': r_nokt[:8],
    'dilim_byte': d_byte, 'temel_token': t_tok, 'regex_token': r_tok,
    'temel_oran': d_byte / t_tok, 'regex_oran': d_byte / r_tok,
    'egitim_sn': sure_r,
    'kesme': {k: desen4.findall(k) for k in kelimeler},
    'ek_bozuk': [[e, p] for e, p in bozuk],
    'ek_temiz': [[e, p] for e, p in temiz],
    'ek_toplam': len(EKLER),
})
json.dump(sonuc, open('sonuc_gorev4.json', 'w'), ensure_ascii=False, indent=1)
json.dump({'merges': [[list(k), v] for k, v in rt.merges.items()]},
          open('tokenizer_regex.json', 'w'))
print("  -> sonuc_gorev4.json ve tokenizer_regex.json kaydedildi")
