# Gorev 6 (ek) - (a) GPT-4'un birlestirmelerini yukle, tiktoken ile ayni sonucu ver
# YZ50 Hafta 9  (video 1:11:38 - 1:25:28; minbpe alistirmasinin 3. adimi)
#
# Amac: kendi RegexTokenizer'imi GPT-4'un birlestirme tablosuyla besleyip,
# tiktoken'in urettigi token'larin BIRE BIR AYNISINI uretmek.

import json
import random

from bpe import GPT4Tokenizer, GPT4_DESEN
from gpt4_yukle import encoding, mergeable_ranks

print("=" * 76)
print("6a) GPT-4'UN TOKENIZER'INI KENDI KODUMLA CALISTIRMAK")
print("=" * 76)

ranks = mergeable_ranks()
enc = encoding()
print(f"  yuklenen birlestirilebilir token : {len(ranks):,}".replace(',', '.'))
print()

print("""  IKI PURUZ VARDI

  1) tiktoken bana 'token -> sira numarasi' veriyor, ama hangi iki
     token'in birlesip bu token'i olusturdugunu SOYLEMIYOR. Benim
     encode fonksiyonum ise merges sozlugune, yani ikililere ihtiyac
     duyuyor.

     Cozum: her token'i tek tek byte'larina ayirip, SADECE o token'dan
     ONCE gelmis birlestirmeleri (rank < o token'in rank'i) uyguluyorum.
     Geriye tam iki parca kaliyor; iste o ikili aradigim sey.
     (bpe.py, _bpe_coz ve gpt4_merges)

     Ilk denememde daha basit bir sey yapmistim: 'token'i ilk gecerli
     yerden ikiye bol'. Yanlisti ve KeyError ile coktu. Sebep: bir
     token birden fazla yerden bolunebiliyor ve dogrusu her zaman
     ilki degil. Dogru cevap icin birlestirme SIRASINI kullanmak
     gerekiyor - Gorev 3'te encode yazarken ogrendigim seyin aynisi.

  2) GPT-4 taban byte'larini KARISTIRIYOR. tiktoken'de 0 numarali
     token b'!' , 1 numarali b'"' ... Yani byte degeri 33 olan '!'
     karakteri 0 numarali token. Neden boyle oldugunu bilmiyorum,
     muhtemelen tarihsel bir sebep. Benim kodum 0-255 byte'larini
     dogrudan 0-255 token sayiyordu.

     Cozum: encode'dan once byte'lari bu siraya ceviriyorum
     (byte_shuffle), decode'da geri aliyorum.
""")

tok = GPT4Tokenizer(ranks)
print(f"  kurtarilan birlestirme sayisi: {len(tok.merges):,}".replace(',', '.'))
print()

# byte karisikligini goster
print("  BYTE KARISIKLIGI - ilk birkac ornek")
print(f"  {'byte':>6}{'karakter':>12}{'GPT-4 token no':>18}")
print("  " + "-" * 38)
for b in [33, 34, 97, 65, 32, 10]:
    ch = repr(chr(b)) if chr(b).isprintable() else repr(chr(b))
    print(f"  {b:>6}{ch:>12}{tok.byte_shuffle[b]:>18}")
print()

# ---------------------------------------------------------------------
# DENKLIK TESTI
# ---------------------------------------------------------------------
print("=" * 76)
print("DENKLIK TESTI")
print("=" * 76)
testler = [
    "hello world",
    "Merhaba dünya",
    "Çiğdem'in ağırlığı 72 kg, İzmir'de üşüdü.",
    "Ankara'dan İstanbul'a 2026'da gittim.",
    "   bosluklu    metin\n\nsatir sonlari\t tab",
    "emoji 🙂🇹🇷 ve 日本語 ve ру́сский",
    "1234567890 3.14159 0x1F",
    "def f(x):\n    return x ** 2  # yorum",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    "",
    " ",
    "a",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
]

print(f"  {'metin':<48}{'tiktoken':>10}{'benim':>8}{'sonuc':>9}")
print("  " + "-" * 76)
hepsi = True
for s in testler:
    a = enc.encode(s)
    b = tok.encode(s)
    ok = (a == b)
    hepsi &= ok
    gor = (s[:44] + '...') if len(s) > 44 else s
    print(f"  {gor!r:<48}{len(a):>10}{len(b):>8}"
          f"{'AYNI' if ok else 'FARKLI':>9}")
print("  " + "-" * 76)
print(f"  elle secilen {len(testler)} ornegin hepsi ayni mi: {hepsi}")
print()

# gercek metin uzerinde
print("  GERCEK METIN UZERINDE")
metin = open('turkce.txt', encoding='utf-8').read()
dilim = metin[:40_000]
a = enc.encode(dilim)
b = tok.encode(dilim)
print(f"  turkce.txt ilk {len(dilim):,} karakter".replace(',', '.'))
print(f"    tiktoken : {len(a):,} token".replace(',', '.'))
print(f"    benim    : {len(b):,} token".replace(',', '.'))
print(f"    ayni mi  : {a == b}")
if a != b:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"    ilk fark {i}. token'da: {x} vs {y}")
            break
print()

# rastgele metinler
print("  RASTGELE METINLER")
rnd = random.Random(42)
alfabe = ("abcdefghijklmnopqrstuvwxyzABCZ çğıöşüÇĞİÖŞÜ 0123456789 "
          ".,!?'\"()-\n\t  🙂日本")
bozuk = 0
for _ in range(300):
    n = rnd.randrange(1, 60)
    s = ''.join(rnd.choice(alfabe) for _ in range(n))
    if enc.encode(s) != tok.encode(s):
        bozuk += 1
print(f"  300 rastgele metinden farkli cikan: {bozuk}")
print()

# decode
print("  DECODE TESTI")
d_ok = all(tok.decode(enc.encode(s)) == s for s in testler)
print(f"  decode(tiktoken_encode(metin)) == metin : {d_ok}")
print()

print(f"""  SONUC

  Kendi yazdigim RegexTokenizer, GPT-4'un birlestirme tablosuyla
  beslendiginde tiktoken ile BIRE BIR ayni token'lari uretiyor
  ({'hepsi ayni' if (hepsi and a == b and bozuk == 0) else 'bazi farklar var'}).

  Bu benim icin onemliydi, cunku kodun dogru olup olmadigini kendi
  testimle degil, BAGIMSIZ bir referansla olcmus oldum. Hafta 5'te
  manuel geri yayilimi PyTorch'un gradyanlariyla karsilastirmistim;
  mantik ayni. Kendi kodunu kendi beklentinle test edersen, ayni
  yanlisi iki kere yapmis olabilirsin.

  Not: 'ayni token' demek 'ayni sayilar' demek. Kendi egittigim
  tokenizer (Gorev 2 ve 4) farkli sayilar uretiyor; buradaki denklik
  sadece GPT-4'un tablosunu yukledigim surum icin gecerli.
""")

json.dump({
    'birlestirme': len(tok.merges), 'rank': len(ranks),
    'elle_test': len(testler), 'elle_ayni': bool(hepsi),
    'metin_karakter': len(dilim), 'metin_tiktoken': len(a),
    'metin_benim': len(b), 'metin_ayni': bool(a == b),
    'rastgele': 300, 'rastgele_bozuk': bozuk, 'decode_ok': bool(d_ok),
}, open('sonuc_gorev6a.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev6a.json kaydedildi")
