# Gorev 3 - encode ve decode
# YZ50 Hafta 9  (video 0:42:47 - 0:57:36)
#
# encode: metin -> token id'ler
# decode: token id'ler -> metin
# Ikisi birbirinin tersi olmali: decode(encode(x)) == x

import json

from bpe import BasicTokenizer, goster

# Gorev 2'de egittigim tokenizer'i geri yukluyorum
d = json.load(open('tokenizer_temel.json'))
tok = BasicTokenizer()
tok.merges = {tuple(k): v for k, v in d['merges']}
tok._vocab_kur()
print(f"tokenizer yuklendi: {len(tok.vocab)} token\n")

sonuc = {}

# ---------------------------------------------------------------------
# 3a) GIDIS-DONUS TESTI
# ---------------------------------------------------------------------
print("=" * 74)
print("3a) decode(encode(metin)) == metin  MI?")
print("=" * 74)
ornekler = [
    "Çiğdem'in ağırlığı 72 kg.",
    "İzmir'de üşüdüm, şaşırdım.",
    "Merhaba dünya! 🙂🇹🇷 Türkçe + emoji",
    "Tokenizer yazıyorum 🤖, sıkıştırma oranı 2.5x oldu.",
    "aynı gün ağaç gölgesinde oturduk",
    "",                                   # bos metin
    " ",                                  # tek bosluk
    "🙂",                                 # sadece emoji
    "ÇĞIİÖŞÜçğıiöşü",                     # butun Turkce harfler
]
print(f"  {'metin':<46}{'token':>7}{'byte':>7}  gidis-donus")
print("  " + "-" * 74)
hepsi_tamam = True
for s in ornekler:
    ids = tok.encode(s)
    geri = tok.decode(ids)
    ok = (geri == s)
    hepsi_tamam &= ok
    print(f"  {s!r:<46}{len(ids):>7}{len(s.encode('utf-8')):>7}  "
          f"{'TAMAM' if ok else 'HATA'}")
print()
print(f"  Hepsi tamam mi: {hepsi_tamam}")
sonuc['gidis_donus'] = hepsi_tamam
print()
print("  Emoji dikkat cekici: tokenizer'i SADECE Turkce metinde egittim,")
print("  egitim verisinde hic emoji yoktu. Yine de dogru calisiyor.")
print("  Sebebi: taban vocab 256 byte ve her UTF-8 metin byte'lara")
print("  cevrilebiliyor. Emoji sadece 4 ayri token olarak gidiyor,")
print("  ama kaybolmuyor. Hafta 8'deki karakter tokenizer'ima emoji")
print("  versem KeyError alirdim - cunku alfabem 87 karakterdi ve")
print("  emoji o listede yoktu.")
print()

# emojinin nasil parcalandigini goster
print("  '🙂' nasil token'laniyor:")
ids = tok.encode("🙂")
print(f"    token id'ler : {ids}")
print(f"    her token    : {[tok.vocab[i] for i in ids]}")
print(f"    yani 4 ayri byte, hicbiri tek basina gecerli UTF-8 degil")
print()

# ---------------------------------------------------------------------
# 3b) TERSI DE DOGRU MU?  encode(decode(ids)) == ids
# ---------------------------------------------------------------------
print("=" * 74)
print("3b) TERS YON: encode(decode(ids)) == ids  MI?")
print("=" * 74)
import random
rnd = random.Random(0)
bozuk = 0
for _ in range(200):
    ids = [rnd.randrange(len(tok.vocab)) for _ in range(8)]
    try:
        if tok.encode(tok.decode(ids)) != ids:
            bozuk += 1
    except Exception:
        bozuk += 1
print(f"  200 rastgele token dizisinden {bozuk} tanesi geri donmedi.")
print()
print("""  Bu YON GARANTILI DEGIL ve olmasi da gerekmiyor.

  Sebep: her token dizisi 'dogal' bir metne karsilik gelmiyor. Rastgele
  diziler gecersiz UTF-8 uretebiliyor; decode bunlari onarip (asagida
  anlatiyorum) bir metin donduruyor, ama o metni yeniden encode edince
  BPE kendi kurallarina gore BASKA bir token dizisi uretiyor.

  Onemli olan yon decode(encode(metin)) == metin. Model egitirken ve
  metin uretirken kullandigim yon bu.
""")
sonuc['ters_bozuk'] = bozuk

# ---------------------------------------------------------------------
# 3c) GECERSIZ UTF-8
# ---------------------------------------------------------------------
print("=" * 74)
print("3c) GECERSIZ UTF-8: decode NE YAPMALI?")
print("=" * 74)
print("""  Problem: UTF-8'de her byte dizisi gecerli degil. Kurallar var:
    0xxxxxxx                     -> tek baslik byte'i (ASCII)
    110xxxxx 10xxxxxx            -> iki byte'lik karakter
    1110xxxx 10xxxxxx 10xxxxxx   -> uc byte'lik
  10xxxxxx ile baslayan bir byte DEVAM byte'idir; tek basina duramaz.
""")

for ham in [b'\x80', b'\xc3', b'\xc3\x28', b'\xc3\xa7']:
    print(f"  {ham!r}")
    try:
        print(f"    strict  : {ham.decode('utf-8')!r}")
    except UnicodeDecodeError as e:
        print(f"    strict  : CRASH - {e.reason}")
    print(f"    replace : {ham.decode('utf-8', errors='replace')!r}")
    print(f"    ignore  : {ham.decode('utf-8', errors='ignore')!r}")
print()

# gercek senaryo: tek bir token id decode edilirse
print("  GERCEK SENARYO")
ids = tok.encode("şeker")
print(f"    'şeker' -> {ids}")
for i in ids[:3]:
    b = tok.vocab[i]
    try:
        d1 = b.decode('utf-8')
        not_ = 'gecerli'
    except UnicodeDecodeError:
        d1 = b.decode('utf-8', errors='replace')
        not_ = 'GECERSIZ -> replace devreye girdi'
    print(f"    token {i:>4} = {b!r:<12} -> {d1!r:<8} ({not_})")
print()

print("""  BENIM SECIMIM: errors='replace'

  Neden crash degil: model egitilirken ne urettiginin garantisi yok.
  Uretim sirasinda model, ortasinda kesilmis bir cok-byte'li karakter
  uretebilir - mesela 'ş' harfinin ilk byte'ini uretip sonra baska bir
  seye gecebilir. Burada program cokerse metin uretimi durur. Oysa
  olan sey sadece bir karakterin yarim kalmasi.

  Neden ignore degil: ignore sessizce siliyor. Bir sey ters gittiginde
  haberim olmuyor. replace ise yerine U+FFFD (�) koyuyor; ciktida
  goruyorum ve ne kadar sik oldugunu olcebiliyorum.

  Pratikte ne zaman oluyor: tek tek token decode edince surekli. Model
  cikisinin tamamini birlikte decode edersem cok nadir, cunku
  birlestirmeler genelde tam karakterleri kapsiyor.

  Hafta 7'de uretim kodunda idx[:, -block:] kirpmasini atlarsam
  programin cokecegini gormustum. Buradaki mantik ayni: kenar
  durumlarini kodun kendisi karsilamali, kullanici degil.
""")

# ---------------------------------------------------------------------
# 3d) SIRA NEDEN ONEMLI
# ---------------------------------------------------------------------
print("=" * 74)
print("3d) ENCODE'DA BIRLESTIRME SIRASI NEDEN ONEMLI")
print("=" * 74)
print("""  encode icinde su satir var:

      ikili = min(sayac, key=lambda p: self.merges.get(p, inf))

  Yani "en sik gecen" degil, "EN ERKEN ogrenilmis" birlestirmeyi
  seciyorum. Sebep: egitimde birlestirmeler bir sirayla olustu ve her
  birlestirme bir oncekinin sonucu uzerine bindi. 5. birlestirmeyi
  3.'den once uygularsam, 3.'nun uretecegi token hic olusmaz ve farkli
  bir sonuc cikar.

  Bu, egitim ile cikarim arasindaki tutarlilik meselesi - Hafta 6'da
  BatchNorm'da running_mean ile ugrasirken ogrendigim seyin ayni
  ailesinden. Iki asama birbirinin aynisini yapmak zorunda.
""")
ornek = "ağaçlar"
print(f"  Ornek: {ornek!r}")
ids = list(ornek.encode('utf-8'))
print(f"    baslangic: {ids}")
adim = 0
while len(ids) >= 2 and adim < 6:
    from bpe import istatistik, birlestir
    sayac = istatistik(ids)
    ikili = min(sayac, key=lambda p: tok.merges.get(p, float('inf')))
    if ikili not in tok.merges:
        break
    yeni = tok.merges[ikili]
    ids = birlestir(ids, ikili, yeni)
    adim += 1
    print(f"    adim {adim}: {ikili} -> {yeni} "
          f"({goster(tok.vocab[yeni])!r}); kalan {ids}")
print(f"    sonuc: {[goster(tok.vocab[i]) for i in tok.encode(ornek)]}")
print()

json.dump(sonuc, open('sonuc_gorev3.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev3.json kaydedildi")
