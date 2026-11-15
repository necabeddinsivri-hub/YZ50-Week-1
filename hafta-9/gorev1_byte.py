# Gorev 1 - Metni byte'lara cevir
# YZ50 Hafta 9  (video 0:14:56 - 0:23:50)
#
# Sekiz haftadir modelim metni HARF harf okuyordu. Hafta 8'de Turkce
# metinde 87 farkli karakter saymistim ve her birine bir numara vermistim.
# Bu hafta o numaralandirmanin neden kotu bir fikir oldugunu ve yerine
# ne konuldugunu anliyorum.

import sys
import unicodedata

CUMLE = "Çiğdem'in ağırlığı 72 kg, İzmir'de üşüdü ve şaşırdı."

print("=" * 74)
print("1a) BIR CUMLE, UC FARKLI GORUNUSU")
print("=" * 74)
print(f"  cumle            : {CUMLE}")
print(f"  karakter sayisi  : {len(CUMLE)}")
print(f"  UTF-8 byte sayisi: {len(CUMLE.encode('utf-8'))}")
print(f"  fark             : {len(CUMLE.encode('utf-8')) - len(CUMLE)} byte")
print()
print("  Python'da len(str) KARAKTER sayar, len(bytes) BYTE sayar.")
print("  Turkce'de ikisi ayni degil; Ingilizce'de (ASCII) ayni olurdu.")
print()

print("=" * 74)
print("1b) HER HARFIN KOD NOKTASI VE UTF-8 BYTE'LARI")
print("=" * 74)
print(f"  {'harf':<6}{'kod noktasi':>14}{'byte':>7}{'UTF-8 byte dizisi':>26}"
      f"   {'ad'}")
print("  " + "-" * 92)
tek, cift = [], []
for ch in dict.fromkeys(CUMLE):          # her karakteri bir kez goster
    kod = ord(ch)
    b = ch.encode('utf-8')
    gor = repr(ch) if ch != ' ' else "' '"
    ad = unicodedata.name(ch, '?')
    print(f"  {gor:<6}{'U+%04X' % kod:>14}{len(b):>7}"
          f"{' '.join(f'{x:3d}' for x in b):>26}   {ad[:34]}")
    (tek if len(b) == 1 else cift).append(ch)
print()
print(f"  1 byte tutanlar ({len(tek)}): {''.join(tek)!r}")
print(f"  2 byte tutanlar ({len(cift)}): {''.join(cift)!r}")
print()

print("""  NE OLUYOR

  UTF-8 degisken uzunluklu bir kodlama:
    U+0000 - U+007F   -> 1 byte   (ASCII: a-z, A-Z, 0-9, noktalama)
    U+0080 - U+07FF   -> 2 byte   (c,g,i,o,s,u uzerine isaretli Turkce harfler,
                                   Yunanca, Kiril...)
    U+0800 - U+FFFF   -> 3 byte   (Cince, Japonca, Korece...)
    U+10000 ve ustu   -> 4 byte   (emoji)

  Turkce harfler ikinci araliga dusuyor. Yani ayni anlamdaki bir Turkce
  cumle, Ingilizcesine gore daha fazla byte tutuyor. Gorev 5'te bunun
  token sayisina nasil yansidigini olcecegim.
""")

# emoji ornegi
print("=" * 74)
print("1c) EMOJI VE DAHA UZUN KARAKTERLER")
print("=" * 74)
for ch in ['a', 'ş', 'ド', '🙂', '🇹🇷']:
    b = ch.encode('utf-8')
    kodlar = ' '.join('U+%04X' % ord(c) for c in ch)
    print(f"  {ch!r:<10} {kodlar:<18} {len(b)} byte: "
          f"{' '.join(f'{x:3d}' for x in b)}")
print()
print("  Not: Turk bayragi TEK karakter degil, iki 'bolgesel gosterge'")
print("  harfinin yan yana gelmis hali. Ekranda bir sey gorunuyor ama")
print("  Python icin iki karakter ve 8 byte. 'Karakter' kavraminin")
print("  gorundugu kadar net olmadiginin iyi bir ornegi.")
print()

print("=" * 74)
print("1d) KOD NOKTALARINI DOGRUDAN VOCAB YAPMAK - NEDEN KOTU")
print("=" * 74)
print(f"  Unicode'da tanimli kod noktasi sayisi : 1.114.112")
print(f"  Bunlarin su an atanmis olani          : ~150.000")
print(f"  Hafta 8'de benim Turkce alfabem       : 87")
print(f"  GPT-4'un vocab'i                      : 100.256")
print()
print("""  KENDI CUMLELERIMLE

  Kod noktalarini dogrudan vocab yapmak demek, "her Unicode karakterine
  bir numara verip embedding tablosuna bir satir acmak" demek. Dort
  sorunu var.

  BIRINCI: tablo cok buyuk ve cogu bos duruyor. 150 bin satirlik bir
  embedding tablosu kurardim. Benim Turkce metnimde bunlarin 87'si
  geciyor; geri kalan 149.900 satir hic egitilmeden oylece durur.
  Hafta 8'de 816 bin parametreli bir model egitmistim; sadece embedding
  tablosu 150.000 x 128 = 19 milyon parametre olurdu. Modelin kendisinden
  23 kat buyuk bir tablo, tamami bos.

  IKINCI: kucultursem bu sefer "bilinmeyen karakter" sorunu cikar. Hafta
  8'de alfabemi metinden cikarmistim: 87 karakter. O modele bir emoji ya
  da Cince bir harf versem, stoi sozlugunde bulamaz ve KeyError verir.
  Yani modelim gormedigi bir karakteri okuyamiyor. Byte seviyesinde bu
  sorun tanimi geregi yok: her sey 0-255 arasi byte'lara donuyor, 256
  satirlik bir tablo HER metni temsil edebiliyor.

  UCUNCU: Unicode standardi degisiyor. Her yil yeni karakterler
  ekleniyor (ozellikle emoji). Vocab'i kod noktalarina baglarsam,
  standart guncellendiginde tokenizer'im eskimis olur. Byte'lar
  degismiyor; 0-255 her zaman 0-255.

  DORDUNCU: ayni gorunen metnin birden fazla kod noktasi dizilisi
  olabiliyor. Asagida gosterecegim.

  Peki neden dogrudan byte kullanmiyoruz? Kullanabiliriz, vocab 256 olur.
  Ama o zaman diziler cok uzar: benim cumlem {} karakter ama {} byte.
  Hafta 8'de block_size 64'tu; byte seviyesinde ayni pencereye daha az
  metin sigardi. Attention maliyeti dizi uzunlugunun KARESI ile artiyor,
  yani uzun dizi pahali.

  BPE tam bu ikilemi cozuyor: tabandan (256 byte) basla, ama sik gecen
  byte gruplarini birlestirerek vocab'i kontrollu sekilde buyut. Ne
  150 bin satirlik bos tablo, ne de gereksiz uzun diziler.
""".format(len(CUMLE), len(CUMLE.encode('utf-8'))))

print("=" * 74)
print("1e) AYNI GORUNEN METIN, FARKLI KOD NOKTALARI")
print("=" * 74)
a = "ş"                       # U+015F  tek kod noktasi
b = "ş"                 # s + birlestirici cedilla
print(f"  a = {a!r}   kod noktalari: {[hex(ord(c)) for c in a]}   "
      f"{len(a.encode('utf-8'))} byte")
print(f"  b = {b!r}   kod noktalari: {[hex(ord(c)) for c in b]}   "
      f"{len(b.encode('utf-8'))} byte")
print(f"  ekranda ayni mi gorunuyor? neredeyse. a == b ? {a == b}")
print(f"  NFC normalizasyonundan sonra: "
      f"{unicodedata.normalize('NFC', a) == unicodedata.normalize('NFC', b)}")
print()
print("  Hafta 8'de Turkce korpusu derlerken NFC normalizasyonu yapmistim;")
print("  sebebi tam buydu. Yapmasaydim ayni harf iki farkli token olurdu")
print("  ve alfabem gereksiz yere sisecekti.")
print()
print("  Sinifta Turkce metindeki farkli karakter sayisi 89'dan 353'e")
print("  kadar degismis. Sebep bu: metni nasil temizledigin vocab'i")
print("  belirliyor. Byte seviyesine inince bu belirsizlik ortadan")
print("  kalkiyor - herkesin tabani ayni 256 byte.")
