# turkce_derle.py - kendi Turkce metin korpusumu derliyorum (~1 MB)
#
# Hafta 8, Gorev 4 icin veri hazirligi.
#
# Kaynak: Universal Dependencies Turkce treebank'lari (BOUN, Penn, Kenet, IMST).
# Bunlar serbest lisansli, gercek Turkce metinlerden derlenmis veri kumeleri:
# roman, deneme, gazete yazisi, haber, ansiklopedi maddesi karisimi.
# CoNLL-U dosyalarinin icinde her cumlenin duz hali "# text = ..." satirinda
# duruyor; ben sadece o satirlari cekip birlestiriyorum. Yani dil bilgisi
# etiketlerini atip sadece ham metni aliyorum.
#
# Neden Vikikaynak degil: kodu calistirdigim ortamdan wikimedia alan adlarina
# cikis kapali. Isteneni (kendim derledigim ~1 MB Turkce duz metin) ayni sekilde
# karsilayan, erisebildigim bir kaynak sectim.

import os
import re
import unicodedata

KAYNAKLAR = ['tr_kaynak/boun.conllu', 'tr_kaynak/penn.conllu',
             'tr_kaynak/kenet.conllu', 'tr_kaynak/imst.conllu']
HEDEF_BAYT = 1_050_000          # ~1 MB
CIKTI = 'turkce.txt'

# Shakespeare dosyasindaki gibi sade bir alfabe istiyorum. Turkce harfler +
# temel noktalama. Bunun disinda kalan karakter cumleyi disliyor (alfabeyi
# sismeye birakmamak icin).
IZIN = set(
    "abcdefghijklmnopqrstuvwxyzçğıöşü"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÇĞIİÖŞÜ"
    "0123456789"
    " .,;:!?'\"()-\n"
)


def temizle(s):
    # CoNLL-U icinde tipografik tirnak/tire cesitleri var; duz karsiliklarina
    # ceviriyorum ki alfabe kabarmasin.
    s = unicodedata.normalize('NFC', s)
    for a, b in [('“', '"'), ('”', '"'), ('‘', "'"),
                 ('’', "'"), ('–', '-'), ('—', '-'),
                 ('…', '...'), (' ', ' '), ('`', "'"),
                 ('ʼ', "'"), ('­', '')]:
        s = s.replace(a, b)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def cumleler(yol):
    with open(yol, 'r', encoding='utf-8') as f:
        for satir in f:
            if satir.startswith('# text = '):
                yield temizle(satir[len('# text = '):])


def main():
    gorulen = set()
    secilen = []
    boyut = 0
    sayac = {k: 0 for k in KAYNAKLAR}
    atilan = 0

    for yol in KAYNAKLAR:
        if not os.path.exists(yol):
            print(f"  ! {yol} yok, atliyorum")
            continue
        for c in cumleler(yol):
            if boyut >= HEDEF_BAYT:
                break
            # cok kisa ya da izinli olmayan karakter iceren cumleleri atiyorum
            if len(c) < 25:
                atilan += 1
                continue
            if any(ch not in IZIN for ch in c):
                atilan += 1
                continue
            if c in gorulen:
                atilan += 1
                continue
            gorulen.add(c)
            secilen.append(c)
            boyut += len(c.encode('utf-8')) + 1
            sayac[yol] += 1
        if boyut >= HEDEF_BAYT:
            break

    metin = '\n'.join(secilen) + '\n'
    with open(CIKTI, 'w', encoding='utf-8') as f:
        f.write(metin)

    harfler = sorted(set(metin))
    print("=" * 62)
    print("TURKCE KORPUS DERLENDI")
    print("=" * 62)
    for k, v in sayac.items():
        if v:
            print(f"  {k:<28} {v:>7,} cumle".replace(',', '.'))
    print(f"  {'atilan (kisa/yabanci karakter/tekrar)':<28} {atilan:>7,}".replace(',', '.'))
    print()
    print(f"  cumle sayisi      : {len(secilen):,}".replace(',', '.'))
    print(f"  karakter sayisi   : {len(metin):,}".replace(',', '.'))
    print(f"  dosya boyutu      : {len(metin.encode('utf-8')):,} bayt"
          .replace(',', '.'))
    print(f"  farkli karakter   : {len(harfler)}")
    print(f"  alfabe            : {''.join(harfler)!r}")
    print()
    print("  ilk 300 karakter:")
    print("  " + repr(metin[:300]))


if __name__ == '__main__':
    main()
