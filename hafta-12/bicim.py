# bicim.py - sayi bicimleme
# YZ50 Hafta 11
#
# NEDEN AYRI BIR DOSYA
# Ayni hatayi uc kez yaptim. Turkce'de binlik ayirici nokta oldugu
# icin f"{n:,}" ciktisini .replace(',', '.') ile duzeltiyordum. Sorun
# su: replace BUTUN satira uyguluyor, yani duz yazidaki virgulleri de
# noktaya ceviriyor. Hafta 9'da "(196, 177)" tuple'ini "(196. 177)"
# yapti, Hafta 10'da "(768,)" -> "(768.)" oldu, bu hafta da
# "8.192 token, yani" -> "8.192 token. yani" cikti.
#
# Cozum basit: sayiyi AYRI bicimle, satira oyle koy. Tek bir yerde
# durdugu icin bir daha unutmam.

def bin(n):
    """1115394 -> '1.115.394'"""
    s = f"{int(round(n)):,}"
    return s.replace(',', '.')


def ond(x, basamak=2):
    """3.74 -> '3,74'"""
    return f"{x:.{basamak}f}".replace('.', ',')
