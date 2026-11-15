# Hafta 12 - sonuclari gormeden once yazdigim tahminler

Yazildigi an: 2026-11-15 02:58 CET (egitim 600 adimlik kosunun basindayken,
HellaSwag ve Turkce kosu hic baslamamisken). Bu dosyayi sonuclardan sonra
degistirmiyorum; karsilastirma rapordaki "Tahminlerim nasil cikti" bolumunde.

1. Son val loss 6,0 ile 7,0 arasinda olacak.
2. Train ile val arasindaki fark 0,1 nat'tan kucuk kalacak, cunku model
   egitim verisinin tek bir devrini bile bitirmiyor; ezberleyecegi tekrar yok.
3. Benim modelimin HellaSwag acc_norm'u rastgelenin %95 araliginda kalacak
   (1.000 ornekte yaklasik %22,3 - %27,7). Yani rastgeleden ayirt edilemeyecek.
4. OpenAI'in GPT-2'si ilk 1.000 ornekte %28 - %31 bulacak (Karpathy'nin tam
   setteki %29,55'inin cevresinde, orneklem gurultusuyla).
5. Turkce modelde val loss bir devirden sonra yukselmeye baslayacak
   (124M parametre, ~470 bin token: ezber bekliyorum).
6. Ayrilmis Turkce metinde, 15 dakika egittigim Turkce modelim OpenAI'in
   GPT-2'sinden DAHA IYI bit/karakter verecek.
7. FineWeb-Edu'da egittigim model Turkce metinde ucunun en kotusu olacak.
