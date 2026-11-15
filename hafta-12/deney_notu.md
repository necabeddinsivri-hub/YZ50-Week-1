# Deney notu - Hafta 12

Kosu sirasinda tuttugum not. Ters giden seyleri de yaziyorum.

## 1. Veriye ulasmak

- Calistigim ortamdan huggingface.co kapali. FineWeb-Edu'yu tarayicidan, satir API'siyle
  100'er belgelik isteklerle cektim. 4.000. belgeden sonra istekler "Failed to fetch" ile
  dusmeye basladi; 15 saniye bekleyip denedim, yine olmadi. Hiz siniri. 4.000 belgeyle devam ettim.
- Karpathy'nin `hellaswag.py` dosyasindaki GitHub adresi 404. Ayni veriyi Hugging Face'te
  (Rowan/hellaswag) buldum. Ayni veri oldugunu ilk ornekten kontrol ettim: iki kaynakta da
  "ind 24, Roof shingle removal".
- 4.000 belge = 4.138.652 token. Dusuk gibi gorunuyor ama bu CPU saatte birkac milyon
  token isleyebiliyor; model veriyi tuketmeden sure bitiyor. Kisitlayan veri degil, hesap.

## 2. Egitimin uc kez baslamasi

- Ilk kosu, calisma oturumu kapaninca 50. adimda durduruldu. Elimdeki `model_son.pt` 8 adimlik
  derleme testinden kalmaydi. Dosyanin tarihine bakmasaydim HellaSwag'i neredeyse egitilmemis bir
  modelde olcup "benim modelim rastgele" diye yazacaktim - sonuc dogru gorunurdu ama sebebi yanlis
  olurdu.
- Ikinci ve ucuncu baslangic hiz yuzunden: test kosusunda adim 5,1 sn, asil kosuda 6,8 sonra 8,2 sn.
  80 dakikalik guvenlik sinirina takilip cosine programinin yarida kesilmemesi icin plani 680 -> 600
  -> 500 adima indirdim.
- Son kosu: 500 adim, 63,3 dakika, 2.048.000 token,
  559 token/sn.

## 3. Hafta 11'in hiz tablosu burada gecerli degil

Ortam yenilenince PyTorch 2.15 kuruldu. Hafta 11'de en iyi ayar 405 token/sn veriyordu; bu hafta
derlemesiz hali bile 585. torch.compile'i yeniden olctum (8 adimlik test, ~800 token/sn) ve hala
kazandirdigini gordum. Hafta 11'de "bir hiz tablosu donanima ait" demistim; bu hafta ekliyorum:
yazilim surumune de ait.

## 4. Tahminleri once yazdim

Hafta 9-11'de sekiz tahminim yanlis cikti ve hepsini ancak onceden yazdigim icin yakaladim. Bu hafta
bunu kurala cevirdim: `tahminler.md` egitimin basinda, sonuc yokken yazildi ve degismedi.
Karsilastirma raporun 7. bolumunde.

## 5. HellaSwag kodunu dogrulamak

Benim modelim rastgele seviyesinde cikarsa bunun sebebi model mi, olcum kodum mu? Ayirt etmek icin
OpenAI'in modelini tam sette (bf16) kostum ve Karpathy'nin raporladigi sayiyla karsilastirdim.
Benim kodum acc_norm %29,68, Karpathy %29,55. Kod dogru; fark modelden.

Bu kosu da bir kez yarida kaldi: ortam askiya alininca 1.100. ornekte durdu. Koda ara kayit ve
kaldigi yerden devam ekledim, fp32 yerine bf16 ile yeniden kostum (ilk 1.000 ornekte fark 0,4 puan).

## 6. Turkce HellaSwag

Turkce HellaSwag'i (malhajar/hellaswag-tr) indirmek istedigimde bilgisayarimla baglanti koptu;
baglanti geri gelince indirip uc modeli ilk 1.000 ornekte kostum (HS_DIL=tr). Sonuc raporun
6.2 bolumunde. Ayni kod, yalniz veri dosyasi degisiyor; Turkce sette Ingilizce'deki ilk ornek
(ind 24) yok, 10.041 ornek var.
