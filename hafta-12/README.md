# YZ50 Hafta 12 - Kendi GPT-2'mi egitmek ve HellaSwag'de olcmek

Karpathy, "Let's reproduce GPT-2 (124M)", video 3:10:21 - sonu. Serinin son haftasi.

Hafta 10-11'de yeniden kurdugum GPT-2'yi (124M) FineWeb-Edu'da sifirdan egittim ve
OpenAI'in GPT-2 agirliklariyla ayni HellaSwag orneklerinde olctum.

## Ozet

| Olcum | Deger |
|---|---|
| gorulen token | 2.048.000 |
| token / saniye | 559 (CPU, derleme adimi haric) |
| son train loss | 6,8907 (son 10 adim ort.) |
| son val loss | 6,7985 |
| HellaSwag acc_norm, benim (ilk 1000) | %23,5 |
| HellaSwag acc_norm, OpenAI GPT-2 (ilk 1000) | %34,0 |
| OpenAI GPT-2, tam set 10.042 (kod dogrulamasi) | %29,68 (Karpathy %29,55) |
| rastgele tahmin | %25,0 |

## Dosyalar

| Dosya | Ne yapiyor |
|---|---|
| `gpt2.py` | GPT-2 modeli (Hafta 10-11), flash attention ve configure_optimizers ile |
| `tokenizer.py` | GPT-2 tokenizer'ini cevrimdisi yukler (veri/gpt2.tiktoken) |
| `veri.py` | ShardYukleyici: dosyalara bolunmus token'lari sirayla okuyan data loader |
| `agirlik.py` | OpenAI'in GPT-2 agirliklarini yerel model.safetensors'tan kendi sinifima yukler |
| `bicim.py` | Turkce sayi bicimleme (Hafta 11'den) |
| `gorev1_veri.py` | FineWeb-Edu -> GPT-2 token'lari -> uint16 .npy dosyalari, ilki val |
| `gorev23_egit.py` | Egitim: val loss ve ornek metin belli araliklarla, her adim JSON'a |
| `hellaswag.py` | HellaSwag: ornegi (4, L) matrise cevirir, devam loss'una gore secer |
| `gorev4_hellaswag.py` | HellaSwag'i bir modelde kosar (benim | openai), ornek ornek kaydeder |
| `gorev4_karsilastir.py` | Iki modeli karsilastirir: istatistik, ayrisma, uzunluk, ornek, referans |
| `gorev5_turkce.py` | Ek gorev: Turkce egitim + 3 model x 2 dil bit/karakter tablosu |
| `gorev5b_hellaswag_tr.py` | Ek gorev: Turkce HellaSwag sonuclarini yan yana koyar |
| `gorseller.py` | Butun grafikler, yalniz sonuc_*.json'dan |
| `tahminler.md` | Sonuclari gormeden ONCE yazdigim tahminler (zaman damgali) |

## Calistirma

```bash
pip install torch tiktoken numpy matplotlib pyarrow safetensors
python3 gorev1_veri.py                  # veri/ham/fineweb_edu_4k.jsonl gerekli
MAX_ADIM=500 python3 gorev23_egit.py    # ~75 dk (CPU)
python3 gorev4_hellaswag.py benim 1000
python3 gorev4_hellaswag.py openai 1000  # veri/agirlik/model.safetensors gerekli
python3 gorev4_hellaswag.py openai 10042        # istege bagli: kod dogrulamasi (~25 dk)
python3 gorev4_karsilastir.py
python3 gorev5_turkce.py
for m in turkce openai benim; do HS_DIL=tr python3 gorev4_hellaswag.py $m 1000; done
python3 gorev5b_hellaswag_tr.py
python3 gorseller.py
```

## Depoda olmayan buyuk dosyalar

- `veri/ham/fineweb_edu_4k.jsonl` (19 MB): FineWeb-Edu sample-10BT'nin ilk 4.000 belgesi, Hugging Face satir API'sinden (`datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT`).
- `veri/ham/hellaswag_validation.parquet` (6 MB): Hugging Face `Rowan/hellaswag`, validation. Karpathy'nin kodundaki GitHub adresi artik 404 veriyor.
- `veri/ham/hellaswag_tr_validation.parquet` (7 MB): Hugging Face `malhajar/hellaswag-tr`, validation (Turkce ceviri).
- `veri/agirlik/model.safetensors` (548 MB): Hugging Face `openai-community/gpt2`. MD5 `8b9634194d1029deff7964617b8bcc75`.
- `veri/shardlar/`, `veri/checkpoint/`: betikler yeniden uretir (kontrol noktalari 548 MB).

## Notlar

- Butun sayilar `sonuc_*.json` dosyalarindan geliyor; raporda ve sunumda elle yazilmis sayi yok.
- GPU yok: egitim iki cekirdekli bir CPU'da (bfloat16 AMX ile). Hafta 11'in hiz tablosu bu ortamda
  yeniden olculdu; PyTorch surumu degistigi icin sayilar farkli.
- Kodu kendim yazdim; kavramlari ogrenmek icin AI'a danistim.
