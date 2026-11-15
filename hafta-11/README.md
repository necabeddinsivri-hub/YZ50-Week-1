# YZ50 Hafta 11 - Egitimi hizlandirmak ve GPT-3'un ayarlari

Karpathy, "Let's reproduce GPT-2 (124M)", video 1:02:00 - 2:46:52.

Bu hafta modelin kendisine dokunmadim. Ayni GPT-2'yi (124M) daha hizli
egitilebilir hale getirdim ve GPT-3 makalesinin ek B'sindeki egitim
ayarlarini kurdum.

## Donanim notu

GPU yok: iki cekirdekli bir Intel Xeon. Ama islemcide **AMX** var
(`amx_bf16`, `amx_tile`) - Intel'in matris carpma birimi, GPU'daki tensor
core'un CPU karsiligi. Bu yuzden videodaki bes adimin dordu olculdu; TF32
CUDA'ya ozgu oldugu icin atlandi ve yerine ayni fikrin CPU karsiligi
(fp32 matmul hassasiyet ipucu) olculdu.

## Dosyalar

| Dosya | Ne yapiyor |
|---|---|
| `gpt2.py` | GPT-2 modeli (Hafta 10). Eklenenler: `flash` secenegi, `configure_optimizers` (weight decay yalniz 2 boyutlu parametrelere), `param_sayisi` |
| `veri.py` | Metni B x T'lik dilimlere bolup sirayla veren data loader |
| `gorev1_dataloader.py` | Data loader, weight tying, baslangic degerleri + olceklemenin olcumu |
| `gorev2_hizlandir.py` | Hizlandirma tablosu: her adim tek tek eklenip olculuyor |
| `gorev3_gpt3.py` | AdamW, clipping, warmup+cosine lr, weight decay ayrimi, gradient accumulation + 50 adimlik egitim |
| `gorev4_ek.py` | Ek gorev (b): ayni model, iki tokenizer (GPT-2 ve Hafta 9), esit sure |
| `bpe.py`, `tokenizer_regex.json` | Hafta 9'da yazdigim BPE tokenizer'i (ek gorev icin) |
| `tokenizer.py`, `veri/` | GPT-2 tokenizer'ini cevrimdisi yukler |
| `bicim.py` | Turkce sayi bicimleme |
| `gorseller.py` | Grafikler (sonuc_gorev*.json dosyalarindan) |

## Calistirma

```bash
pip install torch tiktoken matplotlib
python3 gorev1_dataloader.py
python3 gorev2_hizlandir.py
python3 gorev3_gpt3.py
python3 gorev4_ek.py
python3 gorseller.py
```

## Sonuclar

Asagidaki sayilar teslim ettigim teknik nottaki kosulardan. Kosu
ciktilari (`sonuc_gorev*.json`) calisma ortami yenilenince kayboldu;
betikler onlari yeniden uretir. Not: PyTorch surumu degisirse Gorev 2'nin
hiz sayilari degisir (Hafta 12'de bunu bizzat yasadim).

### Gorev 1 - data loader, weight tying, baslangic degerleri
- Weight tying kazanci: **38.597.376** parametre = V x n_embd = 50.257 x 768
- Baslangic std'leri TAZE modelde: residual katmanlar 0,00408 = 0,02 / sqrt(24)
- Olceklemenin olcumu: ayni derinlikte olceksiz / olcekli akis orani
  **5,33x**, teorik sqrt(2 x n_layer) = 4,90. Tahminimi yanlis nicelige
  baglamistim: sqrt(24) derinlik boyunca buyumede degil, iki modelin
  ayni derinlikteki oraninda cikiyor.

### Gorev 2 - hizlandirma tablosu (B=4, T=256, 1.024 token/adim, medyan)

| Ayar | Bu adim |
|---|---|
| fp32 matmul "high" (TF32'nin CPU karsiligi) | 1,01x |
| bfloat16 autocast (AMX) | **1,57x** |
| flash attention (SDPA) | 1,04x |
| vocab 50257 -> 50304 | **0,96x** |
| torch.compile | 1,47x (72 sn derleme, 21 adimda basabas) |

Toplam: **5,92 s -> 2,53 s (2,34x)**, 173 -> 405 token/sn. En cok
kazandiran adim bfloat16. Vocab'i yuvarlamak bu donanimda kazandirmadi:
GPU'ya ozgu bir kazanc.

### Gorev 3 - GPT-3 ayarlari
- AdamW betas (0,9, 0,95), eps 1e-8, weight decay 0,1 yalniz 2 boyutlu parametrelere
- Warmup (10 adim) + cosine, max_lr 6e-4, min_lr 6e-5
- Gradient accumulation kaniti: loss'u N'e bolerek biriktirmek tek
  seferle ayni (fark ~1e-9); bolmeden biriktirmek gradyan normunu tam
  **4,00x** (N=4) buyutuyor
- 50 adimlik egitim (8.192 token/adim): loss 10,99 -> 5,82; ilk adimda
  grad norm 27,4; **50 adimin 30'u kirpildi**

### Gorev 4 (ek, b) - iki tokenizer, esit sure (180 sn)
- Hafta 9'daki 1.024'luk tokenizer'im Turkce'de GPT-2'nin 50.257'lik
  vocab'ini %18 yeniyor (token sayisinda)
- Esit surede benim tokenizer'imla model **10,84 kat** daha cok metin gordu
- Bit/karakter: benim 2,54, GPT-2 3,68 (az iyi). Ama loss/ln(V) tersini
  soyluyor (GPT-2 %47,2, benim %62,0): goreceli olcek ile mutlak olcek
  celisiyor; soruyu cevaplayan bit/karakter.

## Notlar

- Kodu kendim yazdim; kavramlari ogrenmek icin AI'a danistim.
- Teknik not (Word), sunum ve video metni teslim mailinde.
