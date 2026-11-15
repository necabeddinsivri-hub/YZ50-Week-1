# Gorev 2 - GPT-2'yi kendim yaz ve agirliklari yukle
# YZ50 Hafta 10  (video 0:13:47 - 0:31:00)
#
# Modelin kendisi gpt2.py'de. Burada agirliklari yukluyorum ve kendi
# modelimin Hugging Face'in modeliyle AYNI logits'i urettigini
# gosteriyorum.

import json

import torch

import agirlik
from gpt2 import GPT, GPTConfig
from tokenizer import enc

torch.manual_seed(1337)
sonuc = {}

print("=" * 80)
print("2a) HAFTA 8'DEKI GPT'DEN FARKLAR")
print("=" * 80)
print("""
  Mimari olarak buyuk bir fark yok: ayni attention, ayni blok, ayni
  residual, ayni pre-norm. Ama agirliklari yukleyebilmem icin kodumun
  OpenAI'inkiyle birebir ayni sekilde dizilmis olmasi gerekiyordu.
  Degistirdiklerim:

  1. OLCEK
     Hafta 8: n_embd=128, 4 head, 4 blok, block=64, V=1024, 1,06M param
     GPT-2  : n_embd=768, 12 head, 12 blok, block=1024, V=50257, 124M
     Yani 117 kat buyuk. Mimari ayni, olcek farkli.

  2. ISIMLER
     Hafta 8: token_embedding_table, position_embedding_table, bloklar,
              son_ln, lm_head
     GPT-2  : transformer.wte, transformer.wpe, transformer.h,
              transformer.ln_f, lm_head
     Bu kozmetik bir degisiklik degil: state_dict anahtarlari bire bir
     tutmazsa agirliklar yuklenmiyor.

  3. Q, K, V TEK KATMANDA
     Hafta 8'de her head'in kendi key/query/value Linear'i vardi
     (dongulu surum). GPT-2'de tek bir Linear var: c_attn, cikti boyutu
     3*n_embd, sonra split ile uce bolunuyor.
     Bunu Hafta 9'da Gorev 5(a)'da zaten yazmistim ve dongulu surumle
     BIREBIR ayni sonucu verdigini gostermistim - fark tam sifir
     cikmisti. Yani bu degisiklik matematigi degistirmiyor, sadece
     hesabi tek matmul'de topluyor.

  4. ReLU YERINE GELU
     Hafta 8: nn.ReLU()
     GPT-2  : nn.GELU(approximate='tanh')
     ReLU negatif girdileri tam sifirliyor; o bolgede gradyan da tam
     sifir oluyor ve noron "oluyor". GELU negatif tarafta kucuk ama
     sifirdan farkli bir deger birakiyor, yani gradyan akmaya devam
     ediyor. 'tanh' yaklasimi ise tarihsel: GPT-2 yazildiginda tam
     formulun erf'i yavasti, Karpathy videoda bunu soyluyor. Bugun
     gerek yok ama agirliklari yukleyecegim icin aynisini kullanmam
     lazim.

  5. lm_head'DE BIAS YOK
     Hafta 8: nn.Linear(n_embd, V)            -> bias var
     GPT-2  : nn.Linear(n_embd, V, bias=False)

  6. WEIGHT TYING
     GPT-2'de giris embedding'i (wte) ile cikis katmani AYNI tensor.
     Hafta 8'de ayriydi. Gorev 5(a)'da kac parametre kazandirdigini
     hesapliyorum.

  7. LayerNorm ISIMLERI ln_1 / ln_2, ve attention maskesi 'bias' adiyla
     bir buffer olarak tutuluyor (benim Hafta 8'deki 'tril' buffer'im).
""")

if not agirlik.var_mi():
    print("  !! veri/model.safetensors bulunamadi, yukleme atlaniyor")
    raise SystemExit(0)

# =====================================================================
# 2b) TRANSPOZE MESELESI
# =====================================================================
print("=" * 80)
print("2b) BAZI AGIRLIKLARI NEDEN TRANSPOZE ETMEM GEREKIYOR")
print("=" * 80)
sd_hf = agirlik.state_dict()
ornek = [('transformer.h.0.attn.c_attn.weight', (768, 2304)),
         ('transformer.h.0.attn.c_proj.weight', (768, 768)),
         ('transformer.h.0.mlp.c_fc.weight', (768, 3072)),
         ('transformer.h.0.mlp.c_proj.weight', (3072, 768)),
         ('transformer.wte.weight', (50257, 768))]
print(f"  {'tensor':<42}{'HF dosyasinda':>16}{'benim Linear':>16}")
print("  " + "-" * 76)
for k, _ in ornek:
    hf = tuple(sd_hf[k].shape)
    if k.endswith('wte.weight'):
        benim, not_ = hf, 'ayni (Embedding)'
    else:
        benim = hf[::-1]
        not_ = 'TRANSPOZE'
    print(f"  {k:<42}{str(hf):>16}{str(benim):>16}   {not_}")
print()
print("""  SEBEBI

  OpenAI'in orijinal GPT-2 kodu TensorFlow'da yazilmisti. Hugging Face
  o kodu PyTorch'a tasirken lineer katmanlar icin nn.Linear yerine
  "Conv1D" adinda kendi sinifini kullandi (ismi yaniltici, evrisim
  degil, duz bir carpim).

  Fark su:
    Conv1D     : agirlik (giris, cikis) seklinde,   y = x @ W
    nn.Linear  : agirlik (cikis, giris) seklinde,   y = x @ W.T

  Yani ayni matris, DEVRIK saklanmis. Dort tur katman bundan
  etkileniyor: c_attn, c_proj (attention), c_fc, c_proj (mlp).

  Transpoze etmezsem iki sey olur. c_fc gibi dikdortgen matrislerde
  boyut tutmaz ve kod hata verir - iyi haber, cunku hatayi hemen
  gorurum. Ama c_proj (768x768) KARE bir matris; transpoze etmeden
  yuklersem boyut tutar, kod calisir ve model sessizce sacmalar. Daha
  kotu olan bu.

  Embedding'ler (wte, wpe), bias'lar ve LayerNorm'lar Conv1D degil, o
  yuzden dogrudan kopyalaniyor.
""")

# =====================================================================
# 2c) YUKLEME
# =====================================================================
print("=" * 80)
print("2c) AGIRLIKLARI KENDI MODELIME YUKLE")
print("=" * 80)
model = GPT.from_pretrained(sd_hf)
model.eval()
p = sum(x.numel() for x in model.parameters())
print(f"  kendi modelim: {p:,} parametre".replace(',', '.'))
print()

# =====================================================================
# 2d) LOGITS KARSILASTIRMASI
# =====================================================================
print("=" * 80)
print("2d) KENDI MODELIM ILE HUGGING FACE AYNI LOGITS'I VERIYOR MU?")
print("=" * 80)
hf, yuklenen = agirlik.hf_model()
print(f"  HF modeline yuklenen tensor: {yuklenen}")
print()

e = enc()
testler = [
    "Hello, I am a language model,",
    "The capital of France is",
    "Merhaba, ben bir dil modeliyim,",
    "def fibonacci(n):\n    if n < 2:",
    "1 2 3 4 5 6 7 8 9",
]
print(f"  {'girdi':<42}{'token':>7}{'en buyuk fark':>16}{'allclose':>11}")
print("  " + "-" * 78)
hepsi = True
farklar = []
for s in testler:
    ids = torch.tensor([e.encode(s)])
    with torch.no_grad():
        benim_l, _ = model(ids)
        hf_l = hf(ids).logits
    fark = (benim_l - hf_l).abs().max().item()
    ok = torch.allclose(benim_l, hf_l, atol=1e-4)
    hepsi &= ok
    farklar.append(fark)
    gor = s.replace('\n', '\\n')
    print(f"  {gor[:40]!r:<42}{ids.shape[1]:>7}{fark:>16.2e}{str(ok):>11}")
print("  " + "-" * 78)
print(f"  hepsi ayni mi (atol=1e-4): {hepsi}")
print()

# daha uzun, rastgele girdi
torch.manual_seed(0)
rnd = torch.randint(0, 50257, (2, 128))
with torch.no_grad():
    b, _ = model(rnd)
    h = hf(rnd).logits
fark_r = (b - h).abs().max().item()
print(f"  rastgele (2, 128) girdi      : en buyuk fark {fark_r:.2e}  "
      f"allclose {torch.allclose(b, h, atol=1e-4)}")
print()
print(f"""  Farklar 1e-4 mertebesinde, yani float32'nin yuvarlama hatasi
  seviyesinde. Sifir cikmamasinin sebebi islem sirasi: HF'in kodu
  attention'i baska bir sirada hesapliyor ve kayan noktali toplama
  birlesme ozelligini saglamiyor (a+b)+c != a+(b+c).

  Hafta 5'te manuel geri yayilimda kurdugum "EXACT mi approx mi"
  olcutu: burada approx, ama 1e-4 seviyesinde. Hafta 9'da tek matmul
  denkligi EXACT cikmisti cunku orada toplama sirasi hic degismiyordu;
  burada degisiyor.
""")

sonuc = dict(param=p, testler=testler, farklar=farklar, hepsi=bool(hepsi),
             rastgele_fark=fark_r, hf_yuklenen=yuklenen)
json.dump(sonuc, open('sonuc_gorev2.json', 'w'), ensure_ascii=False, indent=1)
torch.save(model.state_dict(), 'veri/benim_gpt2.pt')
print("  -> sonuc_gorev2.json kaydedildi")
