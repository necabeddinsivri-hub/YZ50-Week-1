# Gorev 3 - Modeli konustur
# YZ50 Hafta 10  (video 0:31:00 - 0:45:50)
#
# Sampling dongusu Hafta 7'de yazdigimin aynisi, uzerine top-k ekliyorum.

import json

import torch
from torch.nn import functional as F

import agirlik
from gpt2 import GPT, GPTConfig
from tokenizer import enc

e = enc()
sonuc = {}


@torch.no_grad()
def uret(model, baslangic, kac_dizi=5, uzunluk=30, top_k=50, tohum=42):
    """Her adimda bir sonraki token'i secer.

    top_k: sadece en yuksek olasilikli k token arasindan sec. Geri
    kalanin olasiligini sifirlar. Neden: uzun kuyrukta binlerce cok
    dusuk olasilikli token var; hepsini toplayinca kucumsenmeyecek bir
    olasilik ediyorlar ve ara sira biri secilip metni bozuyor. top_k
    o kuyrugu kesiyor.
    """
    model.eval()
    ids = e.encode(baslangic)
    x = torch.tensor(ids).unsqueeze(0).repeat(kac_dizi, 1)
    g = torch.Generator().manual_seed(tohum)
    while x.size(1) < len(ids) + uzunluk:
        logits, _ = model(x[:, -model.config.block_size:])
        logits = logits[:, -1, :]                     # son adimin skorlari
        probs = F.softmax(logits, dim=-1)
        tk_probs, tk_idx = torch.topk(probs, top_k, dim=-1)
        sec = torch.multinomial(tk_probs, 1, generator=g)
        x = torch.cat((x, torch.gather(tk_idx, -1, sec)), dim=1)
    return [e.decode(x[i].tolist()) for i in range(kac_dizi)]


if not agirlik.var_mi():
    print("  !! veri/model.safetensors bulunamadi")
    raise SystemExit(0)

torch.manual_seed(1337)
model = GPT.from_pretrained(agirlik.state_dict(), sessiz=True)

# =====================================================================
# 3a) INGILIZCE
# =====================================================================
print("=" * 80)
print("3a) INGILIZCE BASLANGIC")
print("=" * 80)
ing = "Hello, I am a language model,"
print(f"  baslangic: {ing!r}  ({len(e.encode(ing))} token)")
print(f"  top_k=50, 5 ayri dizi, 30'ar token")
print()
ing_ciktilar = uret(model, ing)
for i, s in enumerate(ing_ciktilar):
    print(f"  > {s}")
print()

# =====================================================================
# 3b) TURKCE
# =====================================================================
print("=" * 80)
print("3b) TURKCE BASLANGIC")
print("=" * 80)
tur = "Merhaba, ben bir dil modeliyim,"
print(f"  baslangic: {tur!r}  ({len(e.encode(tur))} token)")
print()
tur_ciktilar = uret(model, tur)
for i, s in enumerate(tur_ciktilar):
    print(f"  > {s}")
print()

# =====================================================================
# 3c) NEDEN TURKCE KOTU
# =====================================================================
print("=" * 80)
print("3c) GPT-2'NIN TURKCE CIKTISI NEDEN KOTU")
print("=" * 80)

ciftler = [
    ("Merhaba, ben bir dil modeliyim,", "Hello, I am a language model,"),
    ("Türkiye'nin başkenti Ankara'dır.", "The capital of Turkey is Ankara."),
    ("Bugün hava çok güzel ve dışarı çıkmak istiyorum.",
     "The weather is very nice today and I want to go outside."),
    ("Çiğdem'in ağırlığı yetmiş iki kilogram.",
     "Cigdem's weight is seventy two kilograms."),
]
print(f"  {'Turkce':<50}{'tk':>4}   {'Ingilizce':<52}{'tk':>4}")
print("  " + "-" * 112)
tr_t = en_t = 0
for t, i in ciftler:
    a, b = len(e.encode(t)), len(e.encode(i))
    tr_t += a
    en_t += b
    print(f"  {t:<50}{a:>4}   {i:<52}{b:>4}")
print("  " + "-" * 112)
print(f"  {'TOPLAM':<50}{tr_t:>4}   {'':<52}{en_t:>4}")
print(f"  Turkce, ayni icerik icin {tr_t/en_t:.2f} kat token harciyor.")
print()

print("  'Merhaba, ben bir dil modeliyim,' nasil bolunuyor:")
ids = e.encode(tur)
print(f"    {[e.decode([i]) for i in ids]}")
print()
print("  'Hello, I am a language model,' nasil bolunuyor:")
ids2 = e.encode(ing)
print(f"    {[e.decode([i]) for i in ids2]}")
print()

# tek tek kelimeler
print("  Kelime kelime:")
print(f"  {'Turkce':<26}{'token':>6}  {'parcalar':<40}")
print("  " + "-" * 76)
for k in ["Merhaba", "modeliyim", "başkenti", "Çiğdem", "ağırlığı",
          "istiyorum", "Türkiye'nin"]:
    ii = e.encode(k)
    print(f"  {k:<26}{len(ii):>6}  {str([e.decode([j]) for j in ii]):<40}")
print()
print(f"  {'Ingilizce':<26}{'token':>6}  {'parcalar':<40}")
print("  " + "-" * 76)
for k in ["Hello", "model", "capital", "weight", "want", "today"]:
    ii = e.encode(k)
    print(f"  {k:<26}{len(ii):>6}  {str([e.decode([j]) for j in ii]):<40}")
print()

print(f"""  KENDI CUMLELERIMLE - GECEN HAFTAYLA BAGLANTI

  Gecen hafta GPT-4'un tokenizer'inda (cl100k_base) ayni paragrafin
  Turkcesinin Ingilizcesine gore kelime basina 2,85 kat token
  tuttugunu olcmustum. GPT-2'nin tokenizer'i cl100k'dan daha KUCUK
  (50.257'ye karsi 100.256) ve daha ESKI, yani Turkce'ye ayirdigi pay
  daha da az. Yukaridaki tabloda oran {tr_t/en_t:.2f} cikti.

  Ama asil sorun token sayisi degil. Uc sebep var ve ucuncusu belirleyici.

  1) TOKENIZER. 'Merhaba' tek bir kelime ama GPT-2 onu uce boluyor:
     'Mer', 'hab', 'a'. Model icin bu bir kelime degil, uc ayri parca.
     Ingilizce'de 'Hello' tek token. Yani model Turkce'de daha bastan
     daha parcali bir girdiyle calisiyor.

  2) BAGLAM PENCERESI. Ayni block_size'a daha az Turkce metin siginca
     model daha az sey goruyor. Gecen hafta bunu olcmustum.

  3) EGITIM VERISI - asil sebep bu. GPT-2 WebText uzerinde egitildi:
     Reddit'te en az uc oy almis baglantilardan toplanmis ~40 GB
     Ingilizce agirlikli metin. Makalede Ingilizce disi veriyi
     temizlediklerini soyluyorlar. Yani modelin Turkce'yi bilmemesinin
     sebebi tokenizer degil, Turkce'yi neredeyse hic GORMEMIS olmasi.

  Tokenizer bir semptom, veri ise sebep. Yukaridaki ciktilarda bu
  gorunuyor: model Turkce basladigi cumleyi Ingilizce'ye cevirmeye
  calisiyor ya da anlamsiz karakter obekleri uretiyor.

  Not: bu bir eksiklik degil, bir tasarim sonucu. GPT-2 2019'da
  "Ingilizce dil modeli" olarak yayinlandi.
""")

sonuc = dict(ingilizce=ing, turkce=tur,
             ing_token=len(ids2), tur_token=len(ids),
             ing_ciktilar=ing_ciktilar, tur_ciktilar=tur_ciktilar,
             ciftler=[[t, len(e.encode(t)), i, len(e.encode(i))]
                      for t, i in ciftler],
             tr_toplam=tr_t, en_toplam=en_t, oran=tr_t / en_t,
             tur_parcalar=[e.decode([i]) for i in ids],
             ing_parcalar=[e.decode([i]) for i in ids2])
json.dump(sonuc, open('sonuc_gorev3.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev3.json kaydedildi")
