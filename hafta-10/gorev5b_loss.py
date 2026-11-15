# Gorev 5 (ek) - (b) Turkce ve Ingilizce cumlelerin loss'u
# YZ50 Hafta 10
#
# Egitilmis GPT-2'ye ayni icerigin iki dildeki halini verip loss
# olcuyorum. Gorev 3'te ciktinin kotu oldugunu GORDUM; burada
# OLCUYORUM.

import json
import math

import torch
from torch.nn import functional as F

import agirlik
from gpt2 import GPT, GPTConfig
from tokenizer import enc

if not agirlik.var_mi():
    print("  !! veri/model.safetensors bulunamadi")
    raise SystemExit(0)

e = enc()
torch.manual_seed(1337)
model = GPT.from_pretrained(agirlik.state_dict(), sessiz=True)
model.eval()
V = GPTConfig().vocab_size
TABAN = math.log(V)

CIFTLER = [
    ("Merhaba, ben bir dil modeliyim ve metin uretmeye calisiyorum.",
     "Hello, I am a language model and I am trying to generate text."),
    ("Turkiye'nin baskenti Ankara'dir ve en kalabalik sehri Istanbul'dur.",
     "The capital of Turkey is Ankara and its most crowded city is Istanbul."),
    ("Bugun hava cok guzel, disari cikip biraz yurumek istiyorum.",
     "The weather is very nice today, I want to go out and walk a bit."),
    ("Gecen hafta kucuk bir dil modeli egittim ve sonuclari yazdim.",
     "Last week I trained a small language model and wrote up the results."),
    ("Bilgisayar bilimleri bolumunde okuyan ogrenciler matematik dersi alir.",
     "Students studying in the computer science department take a math course."),
]

# Turkce harfli surum de olcuyorum: isaretli harfler etkiyi degistiriyor mu?
CIFTLER_TR_ISARETLI = [
    "Merhaba, ben bir dil modeliyim ve metin üretmeye çalışıyorum.",
    "Türkiye'nin başkenti Ankara'dır ve en kalabalık şehri İstanbul'dur.",
    "Bugün hava çok güzel, dışarı çıkıp biraz yürümek istiyorum.",
    "Geçen hafta küçük bir dil modeli eğittim ve sonuçları yazdım.",
    "Bilgisayar bilimleri bölümünde okuyan öğrenciler matematik dersi alır.",
]


@torch.no_grad()
def olc(metin):
    """Cumlenin ortalama token loss'unu dondurur."""
    ids = e.encode(metin)
    if len(ids) < 2:
        return None
    x = torch.tensor([ids[:-1]])
    y = torch.tensor([ids[1:]])
    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    return dict(token=len(ids), loss=loss.item(),
                ppl=math.exp(loss.item()))


print("=" * 86)
print("5b) AYNI ICERIK, IKI DIL: GPT-2 NE KADAR SASIRIYOR?")
print("=" * 86)
print(f"  egitimsiz taban: ln({V}) = {TABAN:.4f}")
print()
print(f"  {'':<4}{'Turkce':<10}{'token':>7}{'loss':>9}{'ppl':>11}   "
      f"{'Ingilizce':<12}{'token':>7}{'loss':>9}{'ppl':>11}")
print("  " + "-" * 84)
kayit = []
for i, (t, n) in enumerate(CIFTLER, 1):
    a, b = olc(t), olc(n)
    kayit.append(dict(tr=t, en=n, tr_m=a, en_m=b))
    print(f"  {i:<4}{'':<10}{a['token']:>7}{a['loss']:>9.4f}{a['ppl']:>11.1f}   "
          f"{'':<12}{b['token']:>7}{b['loss']:>9.4f}{b['ppl']:>11.1f}")
print("  " + "-" * 84)
tr_l = sum(k['tr_m']['loss'] for k in kayit) / len(kayit)
en_l = sum(k['en_m']['loss'] for k in kayit) / len(kayit)
tr_t = sum(k['tr_m']['token'] for k in kayit)
en_t = sum(k['en_m']['token'] for k in kayit)
print(f"  {'ORT':<4}{'':<10}{tr_t:>7}{tr_l:>9.4f}{math.exp(tr_l):>11.1f}   "
      f"{'':<12}{en_t:>7}{en_l:>9.4f}{math.exp(en_l):>11.1f}")
print()
print(f"  loss farki      : {tr_l - en_l:+.4f}")
print(f"  perplexity orani: {math.exp(tr_l)/math.exp(en_l):.1f}x")
print(f"  token orani     : {tr_t/en_t:.2f}x")
print()

# isaretli harfli surum
print("  TURKCE HARFLERLE (ç ğ ı ö ş ü) ayni cumleler:")
print(f"  {'#':<4}{'token':>7}{'loss':>9}{'ppl':>11}   {'sade surume gore'}")
print("  " + "-" * 62)
isaretli = []
for i, t in enumerate(CIFTLER_TR_ISARETLI, 1):
    a = olc(t)
    isaretli.append(a)
    sade = kayit[i-1]['tr_m']
    print(f"  {i:<4}{a['token']:>7}{a['loss']:>9.4f}{a['ppl']:>11.1f}   "
          f"token {a['token']-sade['token']:+d}, loss {a['loss']-sade['loss']:+.3f}")
print("  " + "-" * 62)
is_l = sum(a['loss'] for a in isaretli) / len(isaretli)
is_t = sum(a['token'] for a in isaretli)
print(f"  ORT {is_t:>7}{is_l:>9.4f}{math.exp(is_l):>11.1f}")
print()

print(f"""  FARK NEDEN BU KADAR BUYUK

  Ingilizce ortalama loss {en_l:.2f}, Turkce {tr_l:.2f}. Fark
  {tr_l-en_l:.2f} nat. Perplexity cinsinden: Ingilizce'de model bir
  sonraki token icin ortalama {math.exp(en_l):.0f} secenek arasinda
  bocalarken, Turkce'de {math.exp(tr_l):.0f} secenek arasinda
  bocaliyor - {math.exp(tr_l)/math.exp(en_l):.0f} kat fazla.

  Egitimsiz tabana ({TABAN:.2f}) gore bakinca:
    Ingilizce      : tabanin %{100*en_l/TABAN:.0f}'i
    Turkce (sade)  : tabanin %{100*tr_l/TABAN:.0f}'i
    Turkce (isaretli): tabanin %{100*is_l/TABAN:.0f}'i

  Yani 124 milyon parametreli, 40 GB metinle egitilmis bir model,
  Turkce'de hicbir sey bilmeyen bir modelden sadece yuzde
  {100*(1-tr_l/TABAN):.0f} daha iyi. Ingilizce'de ayni oran yuzde
  {100*(1-en_l/TABAN):.0f}.
""")

print("=" * 86)
print("TESTIMI KURARKEN YANILDIM")
print("=" * 86)
print(f"""  Isaretli harfli surumu su mantikla ekledim: "Turkce'nin ozel
  harfleri token sayisini sisiriyor olmali; sade harflerle yazarsam
  token sayisi duser ve tokenizer etkisini ayirt edebilirim."

  Iki beklentim de yanlis cikti.

  BIRINCISI: sade yazim token sayisini AZALTIYOR ({is_t} -> {tr_t},
  yani yuzde {100*(is_t-tr_t)/is_t:.0f} daha az). Buraya kadar tahmin dogru.

  IKINCISI VE YANLIS OLANI: az token = dusuk loss beklemistim. Tam
  tersi oldu. Isaretli surum ortalama {is_l:.2f}, sade surum
  {tr_l:.2f}. Yani DAHA COK token kullanan surum
  {tr_l-is_l:.2f} nat DAHA IYI.

  Sebebini soyle anliyorum: internetteki gercek Turkce metin isaretli
  harflerle yaziliyor. GPT-2'nin verisinde Turkce cok az ama OLAN
  kadari dogru yazilmis Turkce. "Turkiye'nin baskenti" gibi sade
  yazim ise ne Ingilizce ne duzgun Turkce - model icin iki arada bir
  yerde duran, hic gormedigi bir sey.

  Yani token sayisi loss'u belirleyen sey degil. Hafta 9'da
  "tokenizer bir semptom, veri ise sebep" diye yazmistim; bu olcum o
  cumleyi bekledigimden daha guclu destekliyor. Token sayisini
  azaltmak loss'u IYILESTIRMEDI, KOTULESTIRDI - cunku onemli olan
  modelin o yaziyi daha once gorup gormedigi.

  Ve isaretli surum bile ({is_l:.2f}) Ingilizce'nin ({en_l:.2f}) cok
  uzaginda, tabanin %{100*is_l/TABAN:.0f}'inde. Sebep tokenizer
  degil, egitim verisi.

  Son bir not: Hafta 8 ve 9'da "loss'lari ancak olcum kosullari
  ayniysa karsilastirabilirsin" diye yazmistim. Burada o uyari
  GECERLI DEGIL, cunku kosullar ayni: ayni model, ayni tokenizer,
  ayni V. Degisen tek sey dil. Bu yuzden karsilastirma durust.
""")

json.dump(dict(ciftler=[dict(tr=k['tr'], en=k['en'],
                             tr_token=k['tr_m']['token'],
                             tr_loss=k['tr_m']['loss'],
                             tr_ppl=k['tr_m']['ppl'],
                             en_token=k['en_m']['token'],
                             en_loss=k['en_m']['loss'],
                             en_ppl=k['en_m']['ppl']) for k in kayit],
               tr_ort=tr_l, en_ort=en_l, tr_token=tr_t, en_token=en_t,
               isaretli_ort=is_l, isaretli_token=is_t,
               taban=TABAN),
          open('sonuc_gorev5b.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev5b.json kaydedildi")
