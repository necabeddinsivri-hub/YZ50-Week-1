# Gorev 1 - Veriyi hazirla: token'la, dosyalara bol, bir dosyayi ayir
# YZ50 Hafta 12  (video 3:10:21 - 3:23:10)
#
# Karpathy FineWeb-Edu'nun 10 milyar token'lik ornegini 100'er milyon
# token'lik 100 dosyaya boluyor ve ilk dosyayi dogrulama (val) icin
# ayiriyor. Ben ayni sekli cok daha kucuk olcekte kuruyorum.
#
# VERI NEREDEN GELDI
# Calistigim ortamdan huggingface.co'ya cikis kapali. FineWeb-Edu'yu
# tarayicidan, Hugging Face'in resmi satir API'si (datasets-server)
# uzerinden cektim: sample-10BT'nin ilk belgeleri, 100'er belgelik
# isteklerle. 4.000. belgeden sonra API istekleri kesti (hiz siniri);
# elimde 4.000 belge kaldi. Bunun ne kadar token ettigini asagida
# olcuyorum ve neden yettigini README'de yaziyorum.

import json
import os

import numpy as np

from bicim import bin as bn
from tokenizer import enc

HAM = 'veri/ham/fineweb_edu_4k.jsonl'
CIKIS = 'veri/shardlar'
SHARD = 500_000          # token / dosya
os.makedirs(CIKIS, exist_ok=True)

e = enc()
EOT = e.eot_token        # <|endoftext|> = 50256
print("=" * 78)
print("1) FINEWEB-EDU'YU TOKEN'LA VE DOSYALARA BOL")
print("=" * 78)

# ---------------------------------------------------------------------
# Her belgenin BASINA <|endoftext|> koyuyorum. Karpathy de oyle yapiyor:
# boylece model "yeni bir belge basliyor" sinyalini goruyor ve iki
# belge birbirine karisiyormus gibi ogrenmiyor. Hafta 10'da bu token'in
# GPT-2'nin vocab'indaki son token oldugunu gormustuk (50256).
# ---------------------------------------------------------------------
belgeler = [json.loads(s) for s in open(HAM, encoding='utf-8')]
karakter = sum(len(b['text']) for b in belgeler)
tokenler = []
for b in belgeler:
    tokenler.append(EOT)
    tokenler.extend(e.encode_ordinary(b['text']))
tokenler = np.array(tokenler, dtype=np.int64)
assert tokenler.max() < 2 ** 16, "uint16'ya sigmiyor"
tokenler = tokenler.astype(np.uint16)

print(f"  belge sayisi      : {bn(len(belgeler))}")
print(f"  karakter          : {bn(karakter)}")
print(f"  token             : {bn(len(tokenler))}")
print(f"  token / belge     : {len(tokenler)/len(belgeler):.0f}")
print(f"  karakter / token  : {karakter/len(tokenler):.2f}")
print()
print("""  NEDEN uint16
  GPT-2'nin vocab'i 50.257, yani en buyuk token id'si 50.256. uint16
  0..65.535 arasini tutuyor; her token 2 bayt. int64 kullansaydim ayni
  veri 4 kat yer kaplardi. Karpathy de ayni nedenle uint16 kullaniyor.
""")

# ---------------------------------------------------------------------
# Dosyalara bol. Ilk dosya val: model egitim sirasinda bu token'lari
# HIC gormuyor, o yuzden val loss "ezberledi mi, genelliyor mu"
# sorusunu cevaplayabiliyor.
# ---------------------------------------------------------------------
for f in os.listdir(CIKIS):
    if f.endswith('.npy'):
        os.remove(os.path.join(CIKIS, f))
dosyalar = []
for i, bas in enumerate(range(0, len(tokenler), SHARD)):
    parca = tokenler[bas:bas + SHARD]
    tur = 'val' if i == 0 else 'train'
    ad = f"edufineweb_{tur}_{i:06d}.npy"
    np.save(os.path.join(CIKIS, ad), parca)
    dosyalar.append(dict(ad=ad, tur=tur, token=int(len(parca))))
    print(f"  {ad:<34}{bn(len(parca)):>10} token")
train_tok = sum(d['token'] for d in dosyalar if d['tur'] == 'train')
val_tok = sum(d['token'] for d in dosyalar if d['tur'] == 'val')
print()
print(f"  egitim : {bn(train_tok)} token, "
      f"{sum(d['tur']=='train' for d in dosyalar)} dosya")
print(f"  val    : {bn(val_tok)} token, 1 dosya")
print()

# ---------------------------------------------------------------------
# Saglik kontrolu: bir dosyayi geri okuyup cozunce metin yerinde mi?
# ---------------------------------------------------------------------
geri = np.load(os.path.join(CIKIS, dosyalar[1]['ad']))
ornek = e.decode(geri[:60].astype(np.int64).tolist())
print("  geri okuma kontrolu (train dosyasinin ilk 60 token'i):")
print("   ", repr(ornek[:240]))
print()

json.dump(dict(belge=len(belgeler), karakter=karakter,
               token=int(len(tokenler)), shard_boyu=SHARD,
               dosyalar=dosyalar, train_token=train_tok, val_token=val_tok,
               kar_token=karakter / len(tokenler)),
          open('sonuc_gorev1.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev1.json kaydedildi")
