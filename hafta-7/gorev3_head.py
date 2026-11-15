# Gorev 3 - Tek bir self-attention head
# YZ50 Hafta 7 (58:26 - 1:19:11)

import torch
import torch.nn as nn
from torch.nn import functional as F
from ortak import V, itos, get_batch, BLOCK

torch.manual_seed(1337)
B, T, C = 4, 8, 32
HEAD = 16
x = torch.randn(B, T, C)

print("=" * 64)
print("3a) QUERY, KEY, VALUE")
print("=" * 64)
print("""
  Her token UC sey uretiyor:
    query (q) : "ben ne ariyorum"
    key   (k) : "bende ne var"
    value (v) : "bana bakarsan sana ne veririm"

  t. token'in j. token'a verecegi agirlik = q[t] ile k[j]'nin ic carpimi.
  Yani aradigim sey ile sendeki sey ne kadar ortusuyorsa o kadar bakiyorum.

  Analoji: bir kutuphanede arama. Elinde bir arama cumlesi var (query).
  Her kitabin sirtinda bir etiket var (key). Etiketler aramanla ne kadar
  ortusuyorsa o kitaba o kadar agirlik veriyorsun. Sonra agirliklara gore
  kitaplarin ICERIGINI (value) karistirip tek bir ozet cikariyorsun.
""")

key = nn.Linear(C, HEAD, bias=False)
query = nn.Linear(C, HEAD, bias=False)
value = nn.Linear(C, HEAD, bias=False)

k = key(x)        # (B, T, HEAD)
q = query(x)
v = value(x)
print(f"  x    : {tuple(x.shape)}")
print(f"  q, k : {tuple(q.shape)}")
print()

wei = q @ k.transpose(-2, -1) * HEAD**-0.5   # olcekleme dahil (bkz. 3c)
print(f"  wei = q @ k.T / sqrt(head_size)  ->  {tuple(wei.shape)}")
print()

tril = torch.tril(torch.ones(T, T))
wei_maskesiz = wei.clone()
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ v
print(f"  cikti: {tuple(out.shape)}")
print()


# ==========================================================
# 3b) BIR SATIRI OKU
# ==========================================================
print("=" * 64)
print("3b) AGIRLIK MATRISININ BIR SATIRINI OKUMAK")
print("=" * 64)
torch.set_printoptions(precision=3, sci_mode=False)
print("  Ilk parcanin wei matrisi (8x8):")
print(wei[0])
print()
satir = 4
print(f"  {satir}. satiri okuyalim (yani {satir}. token):")
for j in range(T):
    agirlik = wei[0, satir, j].item()
    cubuk = '#' * int(agirlik * 60)
    durum = "gelecek, kapali" if j > satir else ("kendisi" if j == satir else "gecmis")
    print(f"    token {j}: {agirlik:.3f}  {cubuk:<30} ({durum})")
print()
print(f"  Satir toplami: {wei[0, satir].sum().item():.4f}  (softmax oldugu icin 1)")
print()
print("  KENDI CUMLELERIMLE: bu satir '4. token, gecmisteki 5 token'in")
print("  her birine ne kadar bakiyor' sorusunun cevabi. Agirliklar esit")
print("  degil: model bazi token'lari daha onemli buluyor. Ve 5, 6, 7")
print("  numarali token'lar tamamen sifir, cunku onlar gelecekte.")
print()


# ==========================================================
# 3c) NEDEN KAREKOK HEAD_SIZE'A BOLUYORUZ
# ==========================================================
print("=" * 64)
print("3c) NEDEN head_size'IN KAREKOKUNE BOLUYORUZ")
print("=" * 64)
print()
torch.manual_seed(42)
k2 = torch.randn(B, T, HEAD)
q2 = torch.randn(B, T, HEAD)
ham = q2 @ k2.transpose(-2, -1)
olcekli = ham * HEAD**-0.5

print(f"  q ve k birim varyansli (std ~ 1).")
print(f"  olceklenmemis wei varyansi : {ham.var().item():.3f}   <- head_size ({HEAD}) kadar buyumus")
print(f"  olceklenmis   wei varyansi : {olcekli.var().item():.3f}   <- tekrar ~1")
print()
print("  Sebep: ic carpim 16 terimin toplami. Bagimsiz birim varyansli")
print("  16 sayiyi toplayinca varyans 16 olur, std 4 olur. Yani skorlar")
print("  head_size buyudukce otomatik olarak buyuyor.")
print()

print("  Peki bu neden sorun? softmax'a bakalim.")
ornek = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
print()
print(f"  skorlar (makul): {ornek.tolist()}")
print(f"    softmax      : {[round(v,3) for v in F.softmax(ornek, -1).tolist()]}")
print(f"  skorlar x 8    : {[round(v,1) for v in (ornek*8).tolist()]}")
print(f"    softmax      : {[round(v,3) for v in F.softmax(ornek*8, -1).tolist()]}")
print(f"  skorlar x 30   : {[round(v,1) for v in (ornek*30).tolist()]}")
print(f"    softmax      : {[round(v,3) for v in F.softmax(ornek*30, -1).tolist()]}")
print()
print("  Buyuk skorlarda softmax neredeyse tek bir token'a 1 veriyor,")
print("  digerlerine 0. Yani agirlikli ortalama almayi birakip tek bir")
print("  token'i secmeye donuyor (one-hot gibi).")
print()
print("  Baslangicta bu kotu: model daha hicbir sey ogrenmeden rastgele")
print("  bir token'a kilitleniyor ve gradient diger token'lara akmiyor.")
print("  Hafta 4'teki tanh doymasiyla ayni mantik: cikti uca yapisinca")
print("  turev sifira gidiyor.")
print()

# gercek wei uzerinde yan yana
wm = wei_maskesiz[0]
tril8 = torch.tril(torch.ones(T, T))
a = F.softmax(wm.masked_fill(tril8 == 0, float('-inf')), -1)[4]
b = F.softmax((wm * HEAD**-0.5).masked_fill(tril8 == 0, float('-inf')), -1)[4]
print("  Gercek wei'nin 4. satiri, yan yana:")
print(f"    bolmeden : {[round(v,3) for v in a[:5].tolist()]}  (maks {a.max().item():.3f})")
print(f"    bolerek  : {[round(v,3) for v in b[:5].tolist()]}  (maks {b.max().item():.3f})")
print()
print("  Bolerek alinan dagilim daha duz. Model baslangicta butun gecmise")
print("  bakip zamanla kime bakacagini ogreniyor.")
print()


# ==========================================================
# 3d) POZISYON EMBEDDING'I
# ==========================================================
print("=" * 64)
print("3d) ATTENTION SIRAYI BILMEZ")
print("=" * 64)
print()
print("  wei[t,j] sadece q[t] ile k[j]'nin ic carpimi. Bu hesapta j'nin")
print("  KACINCI token oldugu hic gecmiyor. Token'lari karistirsak")
print("  ayni sonucu alirdik.")
print()
print("  Deneyelim: ayni iki token'i yer degistirip skoru karsilastiralim.")
i, j = 2, 5
s1 = (q[0, 7] @ k[0, i]).item()
s2 = (q[0, 7] @ k[0, j]).item()
kk = k[0].clone()
kk[[i, j]] = kk[[j, i]]
s1y = (q[0, 7] @ kk[j]).item()
print(f"    token {i} icin skor            : {s1:.4f}")
print(f"    token {i} (yeri {j}'ye tasindi) : {s1y:.4f}   <- ayni")
print()
print("  Yani attention bir TORBA islemi, sira bilgisi tasimiyor.")
print("  Cozum: her pozisyona da bir embedding ogretmek ve token")
print("  embedding'iyle toplamak:")
print()
print("      x = token_embedding(idx) + position_embedding(arange(T))")
print()
print("  Boylece 'ben 3. sıradaki e harfiyim' bilgisi vektore giriyor.")
print("  Hafta 6'daki WaveNet'te buna gerek yoktu: orada sira, agacin")
print("  yapisinda zaten kodluydu.")
