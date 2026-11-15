# Gorev 2 - Gecmisin ortalamasini uc yoldan almak
# YZ50 Hafta 7 (42:13 - 58:26)
#
# Self-attention'in arkasindaki hile: "gecmis token'larin ortalamasini al"
# islemi bir MATRIS CARPIMI olarak yazilabiliyor. Bu yazilinca
# agirliklari ogrenilebilir hale getirmek kolay oluyor.

import torch
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2          # 4 parca, 8 zaman adimi, 2 kanal
x = torch.randn(B, T, C)
print(f"x sekli: {tuple(x.shape)}   (parca, zaman, kanal)")
print()
print("Hedef: her t adiminda, 0..t arasindaki butun token'larin ORTALAMASI.")
print("Yani her token 'gecmiste ne gordum' ozetini tasisin.")
print()


# ==========================================================
# YOL 1: for dongusu (en acik ama en yavas)
# ==========================================================
xbow1 = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]        # (t+1, C)
        xbow1[b, t] = xprev.mean(0)

print("YOL 1 - for dongusu")
print("  for b: for t: xbow[b,t] = x[b,:t+1].mean(0)")
print()


# ==========================================================
# YOL 2: lower triangular matris ile carpim
# ==========================================================
wei = torch.tril(torch.ones(T, T))
print("YOL 2 - tril matrisi")
print("  torch.tril(torch.ones(8,8)):")
print(wei.int().numpy())
print()
wei = wei / wei.sum(1, keepdim=True)
print("  satir toplamina bolunce (agirliklar):")
torch.set_printoptions(precision=3, sci_mode=False)
print(wei)
print()
xbow2 = wei @ x                      # (T,T) @ (B,T,C) -> (B,T,C)
print(f"  wei @ x :  ({T},{T}) @ ({B},{T},{C}) -> {tuple(xbow2.shape)}")
print()


# ==========================================================
# YOL 3: softmax
# ==========================================================
tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros((T, T))
wei3 = wei3.masked_fill(tril == 0, float('-inf'))
print("YOL 3 - softmax")
print("  once sifirlardan basla, gelecegi -inf yap:")
print(wei3[:4, :4])
wei3 = F.softmax(wei3, dim=-1)
print("  sonra softmax al -> ayni agirliklar:")
print(wei3[:4, :4])
xbow3 = wei3 @ x
print()


# ==========================================================
# UCU AYNI MI
# ==========================================================
print("=" * 58)
print("DOGRULAMA")
print("=" * 58)
print(f"  torch.allclose(yol1, yol2)                 : {torch.allclose(xbow1, xbow2)}")
print(f"  torch.allclose(yol1, yol3)                 : {torch.allclose(xbow1, xbow3)}")
print(f"  torch.allclose(yol1, yol3, atol=1e-6)      : {torch.allclose(xbow1, xbow3, atol=1e-6)}")
print(f"  maksimum fark                              : {(xbow1 - xbow3).abs().max().item():.2e}")
print()
print("  DIKKAT: varsayilan allclose False dondu. Sebep hata degil:")
print("  varsayilan atol=1e-8 ve bizim farkimiz 3.2e-08. Yani fark")
print("  float aritmetiginin kendi gurultusu seviyesinde. atol'u 1e-6")
print("  yapinca True donuyor.")
print("  Hafta 5'te de ayni ayrim vardi: EXACT (bit duzeyinde ayni) ve")
print("  approx (1e-9 civari fark). Buradaki durum 'approx'.")
print()


# ==========================================================
# KENDI CUMLELERIMLE
# ==========================================================
print("=" * 58)
print("MATRIS CARPIMI NEDEN AGIRLIKLI ORTALAMA")
print("=" * 58)
print("""
  wei @ x isleminde, sonucun t. satiri soyle hesaplaniyor:
  wei'nin t. SATIRINI al, x'in butun satirlariyla carp, topla.

      sonuc[t] = wei[t,0]*x[0] + wei[t,1]*x[1] + ... + wei[t,T-1]*x[T-1]

  Bu tam olarak agirlikli ortalamanin tanimi. wei[t,j], "t. token
  j. token'a ne kadar baksin" demek.

  tril kullanmamizin sebebi: ust ucgen sifir oldugu icin gelecekteki
  token'larin agirligi sifir. Yani t. token sadece kendisini ve
  gecmisi goruyor.

  Satir toplamina bolmek de agirliklarin toplaminin 1 olmasini
  sagliyor; boylece bu bir ortalama oluyor, toplam degil.

  SIMDI ASIL NOKTA: su anda butun agirliklar esit (1/3, 1/3, 1/3).
  Yani "gecmise bak ama hepsine ayni derecede bak" diyoruz.
  Self-attention bu agirliklari VERIYE BAKARAK belirliyor:
  bir sessiz harf kendinden onceki sesliyi arayabilir, bir kapanis
  tirnagi acilis tirnagini arayabilir.

  Softmax versiyonunun onemi de burada: wei'yi sifirdan baslatmak
  yerine, ogrenilen skorlardan baslatinca formul hic degismiyor.
  Sadece sifirlarin yerine gercek skorlar geliyor.
""")
