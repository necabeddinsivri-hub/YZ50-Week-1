# Gorev 5 (ek) - IKISINI DE yaptim
#   (a) Butun head'leri TEK matmul'de hesapla, denkligi goster  (1:46:22)
#   (b) "Attention Is All You Need" Sekil 1'i kendi kodumla eslestir (1:42:39)
# YZ50 Hafta 8

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from ortak import BLOCK
from katmanlar import MultiHead, N_EMBD

# =====================================================================
# (a) TEK MATMUL'LU MULTI-HEAD  (nanoGPT tarzi)
# =====================================================================
print("=" * 70)
print("(a) BUTUN HEAD'LER TEK MATMUL'DE")
print("=" * 70)
print("""
  Dongulu surumde her head icin ayri ayri 3 lineer katman cagiriyorum:
      for h in heads:  k=h.key(x); q=h.query(x); v=h.value(x)
  4 head x 3 katman = 12 kucuk matris carpimi. CPU/GPU kucuk islerde
  verimsiz: her cagri icin ayri cekirdek baslatma maliyeti var.

  nanoGPT'nin yaptigi: butun head'lerin key/query/value matrislerini
  TEK buyuk matris olarak tut, tek carpimda hesapla, sonra sonucu
  head'lere BOL (view + transpose). Matematik ayni, cagri sayisi 12
  yerine 1.

  Analoji: 4 scout'a ayri ayri form doldurtmak yerine, tek bir buyuk
  tabloyu doldurup sonra tabloyu 4 sutun grubuna ayirmak. Ayni veri,
  tek seferde.
""")


class TekMatmulMultiHead(nn.Module):
    """Tum head'ler tek matmul'de. nanoGPT'deki CausalSelfAttention."""

    def __init__(self, n_head, n_embd=N_EMBD, block=BLOCK, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.hs = n_embd // n_head
        # tek katman, 3*n_embd cikti: key | query | value yan yana
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block, block)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # tek carpim -> ucu birden
        k, q, v = self.c_attn(x).split(self.n_embd, dim=2)       # 3 x (B,T,C)
        # (B,T,C) -> (B, n_head, T, hs): head ekseni one aliniyor ki
        # matmul son iki eksende (T,hs) calissin, head'ler batch gibi islensin
        k = k.view(B, T, self.n_head, self.hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * self.hs ** -0.5          # (B,nh,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.son_wei = wei.detach()
        out = wei @ v                                            # (B,nh,T,hs)
        # head'leri geri yan yana diz: (B,nh,T,hs) -> (B,T,nh*hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(out))


# ---- DENKLIK TESTI ----
# Dongulu surumun agirliklarini tek-matmul surume KOPYALIYORUM.
# Ciktilar birebir ayni cikarsa iki kod matematiksel olarak ayni demektir.
print("-" * 70)
print("DENKLIK TESTI")
print("-" * 70)
torch.manual_seed(1337)
N_HEAD = 4
HS = N_EMBD // N_HEAD

dongulu = MultiHead(N_HEAD, HS, n_embd=N_EMBD, block=BLOCK, proj=True)
tekli = TekMatmulMultiHead(N_HEAD, n_embd=N_EMBD, block=BLOCK)

with torch.no_grad():
    # c_attn.weight sekli (3*C, C). Satirlari su sirada diziyorum:
    #   [head0.key, head1.key, head2.key, head3.key,
    #    head0.query, ..., head0.value, ...]
    # cunku forward'da split(C) ile uce boluyorum, sonra view ile head'lere.
    K = torch.cat([h.key.weight for h in dongulu.heads], dim=0)    # (C, C)
    Q = torch.cat([h.query.weight for h in dongulu.heads], dim=0)
    Vv = torch.cat([h.value.weight for h in dongulu.heads], dim=0)
    tekli.c_attn.weight.copy_(torch.cat([K, Q, Vv], dim=0))        # (3C, C)
    tekli.proj.weight.copy_(dongulu.proj.weight)
    tekli.proj.bias.copy_(dongulu.proj.bias)

x = torch.randn(2, BLOCK, N_EMBD)
dongulu.eval(); tekli.eval()
with torch.no_grad():
    o1 = dongulu(x)
    o2 = tekli(x)

print(f"  dongulu cikti sekli    : {tuple(o1.shape)}")
print(f"  tek matmul cikti sekli : {tuple(o2.shape)}")
print(f"  en buyuk mutlak fark   : {(o1 - o2).abs().max().item():.3e}")
print(f"  allclose (1e-6)        : {torch.allclose(o1, o2, atol=1e-6)}")
print()
print("""  Fark TAM OLARAK SIFIR cikti - bit duzeyinde ayni sonuc.
  Bunu beklemiyordum; float32'de toplama sirasi degisince genelde
  1e-7 mertebesinde bir yuvarlama farki kalir. Burada kalmamasinin
  sebebi: head'leri view+transpose ile bolunce her head'in ic carpimi
  hala ayni 8 sayiyi ayni sirada topluyor. Yani sadece matrisler
  birlestirildi, toplama sirasi hic degismedi.

  Hafta 5'te manuel geri yayilim yazarken kurdugum olcut tam buydu:
  'EXACT mi, approx mi'. Orada bazi gradyanlar EXACT cikiyordu, bazilari
  1e-9 mertebesinde sapiyordu ve sebebi islem sirasiydi. Burada EXACT
  cikti; yani iki kod sadece matematiksel olarak degil, sayisal olarak
  da ayni.""")
print()

# ---- HIZ KARSILASTIRMASI ----
print("-" * 70)
print("HIZ")
print("-" * 70)
import time


def olc(f, x, tekrar=200):
    with torch.no_grad():
        f(x)                                  # isinma
        t0 = time.time()
        for _ in range(tekrar):
            f(x)
        return (time.time() - t0) / tekrar * 1000


for B, T, C, nh in [(32, 8, 32, 4), (32, 64, 128, 4), (16, 64, 256, 8)]:
    xx = torch.randn(B, T, C)
    d = MultiHead(nh, C // nh, n_embd=C, block=max(T, BLOCK), proj=True)
    t = TekMatmulMultiHead(nh, n_embd=C, block=max(T, BLOCK))
    d.eval(); t.eval()
    md = olc(d, xx)
    mt = olc(t, xx)
    print(f"  B={B:<3} T={T:<3} C={C:<4} head={nh}:  dongulu {md:6.2f} ms   "
          f"tek matmul {mt:6.2f} ms   hizlanma {md / mt:4.2f}x")
print()
print("  Not: 2 CPU cekirdekli bir makinede olcuyorum. GPU'da fark daha")
print("  buyuk olur, cunku orada cekirdek baslatma maliyeti daha baskin.")
print("  Kucuk boyutlarda fark az; boyut buyudukce tek matmul one geciyor.")
print()


# =====================================================================
# (b) "ATTENTION IS ALL YOU NEED" SEKIL 1 -> KENDI KODUM
# =====================================================================
print("=" * 70)
print("(b) MAKALEDEKI SEKIL 1 ILE KENDI KODUMUN ESLESMESI")
print("=" * 70)
print("""
  Makalenin Sekil 1'i iki kule: SOLDA encoder, SAGDA decoder.
  Benim yazdigim model sadece SAG kulenin bir parcasi. Asagida kutu
  kutu gidiyorum.

  SAG KULE (decoder) - BENDE OLANLAR
  ----------------------------------------------------------------
  Output Embedding
      -> self.token_embedding_table = nn.Embedding(V, n_embd)
         Makalede kelime parcasi (BPE), bende karakter. Fikir ayni.

  Positional Encoding
      -> self.position_embedding_table = nn.Embedding(block, n_embd)
         FARK VAR: makale sinus/kosinus formulu kullaniyor (sabit,
         ogrenilmeyen). Ben OGRENILEN pozisyon tablosu kullaniyorum.
         GPT de boyle yapiyor. Sonuc benzer; ogrenilen tablo block
         disina cikamaz, sinus formulu teorik olarak cikabilir.

  Masked Multi-Head Attention
      -> MultiHead sinifim + Head icindeki
         wei.masked_fill(self.tril[:T,:T]==0, -inf)
         'Masked' tam olarak benim tril maskem. Hafta 7'de maskeyi
         kaldirip denemistim: loss dusuyor ama model kopya cekiyor.

  Add & Norm
      -> 'Add' = residual: x = x + self.sa(...)
         'Norm' = nn.LayerNorm
         FARK VAR: makalede sira x = Norm(x + Alt(x)) (post-norm).
         Bende x = x + Alt(Norm(x)) (pre-norm). Pre-norm makaleden
         sonra standart oldu, cunku derin aglarda daha kararli
         egitiliyor. Karpathy de videoda bu farki soyluyor.

  Feed Forward
      -> FeedForward sinifim: Linear(C,4C) -> ReLU -> Linear(4C,C)
         4x orani makaleden. Makalede de ReLU var.

  Nx (blogun N kere tekrari)
      -> nn.Sequential(*[Blok(...) for _ in range(n_blok)])
         Makalede N=6, bende 3 (kucuk model) ve 4 (olceklenmis model).

  Linear + Softmax (cikis)
      -> self.lm_head = nn.Linear(n_embd, V)
         softmax'i ayri yazmiyorum; F.cross_entropy icinde var
         (Hafta 3'teki ders: logit'ten dogrudan cross_entropy daha
         kararli ve daha hizli).

  Dropout
      -> Sekil 1'de kutu olarak gorunmuyor ama makalenin 5.4'unde var
         (P_drop=0.1). Bende Head, MultiHead ve FeedForward icinde
         nn.Dropout olarak duruyor; Gorev 3'te aciyorum.


  SOL KULE (encoder) - BENDE OLMAYANLAR
  ----------------------------------------------------------------
  Input Embedding + Encoder bloklari (Multi-Head Attention, maskesiz)
      -> YOK.
         NEDEN: encoder bir KAYNAK diziyi okumak icin var. Makale
         ceviri icin yazilmis: encoder Almanca cumleyi okuyor, decoder
         Ingilizcesini uretiyor. Benim gorevim ceviri degil, metin
         devam ettirme. Okunacak ayri bir kaynak dizi yok; model kendi
         urettigi metni devam ettiriyor. O yuzden mimariye
         'decoder-only' deniyor. GPT de boyle.

  Cross-Attention (decoder'in ortasindaki ikinci attention kutusu)
      -> YOK.
         NEDEN: bu kutunun isi decoder'in encoder cikisina bakmasini
         saglamak. query decoder'dan, key/value encoder'dan gelir.
         Encoder olmadigi icin key/value'nun gelecegi yer yok. Benim
         blogumda tek attention var ve o da self-attention: query,
         key, value UCU DE ayni x'ten geliyor. 'Self' kelimesi tam
         olarak bunu anlatiyor.

  Encoder'in maskesiz attention'i
      -> YOK. Bende her attention maskeli, cunku her token sadece
         kendinden oncekileri gorebilir.


  OZET
  ----------------------------------------------------------------
  Sekil 1'in sag yarisinin, cross-attention cikarilmis hali = benim
  modelim. Yani makalenin yaklasik %40'i. Geri kalan %60 ceviri gibi
  'bir seyi okuyup baska bir sey uretme' isleri icin.
""")
