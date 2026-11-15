# katmanlar.py - Transformer blogunun parcalari, tek tek
# YZ50 Hafta 8
#
# Burada bloğu parca parca kuruyorum. Her parca ayri bir sinif, cunku
# Gorev 2'de "her parcayi ekledikten sonra val loss'u kaydet" deniyor;
# o yuzden parcalari acip kapatabilmem lazim.
#
# Sira (Karpathy'nin videodaki sirasi):
#   Head           -> tek head self-attention            (Hafta 7)
#   MultiHead      -> ayni boyutu n_head parcaya bol     (1:21:59)
#   FeedForward    -> her token kendi basina dusunur     (1:24:25)
#   Blok           -> iletisim + dusunme, residual ile   (1:27:00)
#   LayerNorm      -> satir bazinda normalizasyon        (1:32:00)

import torch
import torch.nn as nn
from torch.nn import functional as F
from ortak import V, BLOCK

N_EMBD = 32      # Hafta 7 ile ayni: adil karsilastirma icin sabit


# =====================================================================
# 1) TEK HEAD  (Hafta 7'den aynen)
# =====================================================================
class Head(nn.Module):
    """Tek bir self-attention head.

    key   : "bende ne var"
    query : "ben ne ariyorum"
    value : "secilirsem ne veririm"
    """

    def __init__(self, head_size, n_embd=N_EMBD, block=BLOCK, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block, block)))
        self.drop = nn.Dropout(dropout)
        self.head_size = head_size
        self.son_wei = None            # gorsellestirme icin

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                     # (B,T,hs)
        q = self.query(x)                                   # (B,T,hs)
        # olcekleme: head_size buyudukce iç carpimlarin varyansi buyuyor,
        # softmax sivrilesiyor. sqrt(head_size)'a boluyoruz (Hafta 7).
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.son_wei = wei.detach()
        wei = self.drop(wei)
        v = self.value(x)
        return wei @ v                                      # (B,T,hs)


# =====================================================================
# 2) MULTI-HEAD  (1:21:59 - 1:24:25)
# =====================================================================
class MultiHead(nn.Module):
    """n_head tane kucuk head, ciktilar yan yana eklenir.

    DIKKAT: toplam boyut BUYUMUYOR. 32 boyutu 4 head x 8 boyut olarak
    boluyorum. Yani parametre sayisi neredeyse ayni kaliyor
    (sadece proj katmani ekleniyor), ama 4 ayri iletisim kanali oluyor.

    Analoji (basketbol): tek head = sahaya tek bir goz. Multi-head =
    4 ayri scout. Biri sadece top tasiyana bakiyor, biri pota altini
    izliyor, biri rakip savunmanin duzenine, biri sure/skor tablosuna.
    Her biri kendi 8 boyutluk raporunu yaziyor, sonra raporlar
    birlestirilip tek sayfa yapiliyor (proj).
    """

    def __init__(self, n_head, head_size, n_embd=N_EMBD, block=BLOCK,
                 dropout=0.0, proj=True):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd=n_embd, block=block, dropout=dropout)
            for _ in range(n_head)])
        # proj: head'lerin ciktilarini karistiran lineer katman. Residual
        # yoluna geri donmeden once "raporlari harmanlama" adimi.
        self.proj = nn.Linear(head_size * n_head, n_embd) if proj else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B,T,n_head*hs)
        if self.proj is not None:
            out = self.drop(self.proj(out))
        return out


# =====================================================================
# 3) FEEDFORWARD  (1:24:25)
# =====================================================================
class FeedForward(nn.Module):
    """Her token KENDI basina, digerlerine bakmadan hesap yapar.

    Attention = iletisim (tokenlar birbirine bakar).
    FeedForward = dusunme (toplanan bilgiyi sindirme).

    Attention'dan sonra token elinde bir bilgi harmani var ama onu
    hemen logit'e ceviriyordum - arada hic "dusunme" adimi yoktu.
    Bu katman tam o bosluga giriyor.

    Ic boyut 4x: makaledeki oran. 32 -> 128 -> 32. Genis bir ara kat
    + ReLU, modelin dogrusal olmayan desen ogrenmesini sagliyor.
    """

    def __init__(self, n_embd=N_EMBD, kat=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, kat * n_embd),
            nn.ReLU(),
            nn.Linear(kat * n_embd, n_embd),     # projeksiyon: residual'a donus
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================
# 4) KENDI LAYERNORM'UM  (1:32:00) - Hafta 6'daki BatchNorm1d'in esi
# =====================================================================
class LayerNorm1d:
    """Hafta 6'da yazdigim BatchNorm1d'in neredeyse kopyasi.

    TEK ONEMLI FARK: hangi eksende ortalama aliniyor.
      BatchNorm1d : dim=0  -> SUTUN bazinda (batch boyunca)
      LayerNorm1d : dim=1  -> SATIR bazinda (ozellikler boyunca)

    Bunun iki sonucu var:
      a) ornekler artik birbirine bagli degil. BatchNorm'da bir ornegin
         cikti degeri, ayni batch'e hangi orneklerin dustugune bagliydi.
         LayerNorm'da her satir kendi basina normalize ediliyor.
      b) running_mean / running_var TUTMAYA GEREK YOK. BatchNorm'da
         egitimde batch istatistigi, cikarimda biriktirilmis istatistik
         kullaniliyordu - cunku tek ornek gelince batch ortalamasi
         hesaplanamaz. LayerNorm'da tek ornek de kendi satirini
         normalize edebilir. Egitim ve cikarim ayni kodu kullanir;
         self.training bayragina bile gerek kalmaz.
    """

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)              # satir ortalamasi
        # unbiased=False: n'e bol, n-1'e DEGIL.
        # Hafta 6'da BatchNorm yazarken Bessel duzeltmesini (n-1)
        # kullanmistim, cunku orada batch butun verinin bir ORNEGI;
        # ornekten evren varyansini kestirirken n-1 gerekiyor.
        # LayerNorm'da ise normalize ettigim sey tek bir token'in
        # butun ozellikleri: ortada kestirim yok, elimdeki veri
        # evrenin kendisi. O yuzden n'e boluyorum. nn.LayerNorm da
        # boyle yapiyor; ilk yazdigimda n-1 kullanip 1.66e-01 fark
        # almistim, hata buradaydi.
        xvar = x.var(1, keepdim=True, unbiased=False)  # satir varyansi
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# =====================================================================
# 5) BLOK  (iletisim + dusunme, residual + LayerNorm secmeli)
# =====================================================================
class Blok(nn.Module):
    """Bir Transformer blogu: multi-head attention + feedforward.

    residual=True  -> x = x + alt_katman(x)  (gradyan icin otoyol)
    norm=True      -> pre-norm: x = x + alt_katman(ln(x))
    """

    def __init__(self, n_embd, n_head, block=BLOCK, residual=True, norm=True,
                 dropout=0.0):
        super().__init__()
        hs = n_embd // n_head
        self.sa = MultiHead(n_head, hs, n_embd=n_embd, block=block,
                            dropout=dropout)
        self.ffwd = FeedForward(n_embd, dropout=dropout)
        self.residual = residual
        self.norm = norm
        if norm:
            # burada nn.LayerNorm kullaniyorum; yukaridaki LayerNorm1d
            # kendi elimle yazdigim hali ve gorev2'de ikisinin ayni
            # sonucu verdigini sayisal olarak gosteriyorum.
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        a = self.ln1(x) if self.norm else x
        x = x + self.sa(a) if self.residual else self.sa(a)
        b = self.ln2(x) if self.norm else x
        x = x + self.ffwd(b) if self.residual else self.ffwd(b)
        return x


# =====================================================================
# 6) MODELLER - ablasyon tablosunun her satiri icin bir tane
# =====================================================================
class BigramDilModeli(nn.Module):
    """Satir 1: taban. Her token dogrudan bir sonrakinin skorlarini okur."""

    def __init__(self, v=V):
        super().__init__()
        self.token_embedding_table = nn.Embedding(v, v)
        self.block = BLOCK

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        return _kayip(logits, targets)

    generate = None      # asagida ortak generate baglanacak


class AblasyonModeli(nn.Module):
    """Satir 2-6: ayni iskelet, parcalar acilip kapanarak.

    asama:
      'tek_head'   -> 1 head (32 boyut), ff yok, residual yok, norm yok
      'multi_head' -> 4 head x 8 boyut, ff yok
      'ff'         -> multi-head + feedforward (tek katman, residual yok)
      'residual'   -> 3 blok, residual VAR, LayerNorm yok
      'ln'         -> 3 blok, residual + LayerNorm
      'blok_duz'   -> 3 blok, residual YOK, norm YOK (tesh is icin)
    """

    def __init__(self, asama, n_embd=N_EMBD, n_head=4, n_blok=3,
                 block=BLOCK, dropout=0.0, v=V):
        super().__init__()
        self.asama = asama
        self.block = block
        self.token_embedding_table = nn.Embedding(v, n_embd)
        self.position_embedding_table = nn.Embedding(block, n_embd)

        if asama == 'tek_head':
            self.govde = Head(n_embd, n_embd=n_embd, block=block)
        elif asama == 'multi_head':
            self.govde = MultiHead(n_head, n_embd // n_head, n_embd=n_embd,
                                   block=block, proj=True)
        elif asama == 'ff':
            self.govde = nn.Sequential(
                MultiHead(n_head, n_embd // n_head, n_embd=n_embd, block=block,
                          proj=True),
                FeedForward(n_embd),
            )
        elif asama in ('blok_duz', 'residual', 'ln'):
            res = asama in ('residual', 'ln')
            nrm = asama == 'ln'
            self.govde = nn.Sequential(*[
                Blok(n_embd, n_head, block=block, residual=res, norm=nrm,
                     dropout=dropout) for _ in range(n_blok)])
            # LayerNorm'lu surumde son bir normalizasyon daha (makalede de var)
            self.son_ln = nn.LayerNorm(n_embd) if nrm else None
        else:
            raise ValueError(asama)

        self.lm_head = nn.Linear(n_embd, v)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T))
        x = tok + pos
        x = self.govde(x)
        if getattr(self, 'son_ln', None) is not None:
            x = self.son_ln(x)
        logits = self.lm_head(x)
        return _kayip(logits, targets)


class GPTDilModeli(nn.Module):
    """Gorev 3: olceklenmis, dropout'lu, tam hali."""

    def __init__(self, n_embd=128, n_head=4, n_blok=4, block=64,
                 dropout=0.1, v=V):
        super().__init__()
        self.block = block
        self.token_embedding_table = nn.Embedding(v, n_embd)
        self.position_embedding_table = nn.Embedding(block, n_embd)
        self.bloklar = nn.Sequential(*[
            Blok(n_embd, n_head, block=block, residual=True, norm=True,
                 dropout=dropout) for _ in range(n_blok)])
        self.son_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, v)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        # Hafta 4'te ogrendigim sey: baslangic olcegi onemli.
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T))
        x = self.son_ln(self.bloklar(tok + pos))
        logits = self.lm_head(x)
        return _kayip(logits, targets)


# ---------------------------------------------------------------------
# ortak yardimcilar
# ---------------------------------------------------------------------
def _kayip(logits, targets):
    if targets is None:
        return logits, None
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
    return logits, loss


@torch.no_grad()
def uret(model, idx, max_new_tokens, g=None):
    """Metin uretir. idx'i son block karaktere kirpmak ZORUNLU:
    pozisyon tablosunda sadece block satir var (Hafta 7)."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1, generator=g)
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return idx


# bigram ve diger modellere ayni uretim fonksiyonunu bagla
for _k in (BigramDilModeli, AblasyonModeli, GPTDilModeli):
    _k.generate = lambda self, idx, n, g=None: uret(self, idx, n, g)
