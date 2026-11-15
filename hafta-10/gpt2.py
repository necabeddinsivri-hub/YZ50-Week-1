# gpt2.py - GPT-2 (124M) sifirdan
# YZ50 Hafta 10
#
# Hafta 8'de yazdigim GPT ile arada BUYUK bir mimari fark yok: ayni
# attention, ayni blok, ayni residual. Fark olcekte ve birkac ayrintida.
# Ayrintilar onemli, cunku OpenAI'in egittigi agirliklari yukleyebilmem
# icin kodumun onlarinkiyle BIREBIR ayni sekilde dizilmis olmasi gerekiyor.
#
# Hafta 8'e gore degisenler (Gorev 2'de tek tek sayiyorum):
#   - isimler: transformer.wte / wpe / h / ln_f, lm_head
#   - q, k, v TEK bir Linear'da (c_attn), sonra uce bolunuyor
#   - ReLU yerine GELU (tanh yaklasimiyla)
#   - LayerNorm'larin adi ln_1 / ln_2
#   - lm_head'de bias YOK
#   - weight tying: wte ile lm_head ayni tensoru paylasiyor

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024      # baglam uzunlugu (Hafta 8'de 64'tu)
    vocab_size: int = 50257     # 50.000 BPE birlestirmesi + 256 byte + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    """Hafta 9'da (Gorev 5a) yazdigim tek-matmul'lu multi-head'in aynisi.

    Orada dongulu surumle bire bir ayni sonucu verdigini gostermistim.
    GPT-2'nin kodu da boyle: tek bir Linear uc ciktiyi birden uretiyor.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # tek katman, 3*n_embd cikti: key | query | value yan yana
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1      # Gorev 5a'da kullaniyorum
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        # DIKKAT: sira q, k, v. HF'in kodunda da boyle; degistirirsem
        # yuklenen agirliklar yanlis yere oturur.
        q, k, v = qkv.split(self.n_embd, dim=2)
        hs = C // self.n_head
        # (B,T,C) -> (B, n_head, T, hs): head ekseni one aliniyor ki
        # matmul son iki eksende calissin, head'ler batch gibi islensin
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """Hafta 8'deki FeedForward. Tek fark: ReLU yerine GELU."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    """Hafta 8'deki Blok ile bire bir ayni: pre-norm + residual."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, weight_tying=False, ozel_init=False):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if weight_tying:
            # Giris embedding'i ile cikis katmani AYNI tensor.
            # Gorev 5a'da kac parametre kazandirdigini hesapliyorum.
            self.transformer.wte.weight = self.lm_head.weight

        if ozel_init:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """GPT-2'nin kendi baslatma semasi (videodaki gibi)."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # Residual yoluna EKLENEN her katman varyansi buyutuyor.
                # Blok basina iki ekleme var (attn + mlp), yani 2*n_layer
                # terim toplaniyor. Varyansin sabit kalmasi icin
                # 1/sqrt(2*n_layer) ile olcekliyorum.
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f'dizi uzunlugu {T}, block_size {self.config.block_size}'
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    # -----------------------------------------------------------------
    # HAZIR AGIRLIKLARI YUKLEME
    # -----------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, sd_hf, config=None, sessiz=False):
        """Hugging Face'in state_dict'ini kendi modelime yukler.

        sd_hf: {isim: tensor} sozlugu (model.safetensors ya da
        pytorch_model.bin'den okunmus).
        """
        config = config or GPTConfig()
        model = cls(config, weight_tying=True)
        sd = model.state_dict()
        # attention maskesi ogrenilen bir parametre degil, sabit bir buffer
        anahtarlar = [k for k in sd if not k.endswith('.attn.bias')]

        # HF'in dosyasinda da maske ve ters maske duruyor; onlari atiyorum
        hf_anahtarlar = [k for k in sd_hf
                         if not k.endswith('.attn.masked_bias')
                         and not k.endswith('.attn.bias')]

        # ---- TRANSPOZE MESELESI ----
        # OpenAI'in orijinal GPT-2 kodu TensorFlow'da yazilmisti ve lineer
        # katmanlar icin nn.Linear yerine "Conv1D" adinda bir sinif
        # kullaniyor. O sinif agirligi (giris, cikis) seklinde tutuyor ve
        # x @ W yapiyor. PyTorch'un nn.Linear'i ise (cikis, giris) seklinde
        # tutup x @ W.T yapiyor. Yani ayni matris, DEVRIK saklanmis.
        # Asagidaki dort katman o sinifla yazilmis; onlari transpoze
        # etmeden yuklersem boyutlar tutmaz (ya da daha kotusu, kare
        # matrislerde boyut tutar ama model sacmalar).
        transpoze = ['attn.c_attn.weight', 'attn.c_proj.weight',
                     'mlp.c_fc.weight', 'mlp.c_proj.weight']

        t_sayi, d_sayi = 0, 0
        for k in hf_anahtarlar:
            if k not in sd:
                raise KeyError('modelimde olmayan anahtar: ' + k)
            if any(k.endswith(w) for w in transpoze):
                assert sd_hf[k].shape[::-1] == sd[k].shape, (k, sd_hf[k].shape)
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
                t_sayi += 1
            else:
                assert sd_hf[k].shape == sd[k].shape, (k, sd_hf[k].shape)
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                d_sayi += 1
        if not sessiz:
            print(f"  yuklendi: {d_sayi} dogrudan, {t_sayi} transpoze edilerek")
            eksik = set(anahtarlar) - set(hf_anahtarlar)
            if eksik:
                print(f"  HF'de olmayan (weight tying ile paylasilan): {sorted(eksik)}")
        return model


def parametre_tablosu(sd):
    """Agirlik adlarini, boyutlarini ve toplam parametreyi dondurur."""
    satirlar = []
    toplam = 0
    for k, v in sd.items():
        n = v.numel()
        satirlar.append((k, tuple(v.shape), n))
        toplam += n
    return satirlar, toplam
