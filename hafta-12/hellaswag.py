# hellaswag.py - HellaSwag'i bir dil modelinde olcmek
# YZ50 Hafta 12  (video 3:28:23 - 3:56:21)
#
# HELLASWAG NE SORUYOR
# Her ornekte kisa bir baglam ("A man is sitting on a roof. he") ve dort
# olasi devam var. Biri dogru, ucu yaniltici. Yaniltici olanlar bir dil
# modeliyle uretilip insanlarin "sacma" bulmadiklari arasindan secilmis;
# yani dilbilgisi dogru ama anlam yanlis. Dort secenek oldugu icin
# rastgele tahmin %25 bulur.
#
# KUCUK BIR MODEL NASIL "CEVAP VERIR"
# Model soruyu okuyup "C" diyemez; o kadar akilli degil. Karpathy'nin
# yaptigi (ve benim de yaptigim) su: dort devami ayri ayri baglamin
# arkasina ekliyorum ve modele soruyorum, "bu devam ne kadar olasi?"
# Olasiligi en yuksek (loss'u en dusuk) devam, modelin cevabi.
# Buna "completion style" deniyor.
#
# IKI OLCU
#   acc      : devamdaki token'larin TOPLAM loss'u en dusuk olan
#   acc_norm : token basina ORTALAMA loss'u en dusuk olan
# Toplam loss uzun devamlari cezalandiriyor (her token loss ekliyor);
# ortalama bunu duzeltiyor. Karpathy acc_norm'u esas aliyor, ben de.
#
# VERI: Hugging Face'teki Rowan/hellaswag'in dogrulama seti (10.042
# ornek). Karpathy'nin kodundaki GitHub adresi artik 404 veriyor; ayni
# veri Hugging Face'te duruyor. Ilk ornek iki yerde de "ind 24".

import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from tokenizer import enc

import os
# HS_DIL=tr ile Turkce ceviri seti (malhajar/hellaswag-tr) kullaniliyor
DIL = os.environ.get('HS_DIL', 'en')
VERI = ('veri/ham/hellaswag_tr_validation.parquet' if DIL == 'tr'
        else 'veri/ham/hellaswag_validation.parquet')
_e = enc()


def ornekler(n=None):
    tablo = pq.read_table(VERI)
    satirlar = tablo.to_pylist()
    return satirlar if n is None else satirlar[:n]


def hazirla(ornek):
    """Bir ornegi (4, L) token matrisine ve (4, L) maskeye cevirir.

    Maske 1: o pozisyon DEVAMA ait (loss'u sayilacak).
    Maske 0: baglam ya da dolgu (sayilmayacak).
    """
    baglam = _e.encode(ornek['ctx'])
    satirlar, maskeler = [], []
    for son in ornek['endings']:
        # Basa bosluk: GPT-2 tokenizer'inda kelimeler genelde bosluklu
        # token'larla basliyor (" the"). Baglamin arkasina dogal sekilde
        # oturmasi icin devami bosluklu kodluyorum.
        devam = _e.encode(' ' + son)
        satirlar.append(baglam + devam)
        maskeler.append([0] * len(baglam) + [1] * len(devam))
    L = max(len(s) for s in satirlar)
    tok = torch.zeros((4, L), dtype=torch.long)
    msk = torch.zeros((4, L), dtype=torch.long)
    for i, (s, m) in enumerate(zip(satirlar, maskeler)):
        tok[i, :len(s)] = torch.tensor(s)
        msk[i, :len(m)] = torch.tensor(m)
    return tok, msk, int(ornek['label'])


@torch.no_grad()
def tahmin(model, tok, msk, bf16=True):
    """(acc tahmini, acc_norm tahmini, toplam loss'lar, ortalama loss'lar)."""
    if bf16:
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            logits, _ = model(tok)
    else:
        logits, _ = model(tok)
    logits = logits.float()
    # t konumundaki logits, t+1 konumundaki token'i tahmin ediyor
    kayik_logits = logits[:, :-1, :].contiguous()
    kayik_tok = tok[:, 1:].contiguous()
    kayik_msk = msk[:, 1:].contiguous()
    l = F.cross_entropy(kayik_logits.view(-1, kayik_logits.size(-1)),
                        kayik_tok.view(-1), reduction='none')
    l = l.view(tok.size(0), -1) * kayik_msk
    toplam = l.sum(dim=1)
    ortalama = toplam / kayik_msk.sum(dim=1)
    return (int(toplam.argmin()), int(ortalama.argmin()),
            toplam.tolist(), ortalama.tolist())
