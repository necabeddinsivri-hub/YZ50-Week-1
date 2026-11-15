# agirlik.py - OpenAI'in GPT-2 agirliklarini yerel dosyadan yukler
# YZ50 Hafta 12 (Hafta 10'daki dosyanin sadelestirilmis hali)
#
# Calistigim ortamdan huggingface.co'ya cikis kapali. model.safetensors
# dosyasini Hafta 10'da tarayicidan indirmistim (MD5
# 8b9634194d1029deff7964617b8bcc75); bu hafta ayni dosyayi kullaniyorum.
# Dosya 548 MB oldugu icin depoda YOK, veri/agirlik/ klasorune elle
# konmasi gerekiyor (README'de adresi var).

import os

KLASOR = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(KLASOR, 'veri', 'agirlik', 'model.safetensors')


def var_mi():
    return os.path.exists(ST)


def state_dict():
    """Agirliklari {isim: tensor} olarak dondurur."""
    from safetensors.torch import load_file
    sd = load_file(ST)
    # Dosyadaki anahtarlarin basinda "transformer." yok; benim modelimde var.
    if not any(k.startswith('transformer.') for k in sd):
        sd = {('transformer.' + k if not k.startswith('lm_head') else k): v
              for k, v in sd.items()}
    return sd


def openai_gpt2(flash=True):
    """Kendi GPT sinifima OpenAI'in agirliklarini yukleyip dondurur."""
    from gpt2 import GPT, GPTConfig
    model = GPT.from_pretrained(state_dict(), sessiz=True)
    if flash:
        for blok in model.transformer.h:
            blok.attn.flash = True
    model.eval()
    return model
