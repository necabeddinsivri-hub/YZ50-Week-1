# agirlik.py - GPT-2 agirliklarini yerel dosyadan yukler
# YZ50 Hafta 10
#
# Normalde `GPT2LMHeadModel.from_pretrained("gpt2")` deyip gecilir ve
# kutuphane dosyalari Hugging Face'ten indirir. Benim calistirdigim
# ortamdan huggingface.co'ya cikis kapali (proxy 403). O yuzden
# model.safetensors ve config.json dosyalarini elle indirip veri/
# klasorune koydum; buradaki kod onlari yerelden okuyor.
#
# Dosyalar HF'teki openai-community/gpt2 deposundan, degistirilmeden.

import json
import os

import torch

KLASOR = os.path.dirname(os.path.abspath(__file__))
VERI = os.path.join(KLASOR, 'veri')
ST = os.path.join(VERI, 'model.safetensors')
BIN = os.path.join(VERI, 'pytorch_model.bin')
CFG = os.path.join(VERI, 'config.json')


def var_mi():
    return os.path.exists(ST) or os.path.exists(BIN)


def state_dict():
    """Agirliklari {isim: tensor} olarak dondurur."""
    if os.path.exists(ST):
        from safetensors.torch import load_file
        sd = load_file(ST)
    elif os.path.exists(BIN):
        sd = torch.load(BIN, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(
            'veri/model.safetensors ya da veri/pytorch_model.bin yok')
    # HF'in eski surumlerinde basta "transformer." on eki olmayabiliyor;
    # benim modelimde var, uyumlu hale getiriyorum.
    if not any(k.startswith('transformer.') for k in sd):
        sd = {('transformer.' + k if not k.startswith('lm_head') else k): v
              for k, v in sd.items()}
    return sd


def config():
    return json.load(open(CFG)) if os.path.exists(CFG) else None


def hf_model():
    """Karsilastirma icin Hugging Face'in kendi modelini yerelden kurar."""
    from transformers import GPT2LMHeadModel, GPT2Config
    c = config()
    if c is None:
        raise FileNotFoundError('veri/config.json yok')
    m = GPT2LMHeadModel(GPT2Config(**{k: v for k, v in c.items()
                                      if k != 'architectures'}))
    sd = state_dict()
    # HF modelinin kendi anahtar adlari; buffer'lari atliyorum
    hedef = m.state_dict()
    yuklenen = 0
    for k, v in sd.items():
        if k in hedef and hedef[k].shape == v.shape:
            with torch.no_grad():
                hedef[k].copy_(v)
            yuklenen += 1
    m.load_state_dict(hedef, strict=False)
    m.eval()
    return m, yuklenen


if __name__ == '__main__':
    print('agirlik dosyasi var mi:', var_mi())
    if var_mi():
        sd = state_dict()
        print(f'tensor sayisi: {len(sd)}')
        print(f'toplam parametre: {sum(v.numel() for v in sd.values()):,}')
