# gpt4_yukle.py - GPT-4'un (cl100k_base) tokenizer'ini CEVRIMDISI yukler
# YZ50 Hafta 9
#
# Normalde tiktoken.get_encoding("cl100k_base") diyorsun ve kutuphane
# birlestirme tablosunu OpenAI'in sunucusundan indiriyor. Benim kodu
# calistirdigim ortamda o adrese cikis kapali. Cozum: ayni tabloyu
# npm'deki js-tiktoken paketinden aldim (veri/cl100k_base.tiktoken) ve
# tiktoken'a elle veriyorum. Tablo bire bir ayni, sadece kaynagi farkli.
#
# Dosya formati: her satirda "base64_token rank".

import base64
import json
import os

import tiktoken

KLASOR = os.path.dirname(os.path.abspath(__file__))
RANKS = os.path.join(KLASOR, 'veri', 'cl100k_base.tiktoken')
META = os.path.join(KLASOR, 'veri', 'cl100k_meta.json')


def mergeable_ranks():
    ranks = {}
    with open(RANKS) as f:
        for satir in f:
            t, r = satir.rstrip('\n').split(' ')
            ranks[base64.b64decode(t)] = int(r)
    return ranks


def encoding():
    meta = json.load(open(META))
    return tiktoken.Encoding(
        name='cl100k_base',
        pat_str=meta['pat_str'],
        mergeable_ranks=mergeable_ranks(),
        special_tokens=meta['special_tokens'],
    )


if __name__ == '__main__':
    enc = encoding()
    r = mergeable_ranks()
    print(f"cl100k_base yuklendi: {len(r):,} birlestirilebilir token"
          .replace(',', '.'))
    print(f"n_vocab (ozel tokenlar dahil): {enc.n_vocab:,}".replace(',', '.'))
    for s in ['hello world', "İstanbul'da yağmur yağıyor"]:
        ids = enc.encode(s)
        print(f"  {s!r} -> {len(ids)} token: "
              f"{[enc.decode([i]) for i in ids]}")
