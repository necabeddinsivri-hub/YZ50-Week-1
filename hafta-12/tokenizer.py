# tokenizer.py - GPT-2'nin tokenizer'ini CEVRIMDISI yukler
# YZ50 Hafta 10
#
# Hafta 9'da ogrendigim sey: tiktoken birlestirme tablosunu OpenAI'in
# sunucusundan indiriyor ve calistirdigim ortamdan o adrese cikis kapali.
# Gecen hafta cl100k_base icin yaptigim seyin aynisini GPT-2'nin
# tablosu icin de yaptim: tabloyu npm'deki js-tiktoken paketinden alip
# veri/gpt2.tiktoken dosyasina koydum, tiktoken'a elle veriyorum.
#
# GPT-2'nin vocab'i: 50.000 BPE birlestirmesi + 256 taban byte + 1 ozel
# token (<|endoftext|>) = 50.257.

import base64
import json
import os

import tiktoken

KLASOR = os.path.dirname(os.path.abspath(__file__))
RANKS = os.path.join(KLASOR, 'veri', 'gpt2.tiktoken')
META = os.path.join(KLASOR, 'veri', 'gpt2_meta.json')


def mergeable_ranks():
    r = {}
    with open(RANKS) as f:
        for satir in f:
            t, n = satir.rstrip('\n').split(' ')
            r[base64.b64decode(t)] = int(n)
    return r


def enc():
    meta = json.load(open(META))
    return tiktoken.Encoding(
        name='gpt2',
        pat_str=meta['pat_str'],
        mergeable_ranks=mergeable_ranks(),
        special_tokens=meta['special_tokens'],
    )


if __name__ == '__main__':
    e = enc()
    print(f"gpt2 tokenizer: n_vocab = {e.n_vocab:,}".replace(',', '.'))
    for s in ['Hello, I am a language model,',
              'Merhaba, ben bir dil modeliyim,']:
        ids = e.encode(s)
        print(f"  {s!r}")
        print(f"    {len(ids)} token: {[e.decode([i]) for i in ids]}")
