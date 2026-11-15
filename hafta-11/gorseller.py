# gorseller.py - rapor ve sunum icin grafikler
# YZ50 Hafta 11
# Butun sayilar sonuc_gorev*.json dosyalarindan geliyor; burada
# hicbir sayi elle yazilmiyor.

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE = '#00285A'; BLUE2 = '#0F6DB5'; GREY = '#63697A'
LITE = '#EEF2F7'; RED = '#A3242B'; GREEN = '#1F6F54'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#C8CCD6',
                     'axes.labelcolor': BLUE, 'text.color': BLUE,
                     'xtick.color': GREY, 'ytick.color': GREY})

J = lambda n: (json.load(open(f'sonuc_gorev{n}.json'))
               if os.path.exists(f'sonuc_gorev{n}.json') else None)
G1, G2, G3, G4 = J(1), J(2), J(3), J(4)


def bas(ax, t):
    ax.set_title(t, color=BLUE, fontsize=10, loc='left')


# =====================================================================
# 1) RESIDUAL AKISININ STD'SI  (Gorev 1d)
# =====================================================================
def akis():
    if not G1 or 'akis' not in G1:
        return
    a = G1['akis']
    n = len(a['olcekli'])
    x = np.arange(n)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.3))

    a1.plot(x, a['olceksiz'], 'o-', color=RED, lw=1.8, ms=4,
            label='olcekleme YOK')
    a1.plot(x, a['olcekli'], 'o-', color=BLUE, lw=1.8, ms=4,
            label='1/sqrt(2*n_layer) ile')
    bas(a1, 'Residual akisinin std\'si, blok blok')
    a1.set_xlabel('blok'); a1.set_ylabel('std')
    a1.legend(frameon=False, fontsize=8)
    a1.grid(alpha=.25, lw=.5)

    oran = [b / c for b, c in zip(a['olceksiz'], a['olcekli'])]
    a2.plot(x, oran, 'o-', color=BLUE2, lw=1.8, ms=4)
    a2.axhline(a['teorik'], color=RED, ls='--', lw=1.2)
    a2.annotate(f"sqrt(2*n_layer) = {a['teorik']:.2f}",
                (n * 0.25, a['teorik']), textcoords='offset points',
                xytext=(0, 7), fontsize=8, color=RED)
    bas(a2, 'Olceksiz / olcekli orani -> teoriye oturuyor')
    a2.set_xlabel('blok'); a2.set_ylabel('oran')
    a2.set_ylim(0, max(oran) * 1.25)
    a2.grid(alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig('akis.png', dpi=200)
    plt.close(fig)
    print('  akis.png')


# =====================================================================
# 2) HIZLANDIRMA TABLOSU  (Gorev 2)
# =====================================================================
def hiz():
    if not G2 or not G2['satirlar']:
        return
    s = G2['satirlar']
    adlar = [r['ad'] for r in s]
    med = [r['medyan'] for r in s]
    tok = [r['tok_sn'] for r in s]
    taban = med[0]
    y = np.arange(len(s))[::-1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                 gridspec_kw=dict(width_ratios=[1.35, 1]))
    renk = [BLUE if m <= taban else RED for m in med]
    a1.barh(y, med, color=renk, height=.6)
    for yy, m in zip(y, med):
        a1.text(m + taban * .02, yy, f'{m:.2f}s', va='center',
                fontsize=8, color=BLUE)
    a1.axvline(taban, color=GREY, ls=':', lw=1)
    a1.set_yticks(y); a1.set_yticklabels(adlar, fontsize=7.5)
    a1.set_xlim(0, max(med) * 1.18)
    bas(a1, 'Adim suresi (medyan, 1024 token/adim)')
    a1.set_xlabel('saniye')

    a2.barh(y, tok, color=BLUE2, height=.6)
    for yy, t in zip(y, tok):
        a2.text(t + max(tok) * .02, yy, f'{t:.0f}', va='center',
                fontsize=8, color=BLUE)
    a2.set_yticks(y); a2.set_yticklabels([])
    a2.set_xlim(0, max(tok) * 1.2)
    bas(a2, 'Islenen token / saniye')
    a2.set_xlabel('token/sn')
    fig.tight_layout(); fig.savefig('hiz.png', dpi=200)
    plt.close(fig)
    print('  hiz.png')


# =====================================================================
# 3) LEARNING RATE EGRISI  (Gorev 3b)
# =====================================================================
def lr():
    if not G3 or 'lr' not in G3:
        return
    d = G3['lr']
    e = d['egri']
    x = np.arange(len(e))
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.plot(x, e, color=BLUE, lw=2)
    ax.axvspan(0, d['warmup'] - 1, color=LITE)
    ax.axvline(d['warmup'] - 1, color=GREY, ls=':', lw=1)
    ax.axvline(d['max_adim'], color=GREY, ls=':', lw=1)
    ax.axhline(d['max_lr'], color=RED, ls='--', lw=1)
    ax.axhline(d['min_lr'], color=GREEN, ls='--', lw=1)
    ax.annotate(f"max_lr = {d['max_lr']:.1e}", (len(e)*.55, d['max_lr']),
                textcoords='offset points', xytext=(0, 5), fontsize=8,
                color=RED)
    ax.annotate(f"min_lr = {d['min_lr']:.1e} (tepenin %10'u)",
                (len(e)*.05, d['min_lr']), textcoords='offset points',
                xytext=(0, 6), fontsize=8, color=GREEN)
    ax.annotate(f"warmup\n{d['warmup']} adim", (d['warmup']/2.6, d['max_lr']*.45),
                fontsize=8, color=GREY, ha='center')
    ax.annotate('cosine inis', ((d['warmup']+d['max_adim'])/2,
                                d['max_lr']*.62), fontsize=8, color=BLUE2,
                ha='center')
    ax.annotate('sabit min', (d['max_adim']+2, d['min_lr']*1.5),
                fontsize=8, color=GREY, ha='center')
    bas(ax, 'Warmup + cosine learning rate programi')
    ax.set_xlabel('optimizer adimi'); ax.set_ylabel('learning rate')
    ax.set_ylim(0, d['max_lr'] * 1.22)
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig('lr_egrisi.png', dpi=200)
    plt.close(fig)
    print('  lr_egrisi.png')


# =====================================================================
# 4) EGITIM: LOSS, GRAD NORM, LR  (Gorev 3d)
# =====================================================================
def egitim():
    if not G3 or 'egitim' not in G3:
        return
    k = G3['egitim']['kayit']
    if not k:
        return
    x = [r['adim'] for r in k]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.4, 3.1))

    a1.plot(x, [r['loss'] for r in k], color=BLUE, lw=1.8)
    tb = G3['egitim'].get('taban')
    if tb:
        a1.axhline(tb, color=RED, ls='--', lw=1.1)
        a1.annotate(f'ln(V) = {tb:.2f}', (max(x)*.35, tb),
                    textcoords='offset points', xytext=(0, 5), fontsize=8,
                    color=RED)
    bas(a1, 'Loss'); a1.set_xlabel('adim'); a1.grid(alpha=.25, lw=.5)

    n = [r['norm'] for r in k]
    a2.plot(x, n, color=BLUE2, lw=1.5)
    a2.axhline(1.0, color=RED, ls='--', lw=1.2)
    a2.annotate('kirpma esigi = 1.0', (max(x)*.3, 1.0),
                textcoords='offset points', xytext=(0, 5), fontsize=8,
                color=RED)
    bas(a2, 'Gradient norm'); a2.set_xlabel('adim')
    a2.grid(alpha=.25, lw=.5)

    a3.plot(x, [r['lr'] for r in k], color=GREEN, lw=1.8)
    bas(a3, 'Learning rate (fiilen uygulanan)')
    a3.set_xlabel('adim'); a3.grid(alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig('egitim.png', dpi=200)
    plt.close(fig)
    print('  egitim.png')


# =====================================================================
# 5) IKI TOKENIZER  (Gorev 4)
# =====================================================================
def tokenizerlar():
    if not G4 or 'gpt2' not in G4 or 'benim' not in G4:
        return
    a, b = G4['gpt2'], G4['benim']
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.4, 3.2))
    etiket = ['GPT-2\nV=50257', 'Hafta 9\nV=1024']
    renk = [BLUE, BLUE2]

    v = [a['karakter'] / 1000, b['karakter'] / 1000]
    a1.bar(etiket, v, color=renk, width=.55)
    for i, t in enumerate(v):
        a1.text(i, t * 1.02, f'{t:.0f}k', ha='center', fontsize=9,
                color=BLUE)
    bas(a1, f"Esit surede ({G4['karsilastirma']['butce']:.0f} sn) gorulen metin")
    a1.set_ylabel('bin karakter'); a1.set_ylim(0, max(v) * 1.2)

    v2 = [a['token_sn'], b['token_sn']]
    a2.bar(etiket, v2, color=renk, width=.55)
    for i, t in enumerate(v2):
        a2.text(i, t * 1.02, f'{t:.0f}', ha='center', fontsize=9, color=BLUE)
    bas(a2, 'Islenen token / saniye')
    a2.set_ylim(0, max(v2) * 1.2)

    v3 = [a['bpc'], b['bpc']]
    a3.bar(etiket, v3, color=renk, width=.55)
    for i, t in enumerate(v3):
        a3.text(i, t * 1.02, f'{t:.2f}', ha='center', fontsize=9, color=BLUE)
    bas(a3, 'ORTAK OLCEK: bit / karakter (az iyi)')
    a3.set_ylim(0, max(v3) * 1.25)
    fig.tight_layout(); fig.savefig('tokenizer.png', dpi=200)
    plt.close(fig)
    print('  tokenizer.png')


# =====================================================================
# 6) ILK EGITIM EGRISI  (Gorev 1e)
# =====================================================================
def ilk_egitim():
    if not G1 or 'egitim' not in G1:
        return
    e = G1['egitim']['egri']
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ax.plot([r['adim'] for r in e], [r['loss'] for r in e],
            color=BLUE, lw=1.8)
    ax.axhline(G1['egitim']['taban'], color=RED, ls='--', lw=1.1)
    ax.annotate(f"ln(50257) = {G1['egitim']['taban']:.2f}",
                (len(e)*.4, G1['egitim']['taban']),
                textcoords='offset points', xytext=(0, 5), fontsize=8,
                color=RED)
    bas(ax, 'Data loader ile metnin tamaminda egitim (Gorev 1e)')
    ax.set_xlabel('adim'); ax.set_ylabel('loss')
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig('ilk_egitim.png', dpi=200)
    plt.close(fig)
    print('  ilk_egitim.png')


if __name__ == '__main__':
    print('grafikler:')
    akis(); hiz(); lr(); egitim(); tokenizerlar(); ilk_egitim()
    print('bitti')
