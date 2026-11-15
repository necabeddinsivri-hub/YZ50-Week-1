# gorseller.py - rapor ve sunum icin grafikler
# YZ50 Hafta 8
# Butun sayilar sonuclar.json'dan, yani GERCEK kosulardan geliyor.

import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE = '#00285A'
BLUE2 = '#0F6DB5'
GREY = '#63697A'
LITE = '#EEF2F7'
RED = '#A3242B'

s = json.load(open('sonuclar.json'))
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#C8CCD6',
                     'axes.labelcolor': BLUE, 'text.color': BLUE,
                     'xtick.color': GREY, 'ytick.color': GREY})


# =====================================================================
# 1) ABLASYON MERDIVENI
# =====================================================================
def ablasyon():
    sira = ['1_bigram', '2_tek_head', '3_multi_head', '4_feedforward',
            '5_residual', '6_layernorm']
    etiket = ['bigram', 'tek head', 'multi-head\n(4x8)', '+ feedforward',
              '+ residual\n(3 blok)', '+ LayerNorm']
    val = [s[k]['val'] for k in sira]
    tr = [s[k]['train'] for k in sira]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(val))
    ax.plot(x, tr, 'o--', color=GREY, lw=1.4, ms=5, label='train')
    ax.plot(x, val, 'o-', color=BLUE, lw=2.2, ms=7, label='val')
    for i, v in enumerate(val):
        ax.annotate(f"{v:.4f}", (i, v), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8,
                    color=BLUE, fontweight='bold')
        if i:
            d = val[i - 1] - v
            ax.annotate(f"{d:+.3f}", (i - 0.5, (val[i - 1] + v) / 2),
                        ha='center', fontsize=7.5,
                        color=BLUE2 if d > 0 else RED)
    # teshis noktasi: residual'siz 3 blok
    if '4b_blok_duz' in s:
        ax.plot([4], [s['4b_blok_duz']['val']], 'X', color=RED, ms=10,
                label='3 blok, residual YOK')
        ax.annotate(f"{s['4b_blok_duz']['val']:.4f}",
                    (4, s['4b_blok_duz']['val']),
                    textcoords='offset points', xytext=(26, -2),
                    fontsize=8, color=RED, fontweight='bold')
    ax.set_xticks(x, etiket, fontsize=8)
    ax.set_ylabel('loss')
    ax.set_title('Exhibit — Transformer blogu parca parca eklenince val loss',
                 color=BLUE, fontsize=10, loc='left')
    ax.grid(axis='y', color=LITE, lw=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig('ablasyon.png', dpi=130)
    plt.close()
    print("  ablasyon.png")


# =====================================================================
# 2) TEK HEAD vs MULTI-HEAD (parametre vs kazanc)
# =====================================================================
def multihead():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    adlar = ['tek head\n(1 x 32)', 'multi-head\n(4 x 8)']
    val = [s['2_tek_head']['val'], s['3_multi_head']['val']]
    par = [s['2_tek_head']['param'], s['3_multi_head']['param']]

    b = a1.bar(adlar, val, color=[GREY, BLUE], width=0.52)
    a1.bar_label(b, fmt='%.4f', fontsize=8.5, color=BLUE, padding=2)
    a1.set_ylim(min(val) - 0.04, max(val) + 0.03)
    a1.set_ylabel('val loss')
    a1.set_title(f"val loss  ({val[0] - val[1]:+.4f})", fontsize=9, loc='left')
    a1.grid(axis='y', color=LITE, lw=1); a1.set_axisbelow(True)

    b2 = a2.bar(adlar, par, color=[GREY, BLUE2], width=0.52)
    a2.bar_label(b2, fmt='%d', fontsize=8.5, color=BLUE, padding=2)
    a2.set_ylabel('parametre')
    a2.set_title(f"parametre  (+{par[1] - par[0]})", fontsize=9, loc='left')
    a2.grid(axis='y', color=LITE, lw=1); a2.set_axisbelow(True)
    fig.suptitle('Exhibit — ayni boyut, bolunmus dikkat', color=BLUE,
                 fontsize=10, x=0.012, ha='left')
    plt.tight_layout()
    plt.savefig('multihead.png', dpi=130)
    plt.close()
    print("  multihead.png")


# =====================================================================
# 3) 4 HEAD'IN ISI HARITALARI
# =====================================================================
def headler():
    try:
        w = np.load('head_agirliklari.npy')      # (4, 8, 8)
    except FileNotFoundError:
        print("  (head_agirliklari.npy yok, atlandi)")
        return
    cumle = "First Ci"
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 3.0))
    for h, ax in enumerate(axes):
        ax.imshow(w[h], cmap='Blues', vmin=0, vmax=w[h].max())
        ax.set_xticks(range(8), list(cumle), fontsize=7)
        ax.set_yticks(range(8), list(cumle), fontsize=7)
        ax.set_title(f"head {h}", fontsize=9, color=BLUE)
        for i in range(8):
            for j in range(i + 1):
                ax.text(j, i, f"{w[h][i, j]:.2f}", ha='center', va='center',
                        fontsize=5.2,
                        color='white' if w[h][i, j] > w[h].max() * .6 else BLUE)
    fig.suptitle("Exhibit — 4 head, 'First Ci' cumlesinde kim kime bakiyor",
                 color=BLUE, fontsize=10, x=0.012, ha='left')
    plt.tight_layout()
    plt.savefig('head_haritalari.png', dpi=130)
    plt.close()
    print("  head_haritalari.png")


# =====================================================================
# 4) LAYERNORM vs BATCHNORM EKSEN SEMASI
# =====================================================================
def eksen():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.5))
    rng = np.random.default_rng(0)
    m = rng.normal(size=(5, 7))
    for ax, baslik, kip in [(a1, 'BatchNorm1d  —  SUTUN (dim=0)', 'bn'),
                            (a2, 'LayerNorm  —  SATIR (dim=1)', 'ln')]:
        ax.imshow(m, cmap='Greys', alpha=.25, aspect='auto')
        for i in range(5):
            for j in range(7):
                ax.text(j, i, f"{m[i, j]:.1f}", ha='center', va='center',
                        fontsize=6.5, color='#555')
        if kip == 'bn':
            for j in [2]:
                ax.add_patch(plt.Rectangle((j - .5, -.5), 1, 5, fill=False,
                                           ec=BLUE, lw=2.4))
            ax.set_xlabel('ozellikler (C)  ->  her sutun ayri normalize')
            ax.set_ylabel('ornekler (B)')
        else:
            for i in [2]:
                ax.add_patch(plt.Rectangle((-.5, i - .5), 7, 1, fill=False,
                                           ec=BLUE, lw=2.4))
            ax.set_xlabel('ozellikler (C)')
            ax.set_ylabel('ornekler (B)  ->  her satir ayri normalize')
        ax.set_title(baslik, fontsize=9, color=BLUE, loc='left')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Exhibit — tek karakterlik kod farki: mean(0) / mean(1)',
                 color=BLUE, fontsize=10, x=0.012, ha='left')
    plt.tight_layout()
    plt.savefig('eksen.png', dpi=130)
    plt.close()
    print("  eksen.png")


# =====================================================================
# 5) OLCEKLEME: parametre - val loss - sure
# =====================================================================
def olcek():
    kayit = [('kucuk\n(32d, 3 blok,\nblock 8)', '6_layernorm'),
             ('olcekli\ndropout YOK', '7_olcekli_dropoutsuz'),
             ('olcekli\ndropout 0.1', '8_olcekli_dropout')]
    kayit = [(a, k) for a, k in kayit if k in s]
    if len(kayit) < 2:
        print("  (olcekleme verisi yok, atlandi)")
        return
    ad = [a for a, _ in kayit]
    par = [s[k]['param'] for _, k in kayit]
    val = [s[k]['val'] for _, k in kayit]
    trn = [s[k]['train'] for _, k in kayit]
    sure = [s[k]['sure'] / 60 for _, k in kayit]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    for ax, veri_, baslik, fmt, renk in [
            (axes[0], par, 'parametre', '%d', BLUE2),
            (axes[1], None, 'loss', None, None),
            (axes[2], sure, 'egitim suresi (dk)', '%.1f', GREY)]:
        if baslik == 'loss':
            x = np.arange(len(ad))
            b1 = ax.bar(x - .18, trn, .34, color=GREY, label='train')
            b2 = ax.bar(x + .18, val, .34, color=BLUE, label='val')
            ax.bar_label(b1, fmt='%.3f', fontsize=7, padding=1, color=GREY)
            ax.bar_label(b2, fmt='%.3f', fontsize=7, padding=1, color=BLUE)
            ax.set_xticks(x, ad, fontsize=7)
            ax.legend(frameon=False, fontsize=7.5)
            ax.set_ylim(0, max(val) * 1.25)
        else:
            b = ax.bar(ad, veri_, color=renk, width=.5)
            ax.bar_label(b, fmt=fmt, fontsize=7.5, padding=2, color=BLUE)
            ax.set_xticklabels(ad, fontsize=7)
            ax.set_ylim(0, max(veri_) * 1.2)
        ax.set_title(baslik, fontsize=9, color=BLUE, loc='left')
        ax.grid(axis='y', color=LITE, lw=1); ax.set_axisbelow(True)
    fig.suptitle('Exhibit — olcekleme: ne kazandim, ne odedim', color=BLUE,
                 fontsize=10, x=0.012, ha='left')
    plt.tight_layout()
    plt.savefig('olcekleme.png', dpi=130)
    plt.close()
    print("  olcekleme.png")


# =====================================================================
# 6) TURKCE vs SHAKESPEARE: loss'un karsilastirilamazligi
# =====================================================================
def turkce():
    if '9_turkce' not in s:
        print("  (turkce verisi yok, atlandi)")
        return
    sh_V, tr_V = 65, s['9_turkce']['V']
    sh_val = s['8_olcekli_dropout']['val']
    tr_val = s['9_turkce']['val']
    taban = [math.log(sh_V), math.log(tr_V)]
    val = [sh_val, tr_val]
    ad = [f'Shakespeare\nV={sh_V}', f'Turkce\nV={tr_V}']

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    x = np.arange(2)
    a1.bar(x, taban, .46, color=LITE, edgecolor=GREY,
           label='log(V) = egitimsiz taban')
    b = a1.bar(x, val, .46, color=BLUE, label='val loss')
    a1.bar_label(b, fmt='%.3f', fontsize=8, padding=2, color='white',
                 label_type='center')
    for i in range(2):
        a1.annotate(f"log(V)={taban[i]:.3f}", (i, taban[i]),
                    textcoords='offset points', xytext=(0, 4), ha='center',
                    fontsize=7.5, color=GREY)
        a1.annotate(f"kazanc\n{taban[i] - val[i]:.3f}",
                    (i, (taban[i] + val[i]) / 2), ha='center', fontsize=7.5,
                    color=BLUE2)
    a1.set_xticks(x, ad, fontsize=8)
    a1.set_title('val loss, egitimsiz tabana gore', fontsize=9, loc='left')
    a1.legend(frameon=False, fontsize=7.5, loc='upper left')
    a1.grid(axis='y', color=LITE, lw=1); a1.set_axisbelow(True)

    oran = [val[i] / taban[i] for i in range(2)]
    b2 = a2.bar(ad, oran, .46, color=[GREY, BLUE])
    a2.bar_label(b2, fmt='%.3f', fontsize=8.5, padding=2, color=BLUE)
    a2.set_ylim(0, max(oran) * 1.3)
    a2.set_title('val / log(V)  —  kucuk olan iyi', fontsize=9, loc='left')
    a2.set_xticklabels(ad, fontsize=8)
    a2.grid(axis='y', color=LITE, lw=1); a2.set_axisbelow(True)
    fig.suptitle('Exhibit — iki loss neden dogrudan karsilastirilamaz',
                 color=BLUE, fontsize=10, x=0.012, ha='left')
    plt.tight_layout()
    plt.savefig('turkce_vs_sh.png', dpi=130)
    plt.close()
    print("  turkce_vs_sh.png")


# =====================================================================
# 7) BLOK SEMASI
# =====================================================================
def blok_semasi():
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')

    def kutu(y, h, t, renk=LITE, kalin=False):
        ax.add_patch(plt.Rectangle((2.6, y), 4.8, h, fc=renk, ec=BLUE,
                                   lw=1.6 if kalin else 1.1))
        ax.text(5.0, y + h / 2, t, ha='center', va='center', fontsize=8.5,
                color=BLUE, fontweight='bold' if kalin else 'normal')

    kutu(0.3, 0.9, 'token emb + pos emb')
    kutu(2.0, 0.8, 'LayerNorm')
    kutu(3.1, 1.0, 'Multi-Head Attention (4 head)\nILETISIM', '#DCE8F5', True)
    kutu(4.7, 0.8, 'LayerNorm')
    kutu(5.8, 1.0, 'FeedForward  32->128->32\nDUSUNME', '#DCE8F5', True)
    kutu(7.6, 0.9, 'x3 blok')
    kutu(9.0, 0.8, 'son LayerNorm')
    kutu(10.2, 0.9, 'Linear -> V skor')

    for y0, y1 in [(1.2, 2.0), (2.8, 3.1), (4.1, 4.7), (5.5, 5.8),
                   (6.8, 7.6), (8.5, 9.0), (9.8, 10.2)]:
        ax.annotate('', (5.0, y1), (5.0, y0),
                    arrowprops=dict(arrowstyle='->', color=GREY, lw=1.2))

    # residual oklari
    for ya, yb_, et in [(1.6, 4.4, 'residual'), (4.4, 7.2, 'residual')]:
        ax.annotate('', (7.9, yb_), (7.9, ya),
                    arrowprops=dict(arrowstyle='->', color=BLUE2, lw=1.8,
                                    connectionstyle='arc3,rad=-0.35'))
        ax.text(9.0, (ya + yb_) / 2, et, fontsize=7.5, color=BLUE2,
                rotation=90, va='center')
    ax.text(0.1, 11.4, 'Exhibit — kurdugum blok (pre-norm)', fontsize=10,
            color=BLUE, fontweight='bold')
    ax.text(0.1, 0.0, 'residual = x + alt_katman(x);  toplama dugumunun '
            'gradyani 1, yani gradyan otoyoldan akiyor',
            fontsize=7.5, color=GREY)
    plt.tight_layout()
    plt.savefig('blok_semasi.png', dpi=130)
    plt.close()
    print("  blok_semasi.png")


if __name__ == '__main__':
    print("Grafikler uretiliyor:")
    ablasyon(); multihead(); headler(); eksen(); olcek(); turkce(); blok_semasi()
    print("Bitti.")
