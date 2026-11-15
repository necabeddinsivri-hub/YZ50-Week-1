# gorseller.py - Hafta 12 grafikleri
# Butun sayilar sonuc_*.json dosyalarindan; elle yazilmis sayi yok
# (iki istisna referans: Karpathy'nin 10 milyar token'lik kosusu ve
# GPT-3 makalesinin 300 milyar token'i - kaynaklari grafik basliginda).

import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE = '#00285A'; BLUE2 = '#0F6DB5'; GREY = '#63697A'
LITE = '#EEF2F7'; RED = '#A3242B'; GREEN = '#1F6F54'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#C8CCD6',
                     'axes.labelcolor': BLUE, 'text.color': BLUE,
                     'xtick.color': GREY, 'ytick.color': GREY})
J = lambda f: json.load(open(f)) if os.path.exists(f) else None
G1, G23, G4 = J('sonuc_gorev1.json'), J('sonuc_gorev23.json'), J('sonuc_gorev4.json')
G5 = J('sonuc_gorev5.json')
HB = J('sonuc_hellaswag_benim_1000.json')
HO = J('sonuc_hellaswag_openai_1000.json')


def bas(ax, t):
    ax.set_title(t, color=BLUE, fontsize=10, loc='left')


def kayip():
    if not G23 or not G23.get('kayit'):
        return
    k, v = G23['kayit'], G23['val']
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    x = np.array([r['token'] for r in k]) / 1e6
    y = np.array([r['loss'] for r in k])
    ax.plot(x, y, color=BLUE2, lw=0.8, alpha=0.35)
    w = 15
    if len(y) > w:
        yy = np.convolve(y, np.ones(w) / w, mode='valid')
        ax.plot(x[w - 1:], yy, color=BLUE, lw=1.8, label='train (15 adim ort.)')
    ax.plot([r['token'] / 1e6 for r in v], [r['val'] for r in v], 'o-',
            color=RED, lw=1.6, ms=3.5, label='val (gorulmemis dosya)')
    ax.axhline(G23['taban'], color=GREY, ls=':', lw=1)
    ax.text(x.max() * 0.62, G23['taban'] + 0.12,
            f"egitimsiz taban ln(50257) = {G23['taban']:.2f}", fontsize=8,
            color=GREY)
    bas(ax, 'Train ve val loss, ayni grafikte')
    ax.set_xlabel('gorulen egitim token\'i (milyon)')
    ax.set_ylabel('loss (nat/token)')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig('kayip.png', dpi=200); plt.close(fig)
    print('  kayip.png')


def hellaswag():
    if not (HB and HO):
        return
    n = HB['n']
    se = math.sqrt(0.25 * 0.75 / n)
    tok = f"~{G23['gorulen_token']/1e6:.1f}M token".replace('.', ',') if G23 else ''
    ad = ['rastgele', f'benim modelim\n({tok})', 'OpenAI GPT-2\n(124M)']
    deger = [0.25, HB['acc_norm'], HO['acc_norm']]
    renk = [GREY, BLUE2, BLUE]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.axhspan(0.25 - 1.96 * se, 0.25 + 1.96 * se, color=LITE, zorder=0)
    ax.text(-0.38, 0.25 + 1.96 * se + 0.002, 'rastgelenin %95 araligi',
            fontsize=7.5, color=GREY, ha='left')
    b = ax.bar(ad, deger, color=renk, width=0.55, zorder=2)
    for r, d in zip(b, deger):
        ax.text(r.get_x() + r.get_width() / 2, d + 0.004, f'%{100*d:.1f}',
                ha='center', fontsize=9, color=BLUE)
    if G4 and G4.get('referans'):
        ax.axhline(G4['referans']['acc_norm'], color=RED, ls='--', lw=1)
        ax.text(-0.3, G4['referans']['acc_norm'] + 0.003,
                f"OpenAI, tam set (10.042): %{100*G4['referans']['acc_norm']:.2f}",
                fontsize=7.5, color=RED)
    ax.set_ylim(0.20, max(deger) + 0.04)
    ax.set_ylabel('acc_norm')
    bas(ax, f'HellaSwag, ilk {n} ornek')
    ax.grid(axis='y', alpha=.25, lw=.5)
    fig.tight_layout(); fig.savefig('hellaswag.png', dpi=200); plt.close(fig)
    print('  hellaswag.png')


def olcek():
    if not G23 or not G23.get('gorulen_token'):
        return
    ad = ['benim kosum\n(CPU, ~1 saat)', 'Karpathy\n(8 x A100)', 'GPT-3 makalesi\n(125M model)']
    deger = [G23['gorulen_token'], 10e9, 300e9]
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    b = ax.barh(ad[::-1], deger[::-1], color=[GREY, BLUE2, BLUE2][::-1], height=.55)
    ax.set_xscale('log')
    for r, d in zip(b, deger[::-1]):
        t = f'{d/1e9:.0f} milyar' if d >= 1e9 else f'{d/1e6:.1f} milyon'
        ax.text(d * 1.15, r.get_y() + r.get_height() / 2, t, va='center',
                fontsize=8.5, color=BLUE)
    ax.set_xlim(1e6, 3e12)
    bas(ax, 'Gorulen egitim token\'i (log olcek)')
    fig.tight_layout(); fig.savefig('olcek.png', dpi=200); plt.close(fig)
    print('  olcek.png')


def turkce():
    if not G5 or 'egitim' not in G5:
        return
    k, v = G5['egitim']['kayit'], G5['egitim']['val']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4),
                                 gridspec_kw=dict(width_ratios=[1.1, 1]))
    if k:
        a1.plot([r['token'] / 1e6 for r in k], [r['loss'] for r in k],
                color=BLUE2, lw=1, alpha=.6, label='train')
    a1.plot([r['token'] / 1e6 for r in v], [r['val'] for r in v], 'o-',
            color=RED, ms=3.5, lw=1.5, label='val')
    toplam = G5['veri']['train']
    a1.axvline(toplam / 1e6, color=GREY, ls=':', lw=1)
    a1.text(toplam / 1e6 * 1.02, a1.get_ylim()[1] * 0.92, '1. devir bitti',
            fontsize=7.5, color=GREY)
    bas(a1, 'Turkce egitim: train ve val')
    a1.set_xlabel('gorulen token (milyon)'); a1.legend(frameon=False, fontsize=8)
    a1.grid(alpha=.25, lw=.5)
    if 'bpc' in G5:
        t = G5['bpc']
        adlar = list(t.keys())
        x = np.arange(len(adlar))
        a2.bar(x - 0.18, [t[a]['tr']['bpc'] for a in adlar], 0.36,
               color=RED, label='Turkce val')
        a2.bar(x + 0.18, [t[a]['en']['bpc'] for a in adlar], 0.36,
               color=BLUE2, label='Ingilizce val')
        for i, a in enumerate(adlar):
            a2.text(i - 0.18, t[a]['tr']['bpc'] + 0.05, f"{t[a]['tr']['bpc']:.2f}",
                    ha='center', fontsize=7.5)
            a2.text(i + 0.18, t[a]['en']['bpc'] + 0.05, f"{t[a]['en']['bpc']:.2f}",
                    ha='center', fontsize=7.5)
        kisa = [a.replace(' (', '\n(') for a in adlar]
        a2.set_xticks(x); a2.set_xticklabels(kisa, fontsize=7)
        bas(a2, 'Karakter basina bit (az iyi)')
        a2.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig('turkce.png', dpi=200); plt.close(fig)
    print('  turkce.png')


if __name__ == '__main__':
    kayip(); hellaswag(); olcek(); turkce()
