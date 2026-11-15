# gorseller.py - rapor ve sunum icin grafikler
# YZ50 Hafta 10
# Butun sayilar sonuc_gorev*.json dosyalarindan.

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE = '#00285A'; BLUE2 = '#0F6DB5'; GREY = '#63697A'
LITE = '#EEF2F7'; RED = '#A3242B'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#C8CCD6',
                     'axes.labelcolor': BLUE, 'text.color': BLUE,
                     'xtick.color': GREY, 'ytick.color': GREY})

J = lambda n: (json.load(open(f'sonuc_gorev{n}.json'))
               if os.path.exists(f'sonuc_gorev{n}.json') else None)
G1, G2, G3, G4, G5a, G5b = J(1), J(2), J(3), J(4), J('5a'), J('5b')


def bas(ax, t):
    ax.set_title(t, color=BLUE, fontsize=10, loc='left')


# =====================================================================
# 1) EZBERLEME EGRISI  (Gorev 4)
# =====================================================================
def ezber():
    if not G4:
        return
    k = G4['ezber']['kayit']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    a1.plot(k, color=BLUE, lw=1.8)
    a1.axhline(G4['beklenen'], color=RED, ls='--', lw=1.2)
    a1.annotate(f"ln(50257) = {G4['beklenen']:.2f}", (len(k)*0.42, G4['beklenen']),
                textcoords='offset points', xytext=(0, 6), fontsize=8, color=RED)
    a1.set_xlabel('adim'); a1.set_ylabel('loss')
    bas(a1, 'dogrusal olcek')
    a1.grid(color=LITE, lw=1); a1.set_axisbelow(True)

    a2.semilogy(k, color=BLUE, lw=1.8)
    a2.set_xlabel('adim'); a2.set_ylabel('loss (log)')
    bas(a2, f"log olcek — son deger {k[-1]:.5f}")
    a2.grid(color=LITE, lw=1, which='both'); a2.set_axisbelow(True)
    fig.suptitle('Exhibit — tek batch ezberletme: 10,96 → '
                 f"{k[-1]:.4f}", color=BLUE, fontsize=10, x=.012, ha='left')
    plt.tight_layout(); plt.savefig('ezber.png', dpi=130); plt.close()
    print('  ezber.png')


# =====================================================================
# 2) ILK LOSS + WEIGHT TYING HATASI  (Gorev 4)
# =====================================================================
def ilk_loss():
    if not G4:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    adlar = list(G4['ilk_loss'].keys())
    kisa = ['GPT-2 semasi\n(+tying)', 'GPT-2 semasi\n(tying yok)',
            'PyTorch\nvarsayilan']
    vals = [G4['ilk_loss'][a]['loss'] for a in adlar]
    b = a1.bar(range(3), vals, color=[BLUE, BLUE2, GREY], width=.55)
    a1.bar_label(b, fmt='%.3f', fontsize=8.5, padding=2, color=BLUE)
    a1.axhline(G4['beklenen'], color=RED, ls='--', lw=1.4)
    a1.annotate(f"ln(V) = {G4['beklenen']:.2f}", (2.3, G4['beklenen']),
                textcoords='offset points', xytext=(0, 5), fontsize=8,
                color=RED, ha='right')
    a1.set_xticks(range(3), kisa, fontsize=7.5)
    a1.set_ylim(10.4, max(vals) * 1.03)
    a1.set_ylabel('ilk loss')
    bas(a1, 'rastgele baslatilmis modelin ilk loss’u')
    a1.grid(axis='y', color=LITE, lw=1); a1.set_axisbelow(True)

    if 'tying_hatasi' in G4:
        o = G4['tying_hatasi']
        x = np.arange(3)
        b1 = a2.bar(x - .19, o[0][1:], .36, color=BLUE, label='tying VAR')
        b2 = a2.bar(x + .19, o[1][1:], .36, color=GREY, label='tying YOK')
        a2.bar_label(b1, fmt='%.2f', fontsize=7.5, padding=2, color=BLUE)
        a2.bar_label(b2, fmt='%.2f', fontsize=7.5, padding=2, color=GREY)
        a2.axhline(G4['beklenen'], color=RED, ls='--', lw=1.2)
        a2.set_xticks(x, ['hedef = girdi\n(HATALI test)',
                          'bagimsiz\nrastgele', 'gercek\nmetin'], fontsize=7.5)
        a2.set_ylim(9.5, 11.4); a2.set_ylabel('loss')
        a2.legend(frameon=False, fontsize=8)
        bas(a2, 'sanity check kendi hatami yakaladi')
        a2.grid(axis='y', color=LITE, lw=1); a2.set_axisbelow(True)
    fig.suptitle('Exhibit — ln(V) kontrolu', color=BLUE, fontsize=10,
                 x=.012, ha='left')
    plt.tight_layout(); plt.savefig('ilk_loss.png', dpi=130); plt.close()
    print('  ilk_loss.png')


# =====================================================================
# 3) WEIGHT TYING + EGITIM  (Gorev 5a)
# =====================================================================
def tying():
    if not G5a:
        return
    t = G5a['tying']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    b = a1.bar(['tying YOK', 'tying VAR'], [t['free'] / 1e6, t['tied'] / 1e6],
               color=[GREY, BLUE], width=.5)
    a1.bar_label(b, fmt='%.1fM', fontsize=9, padding=2, color=BLUE)
    a1.set_ylabel('parametre (milyon)')
    a1.set_ylim(0, t['free'] / 1e6 * 1.2)
    a1.annotate('', (1, t['tied'] / 1e6), (1, t['free'] / 1e6),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.4))
    a1.text(1.12, (t['tied'] + t['free']) / 2e6,
            f"−{t['kazanc']/1e6:.1f}M\n({100*t['kazanc']/t['free']:.0f}%)",
            fontsize=8.5, color=RED, va='center')
    bas(a1, 'weight tying kazanci = embedding tablosu')
    a1.grid(axis='y', color=LITE, lw=1); a1.set_axisbelow(True)

    if 'egitim' in G5a:
        k = G5a['egitim']['kayit']
        a2.plot(k, color=BLUE, lw=1.4)
        a2.axhline(G4['beklenen'] if G4 else math.log(50257), color=RED,
                   ls='--', lw=1.2)
        a2.set_xlabel('adim'); a2.set_ylabel('loss')
        bas(a2, f"metnin tamaminda {G5a['egitim']['adim']} adim "
                f"({k[0]:.1f} → {k[-1]:.1f})")
        a2.grid(color=LITE, lw=1); a2.set_axisbelow(True)
    fig.suptitle('Exhibit — weight tying ve ilk gercek egitim', color=BLUE,
                 fontsize=10, x=.012, ha='left')
    plt.tight_layout(); plt.savefig('tying.png', dpi=130); plt.close()
    print('  tying.png')


# =====================================================================
# 4) TURKCE / INGILIZCE LOSS  (Gorev 5b)
# =====================================================================
def diller():
    if not G5b:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.5))
    n = len(G5b['ciftler'])
    x = np.arange(n)
    tr = [c['tr_loss'] for c in G5b['ciftler']]
    en = [c['en_loss'] for c in G5b['ciftler']]
    b1 = a1.bar(x - .19, tr, .36, color=BLUE, label='Turkce')
    b2 = a1.bar(x + .19, en, .36, color=GREY, label='Ingilizce')
    a1.axhline(G5b['taban'], color=RED, ls='--', lw=1.3)
    a1.annotate(f"ln(V) = {G5b['taban']:.2f}\n(egitimsiz taban)",
                (n - 0.6, G5b['taban']), textcoords='offset points',
                xytext=(0, 4), fontsize=7.5, color=RED)
    a1.set_xticks(x, [str(i + 1) for i in range(n)], fontsize=8)
    a1.set_xlabel('cumle'); a1.set_ylabel('loss (nat / token)')
    a1.set_ylim(0, G5b['taban'] * 1.15)
    a1.legend(frameon=False, fontsize=8.5)
    bas(a1, 'ayni icerik, iki dil')
    a1.grid(axis='y', color=LITE, lw=1); a1.set_axisbelow(True)

    kat = ['Ingilizce', 'Turkce\n(sade)', 'Turkce\n(ç ğ ı ö ş ü)']
    v = [G5b['en_ort'], G5b['tr_ort'], G5b['isaretli_ort']]
    b3 = a2.bar(kat, v, color=[GREY, BLUE, BLUE2], width=.5)
    a2.bar_label(b3, fmt='%.3f', fontsize=9, padding=2, color=BLUE)
    a2.axhline(G5b['taban'], color=RED, ls='--', lw=1.3)
    for i, val in enumerate(v):
        a2.annotate(f"tabanin\n%{100*val/G5b['taban']:.0f}'i",
                    (i, val / 2), ha='center', fontsize=7.5, color='white')
    a2.set_ylim(0, G5b['taban'] * 1.15)
    a2.set_ylabel('ortalama loss')
    bas(a2, 'ortalama — ve egitimsiz tabana uzaklik')
    a2.grid(axis='y', color=LITE, lw=1); a2.set_axisbelow(True)
    fig.suptitle('Exhibit — GPT-2 Turkce’de neredeyse egitimsiz gibi',
                 color=BLUE, fontsize=10, x=.012, ha='left')
    plt.tight_layout(); plt.savefig('diller.png', dpi=130); plt.close()
    print('  diller.png')


# =====================================================================
# 5) TOKEN SAYILARI  (Gorev 3)
# =====================================================================
def token():
    if not G3:
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    c = G3['ciftler']
    x = np.arange(len(c))
    b1 = ax.bar(x - .19, [r[1] for r in c], .36, color=BLUE, label='Turkce')
    b2 = ax.bar(x + .19, [r[3] for r in c], .36, color=GREY, label='Ingilizce')
    ax.bar_label(b1, fontsize=8, padding=2, color=BLUE)
    ax.bar_label(b2, fontsize=8, padding=2, color=GREY)
    ax.set_xticks(x, [f"cumle {i+1}" for i in range(len(c))], fontsize=8.5)
    ax.set_ylabel('token sayisi')
    ax.set_ylim(0, max(max(r[1] for r in c), max(r[3] for r in c)) * 1.2)
    ax.legend(frameon=False, fontsize=8.5)
    bas(ax, f"Exhibit — GPT-2 tokenizer’inda Turkce "
            f"{G3['oran']:.2f} kat token")
    ax.grid(axis='y', color=LITE, lw=1); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig('token.png', dpi=130); plt.close()
    print('  token.png')


# =====================================================================
# 6) PARAMETRE DAGILIMI  (Gorev 1)
# =====================================================================
def param():
    if not G1:
        return
    emb = G1['emb_param']
    pos = 1024 * 768
    toplam = G1['hesap_param']
    blok = toplam - emb - pos - 2 * 768
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    kat = ['token embedding\n(wte)', '12 blok', 'pozisyon emb.\n(wpe)',
           'son ln_f']
    v = [emb, blok, pos, 2 * 768]
    b = ax.barh(kat[::-1], [x / 1e6 for x in v][::-1],
                color=[GREY, BLUE2, BLUE, BLUE][::-1], height=.55)
    ax.bar_label(b, fmt='%.1fM', fontsize=9, padding=3, color=BLUE)
    ax.set_xlabel('parametre (milyon)')
    ax.set_xlim(0, max(v) / 1e6 * 1.25)
    bas(ax, f"Exhibit — 124M nereden geliyor "
            f"(wte toplam’in %{100*emb/toplam:.0f}’i)")
    ax.grid(axis='x', color=LITE, lw=1); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig('param.png', dpi=130); plt.close()
    print('  param.png')


if __name__ == '__main__':
    print('Grafikler uretiliyor:')
    ezber(); ilk_loss(); tying(); diller(); token(); param()
    print('Bitti.')
