# gorseller.py - rapor ve sunum icin grafikler
# YZ50 Hafta 9
# Butun sayilar sonuc_gorev*.json dosyalarindan, yani gercek kosulardan.

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE = '#00285A'
BLUE2 = '#0F6DB5'
GREY = '#63697A'
LITE = '#EEF2F7'
RED = '#A3242B'

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#C8CCD6',
                     'axes.labelcolor': BLUE, 'text.color': BLUE,
                     'xtick.color': GREY, 'ytick.color': GREY})


def yukle(n):
    y = f'sonuc_gorev{n}.json'
    return json.load(open(y)) if os.path.exists(y) else None


G2, G3, G4, G5, G6a, G6b = (yukle(2), yukle(3), yukle(4), yukle(5),
                            yukle('6a'), yukle('6b'))


def baslik(ax, t):
    ax.set_title(t, color=BLUE, fontsize=10, loc='left')


# =====================================================================
# 1) ILK 20 BIRLESTIRME
# =====================================================================
def ilk20():
    if not G2:
        return
    r = G2['ilk20']
    etiket = [x['metin'].replace('\\u000a', '\\n') for x in r]
    adet = [x['adet'] for x in r]
    renk = [RED if len(x['metin']) == 1 and ord(x['metin'][0]) > 127 else BLUE2
            for x in r]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    b = ax.bar(range(len(adet)), adet, color=renk, width=.68)
    ax.bar_label(b, labels=[f"{a/1000:.1f}k" for a in adet], fontsize=7,
                 padding=2, color=GREY)
    ax.set_xticks(range(len(etiket)),
                  [f"{i+1}\n{e!r}" for i, e in enumerate(etiket)], fontsize=7)
    ax.set_ylabel('kac kere gecti')
    ax.set_ylim(0, max(adet) * 1.16)
    baslik(ax, 'Exhibit — ilk 20 birlestirme (kirmizi: Turkce’ye ozel '
                'tek harf)')
    ax.grid(axis='y', color=LITE, lw=1)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('ilk20.png', dpi=130)
    plt.close()
    print('  ilk20.png')


# =====================================================================
# 2) DIZI KISALMASI + VOCAB TARAMASI
# =====================================================================
def sikistirma():
    if not G2:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.5))

    k = G2['kisalma']
    x = [p[0] for p in k]
    y = [p[1] / 1000 for p in k]
    a1.plot(x, y, 'o-', color=BLUE, lw=2, ms=6)
    for xi, yi in zip(x, y):
        a1.annotate(f"{yi:.0f}k", (xi, yi), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=7.5, color=BLUE)
    a1.set_xlabel('birlestirme sayisi')
    a1.set_ylabel('kalan token (bin)')
    baslik(a1, 'egitim boyunca dizi kisalmasi')
    a1.grid(color=LITE, lw=1)
    a1.set_axisbelow(True)

    t = G2['tarama']
    v = [r[0] for r in t]
    oran = [r[2] for r in t]
    sure = [r[3] for r in t]
    b = a2.bar(range(len(v)), oran, color=BLUE2, width=.55)
    a2.bar_label(b, fmt='%.2fx', fontsize=8, padding=2, color=BLUE)
    a2.set_xticks(range(len(v)), [str(i) for i in v], fontsize=8)
    a2.set_xlabel('vocab boyutu')
    a2.set_ylabel('sikistirma (byte/token)')
    a2.set_ylim(0, max(oran) * 1.25)
    a3 = a2.twinx()
    a3.plot(range(len(v)), sure, 's--', color=RED, lw=1.4, ms=5)
    a3.set_ylabel('egitim suresi (sn)', color=RED, fontsize=8)
    a3.tick_params(axis='y', colors=RED, labelsize=7.5)
    baslik(a2, 'vocab buyudukce: kazanc ve maliyet')
    a2.grid(axis='y', color=LITE, lw=1)
    a2.set_axisbelow(True)
    fig.suptitle('Exhibit — sikistirma', color=BLUE, fontsize=10, x=.012,
                 ha='left')
    plt.tight_layout()
    plt.savefig('sikistirma.png', dpi=130)
    plt.close()
    print('  sikistirma.png')


# =====================================================================
# 3) VOCAB YAPISI: TEMEL vs REGEX
# =====================================================================
def vocab_yapisi():
    if not G4:
        return
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    kat = ['sonu bosluk\n(’n ’, ’bir ’)',
           'icinde noktalama\n(’.\\nB’)',
           'basi bosluk\n(’ bir’)']
    temel = [G4['temel_sonek'], G4['temel_nokt'], G4['temel_onek']]
    regex = [G4['regex_sonek'], G4['regex_nokt'], G4['regex_onek']]
    x = np.arange(3)
    b1 = ax.bar(x - .19, temel, .36, color=GREY, label='TEMEL (regex yok)')
    b2 = ax.bar(x + .19, regex, .36, color=BLUE, label='REGEX (GPT-4 deseni)')
    ax.bar_label(b1, fontsize=8, padding=2, color=GREY)
    ax.bar_label(b2, fontsize=8, padding=2, color=BLUE)
    ax.set_xticks(x, kat, fontsize=8)
    ax.set_ylabel('token sayisi')
    ax.set_ylim(0, max(temel + regex) * 1.2)
    ax.legend(frameon=False, fontsize=8.5)
    baslik(ax, 'Exhibit — regex deseni vocab’in yapisini degistiriyor')
    ax.grid(axis='y', color=LITE, lw=1)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('vocab_yapisi.png', dpi=130)
    plt.close()
    print('  vocab_yapisi.png')


# =====================================================================
# 4) TURKCE EKLERI
# =====================================================================
def ekler():
    if not G4 or 'ek_bozuk' not in G4:
        return
    bozuk = [e for e, _ in G4['ek_bozuk']]
    temiz = [e for e, _ in G4['ek_temiz']]
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ax.axis('off')
    ax.text(0.0, 1.02, 'Exhibit — GPT-4 deseni 26 Turkce ekin 12’sini '
            'ikiye boluyor', fontsize=10, color=BLUE, fontweight='bold')
    ax.add_patch(plt.Rectangle((0.0, .52), .47, .40, fc='#F7E9EA',
                               ec=RED, lw=1.2, transform=ax.transAxes))
    ax.add_patch(plt.Rectangle((.52, .52), .47, .40, fc=LITE,
                               ec=BLUE2, lw=1.2, transform=ax.transAxes))
    ax.text(.02, .86, f'BOZULAN ({len(bozuk)})', fontsize=9, color=RED,
            fontweight='bold', transform=ax.transAxes)
    ax.text(.02, .60, '  '.join(bozuk), fontsize=8.5, color=BLUE,
            transform=ax.transAxes, va='bottom', wrap=True)
    ax.text(.54, .86, f'TEMIZ ({len(temiz)})', fontsize=9, color=BLUE2,
            fontweight='bold', transform=ax.transAxes)
    ax.text(.54, .60, '  '.join(temiz), fontsize=8.5, color=BLUE,
            transform=ax.transAxes, va='bottom')
    ax.text(0.0, .36, "Bozulanlarin hepsi d / t / s / m ile basliyor — "
            "cunku desenin ILK kurali Ingilizce kisaltmalari "
            "('d 's 'm 't) yakaliyor",
            fontsize=8.5, color=RED, transform=ax.transAxes)
    ax.text(0.0, .22, "ve ekin sadece ilk harfini yiyor:  Ankara'dan  "
            "→  ['Ankara', \"'d\", 'an']",
            fontsize=8.5, color=BLUE, transform=ax.transAxes,
            fontfamily='monospace')
    ax.text(0.0, .04, "Bunlar Turkce'nin en sik ekleri: bulunma (-da/-de/"
            "-ta/-te) ve ayrilma (-dan/-den/-tan/-ten) hali.",
            fontsize=8.5, color=GREY, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig('ekler.png', dpi=130)
    plt.close()
    print('  ekler.png')


# =====================================================================
# 5) TURKCE vs INGILIZCE
# =====================================================================
def diller():
    if not G5:
        return
    tr, en = G5['gpt4_tr'], G5['gpt4_en']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.4))

    kat = ['kelime', 'token', 'token / kelime']
    tv = [tr['kelime'], tr['token'], tr['token_kelime']]
    ev = [en['kelime'], en['token'], en['token_kelime']]
    x = np.arange(2)
    b1 = a1.bar(x - .18, [tr['kelime'], tr['token']], .34, color=BLUE,
                label='Turkce')
    b2 = a1.bar(x + .18, [en['kelime'], en['token']], .34, color=GREY,
                label='Ingilizce')
    a1.bar_label(b1, fontsize=8, padding=2, color=BLUE)
    a1.bar_label(b2, fontsize=8, padding=2, color=GREY)
    a1.set_xticks(x, ['kelime sayisi', 'token sayisi'], fontsize=8.5)
    a1.legend(frameon=False, fontsize=8.5)
    a1.set_ylim(0, max(tv[:2] + ev[:2]) * 1.2)
    baslik(a1, 'ayni paragraf, GPT-4 tokenizer')
    a1.grid(axis='y', color=LITE, lw=1)
    a1.set_axisbelow(True)

    b3 = a2.bar(['Turkce', 'Ingilizce'],
                [tr['token_kelime'], en['token_kelime']],
                color=[BLUE, GREY], width=.5)
    a2.bar_label(b3, fmt='%.3f', fontsize=9, padding=2, color=BLUE)
    a2.set_ylabel('token / kelime')
    a2.set_ylim(0, tr['token_kelime'] * 1.3)
    baslik(a2, f"kelime basina {tr['token_kelime']/en['token_kelime']:.2f}x")
    a2.grid(axis='y', color=LITE, lw=1)
    a2.set_axisbelow(True)
    fig.suptitle('Exhibit — Turkce GPT-4’te daha pahali', color=BLUE,
                 fontsize=10, x=.012, ha='left')
    plt.tight_layout()
    plt.savefig('diller.png', dpi=130)
    plt.close()
    print('  diller.png')


# =====================================================================
# 6) TOKENIZER'LAR YAN YANA
# =====================================================================
def tokenizerlar():
    if not G5:
        return
    sira = [('harf\n(Hafta 8)', G5['harf'], 87),
            ('benim, temel\n(Hafta 9)', G5['benim_temel'], 1024),
            ('benim, regex\n(Hafta 9)', G5['benim_regex'], 1024),
            ('GPT-4\ncl100k_base', G5['gpt4_tr'], 100256)]
    ad = [s[0] for s in sira]
    tok = [s[1]['token'] for s in sira]
    byt = [s[1]['byte_token'] for s in sira]
    v = [s[2] for s in sira]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.5))
    b = a1.bar(range(4), tok, color=[GREY, BLUE2, BLUE, '#8A94A6'], width=.56)
    a1.bar_label(b, fontsize=8.5, padding=2, color=BLUE)
    a1.set_xticks(range(4), ad, fontsize=7.5)
    a1.set_ylabel('ayni paragraf kac token')
    a1.set_ylim(0, max(tok) * 1.18)
    baslik(a1, 'ayni Turkce paragraf')
    a1.grid(axis='y', color=LITE, lw=1)
    a1.set_axisbelow(True)

    b2 = a2.bar(range(4), byt, color=[GREY, BLUE2, BLUE, '#8A94A6'], width=.56)
    a2.bar_label(b2, fmt='%.2f', fontsize=8.5, padding=2, color=BLUE)
    a2.set_xticks(range(4), [f"{a}\nV={vv:,}".replace(',', '.')
                             for a, vv in zip(ad, v)], fontsize=7)
    a2.set_ylabel('byte / token')
    a2.set_ylim(0, max(byt) * 1.2)
    baslik(a2, 'sikistirma')
    a2.grid(axis='y', color=LITE, lw=1)
    a2.set_axisbelow(True)
    fig.suptitle('Exhibit — 1024 token’lik kendi tokenizer’im, '
                 'Turkce’de GPT-4’u geciyor', color=BLUE,
                 fontsize=10, x=.012, ha='left')
    plt.tight_layout()
    plt.savefig('tokenizerlar.png', dpi=130)
    plt.close()
    print('  tokenizerlar.png')


# =====================================================================
# 7) HAFTA 8 vs HAFTA 9 MODELI (bpc)
# =====================================================================
def model():
    if not G6b:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.5))
    ad = [f"Hafta 8\nharf, V={G6b['h8_V']}",
          f"Hafta 9\nBPE, V={G6b['V']}"]
    ham = [G6b['h8_val'], G6b['val']]
    bpc = [G6b['h8_bpc'], G6b['h9_bpc']]

    b = a1.bar(ad, ham, color=[GREY, BLUE], width=.5)
    a1.bar_label(b, fmt='%.4f', fontsize=9, padding=2, color=BLUE)
    a1.set_ylabel('val loss (nat / token)')
    a1.set_ylim(0, max(ham) * 1.3)
    baslik(a1, 'ham val loss — KARSILASTIRILAMAZ')
    a1.grid(axis='y', color=LITE, lw=1)
    a1.set_axisbelow(True)
    a1.text(0.02, .93, 'token = karakter       token ≈ '
            f"{1/G6b['karakter_basina_token']:.2f} karakter",
            transform=a1.transAxes, fontsize=7.5, color=RED)

    b2 = a2.bar(ad, bpc, color=[GREY, BLUE], width=.5)
    a2.bar_label(b2, fmt='%.4f', fontsize=9, padding=2, color=BLUE)
    a2.set_ylabel('bit / karakter')
    a2.set_ylim(0, max(bpc) * 1.3)
    baslik(a2, 'ortak olcek: karakter basina bit')
    a2.grid(axis='y', color=LITE, lw=1)
    a2.set_axisbelow(True)
    a2.text(0.02, .93, f"fark {G6b['h8_bpc']-G6b['h9_bpc']:+.4f}",
            transform=a2.transAxes, fontsize=8, color=BLUE2, fontweight='bold')
    fig.suptitle('Exhibit — ayni model, iki tokenizer', color=BLUE,
                 fontsize=10, x=.012, ha='left')
    plt.tight_layout()
    plt.savefig('model_bpc.png', dpi=130)
    plt.close()
    print('  model_bpc.png')


# =====================================================================
# 8) BPE SEMASI
# =====================================================================
def sema():
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.axis('off')
    ax.text(0, 1.0, 'Exhibit — BPE tek bir dongu', fontsize=10, color=BLUE,
            fontweight='bold', transform=ax.transAxes)
    adimlar = [
        ('1', 'metni UTF-8 byte’lara cevir', '0-255 arasi sayilar'),
        ('2', 'ardisik IKILILERI say', '(97,114) kac kere gecti'),
        ('3', 'en siki sec, yeni numara ver', '(97,114) → 256'),
        ('4', 'dizide o ikiliyi degistir', 'dizi kisaldi'),
        ('5', 'vocab dolana kadar 2’ye don', '768 kere tekrar'),
    ]
    y = .76
    for no, baslik_, alt in adimlar:
        ax.add_patch(plt.Rectangle((.03, y - .055), .94, .125, fc=LITE,
                                   ec='#C8CCD6', lw=.8, transform=ax.transAxes))
        ax.text(.06, y + .005, no, fontsize=13, color=BLUE2,
                fontweight='bold', transform=ax.transAxes, va='center')
        ax.text(.12, y + .005, baslik_, fontsize=10, color=BLUE,
                fontweight='bold', transform=ax.transAxes, va='center')
        ax.text(.62, y + .005, alt, fontsize=8.5, color=GREY,
                transform=ax.transAxes, va='center', fontfamily='monospace')
        y -= .16
    ax.text(.03, .02, 'Tabani 256 byte oldugu icin HER metni temsil '
            'edebiliyor — "bilinmeyen karakter" diye bir sorun yok.',
            fontsize=8.5, color=BLUE2, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig('bpe_semasi.png', dpi=130)
    plt.close()
    print('  bpe_semasi.png')


if __name__ == '__main__':
    print('Grafikler uretiliyor:')
    ilk20(); sikistirma(); vocab_yapisi(); ekler(); diller()
    tokenizerlar(); model(); sema()
    print('Bitti.')
