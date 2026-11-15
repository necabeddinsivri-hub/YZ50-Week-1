# gorseller.py - rapor ve sunum icin diyagramlar (egitim yok)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from matplotlib.patches import FancyBboxPatch

NAVY, GOLD, ICE, GRAY = '#00285A', '#B08D3F', '#C6DAF0', '#9AA3B2'

# ---------- 1) tril maskesi ve softmax ----------
T = 8
tril = torch.tril(torch.ones(T, T))
esit = tril / tril.sum(1, keepdim=True)
torch.manual_seed(7)
skor = torch.randn(T, T)
ogrenilen = F.softmax(skor.masked_fill(tril == 0, float('-inf')), -1)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
for ax, M, t in zip(axes, [tril, esit, ogrenilen],
                    ['1. tril maskesi\n(gelecek kapali)',
                     '2. satira bolunce\n(esit agirlikli ortalama)',
                     '3. ogrenilen skorlarla softmax\n(self-attention)']):
    im = ax.imshow(M.numpy(), cmap='Blues', vmin=0, vmax=1)
    for i in range(T):
        for j in range(T):
            if M[i, j] > 0:
                ax.text(j, i, f"{M[i,j]:.2f}".lstrip('0'), ha='center', va='center',
                        fontsize=7.5, color='white' if M[i, j] > 0.6 else NAVY)
    ax.set_title(t, fontsize=11, color=NAVY, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig('tril_softmax.png', dpi=120, bbox_inches='tight'); plt.close()

# ---------- 2) q, k, v semasi ----------
fig, ax = plt.subplots(figsize=(11, 4.6))


def kutu(x, y, w, h, metin, renk, yazi='white', fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=renk, ec=renk))
    ax.text(x + w/2, y + h/2, metin, ha='center', va='center',
            color=yazi, fontsize=fs, fontweight='bold')


def ok(x1, y1, x2, y2, etiket=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.4))
    if etiket:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.08, etiket, ha='center', fontsize=9, color=GRAY)


kutu(0.2, 1.7, 1.5, 0.7, 'x\n(B,T,32)', ICE, NAVY)
for k, (y, ad, acik) in enumerate([(2.9, 'query  q', '"ne ariyorum"'),
                                   (1.7, 'key    k', '"bende ne var"'),
                                   (0.5, 'value  v', '"ne veririm"')]):
    kutu(2.4, y, 1.9, 0.7, ad, NAVY)
    ax.text(3.35, y - 0.25, acik, ha='center', fontsize=9, color=GRAY, style='italic')
    ok(1.7, 2.05, 2.4, y + 0.35)
kutu(5.1, 2.3, 2.1, 0.7, 'q @ k.T\n/ sqrt(d)', GOLD)
ok(4.3, 3.25, 5.1, 2.8); ok(4.3, 2.05, 5.1, 2.5)
kutu(7.9, 2.3, 1.9, 0.7, 'tril + softmax', GOLD)
ok(7.2, 2.65, 7.9, 2.65)
kutu(7.9, 0.85, 1.9, 0.7, 'wei @ v', NAVY)
ok(8.85, 2.3, 8.85, 1.55, 'wei')
ok(4.3, 0.85, 7.9, 1.1)
kutu(10.3, 0.85, 1.5, 0.7, 'cikti', ICE, NAVY)
ok(9.8, 1.2, 10.3, 1.2)
ax.set_xlim(0, 12); ax.set_ylim(0.2, 3.9); ax.axis('off')
ax.set_title("Tek bir self-attention head", fontsize=13, color=NAVY, fontweight='bold')
plt.tight_layout(); plt.savefig('head_semasi.png', dpi=120, bbox_inches='tight'); plt.close()

# ---------- 3) olcekleme etkisi ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
s = np.array([0.1, -0.2, 0.3, -0.2, 0.5])


def sm(z):
    e = np.exp(z - z.max()); return e / e.sum()


for ax, z, t in zip(axes, [s, s * 8],
                    ['sqrt(head_size) ile bolunmus\n(skorlar kucuk)',
                     'bolunmemis\n(skorlar 8 kat buyuk)']):
    ax.bar(range(5), sm(z), color=[NAVY if i != z.argmax() else GOLD for i in range(5)])
    for i, v in enumerate(sm(z)):
        ax.text(i, v + 0.015, f"{v:.2f}", ha='center', fontsize=9, color=NAVY)
    ax.set_ylim(0, 1.0); ax.set_xticks(range(5), [f"t{i}" for i in range(5)])
    ax.set_title(t, fontsize=11, color=NAVY, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('softmax agirligi')
plt.tight_layout(); plt.savefig('olcekleme.png', dpi=120, bbox_inches='tight'); plt.close()

# ---------- 4) val loss karsilastirmasi ----------
fig, ax = plt.subplots(figsize=(7.6, 3.8))
modeller = ['Bigram\n(1 harf)', 'Tek head\n(8 harf, attention)']
val = [2.4769, 2.3998]
par = [4225, 7553]
c = ax.bar(modeller, val, color=[GRAY, NAVY], width=0.5)
for b, v, p in zip(c, val, par):
    ax.text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.4f}", ha='center',
            fontsize=12, fontweight='bold', color=NAVY)
    ax.text(b.get_x()+b.get_width()/2, 2.355, f"{p:,} param".replace(',', '.'),
            ha='center', fontsize=9.5, color='white' if p > 5000 else NAVY)
ax.set_ylim(2.34, 2.52); ax.set_ylabel('val loss')
ax.set_title('Tiny Shakespeare, 5.000 adim', color=NAVY, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig('karsilastirma.png', dpi=120, bbox_inches='tight'); plt.close()

print("tril_softmax.png, head_semasi.png, olcekleme.png, karsilastirma.png kaydedildi")
