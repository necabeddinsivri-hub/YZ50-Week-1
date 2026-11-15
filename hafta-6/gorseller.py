# gorseller.py - rapor ve sunum icin diyagramlar (egitim yok, sadece cizim)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAVY, GOLD, ICE, GRAY = '#00285A', '#B08D3F', '#C6DAF0', '#9AA3B2'

# ---------- 1) Duz MLP vs WaveNet agaci ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
harfler = ['.', '.', 's', 'o', 'p', 'h', 'i', 'a']


def kutu(ax, x, y, metin, renk, genislik=0.9, yazi='white', boy=0.55, fs=11):
    ax.add_patch(FancyBboxPatch((x - genislik/2, y - boy/2), genislik, boy,
                                boxstyle="round,pad=0.02", fc=renk, ec=renk))
    ax.text(x, y, metin, ha='center', va='center', color=yazi, fontsize=fs, fontweight='bold')


def ok(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=GRAY, lw=1.2, zorder=0)


# sol: duz
ax = axes[0]
for i, h in enumerate(harfler):
    kutu(ax, i, 0, h, ICE, 0.8, NAVY)
    ok(ax, i, 0.28, 3.5, 2.72)
kutu(ax, 3.5, 3.0, 'Linear 80 -> 200', NAVY, 3.2)
kutu(ax, 3.5, 4.2, 'sonraki harf', GOLD, 2.4)
ok(ax, 3.5, 3.28, 3.5, 3.92)
ax.set_title('Duz MLP: 8 harf TEK adimda ezilir', fontsize=13, color=NAVY, fontweight='bold')
ax.set_xlim(-0.8, 7.8); ax.set_ylim(-0.6, 4.8); ax.axis('off')

# sag: wavenet
ax = axes[1]
for i, h in enumerate(harfler):
    kutu(ax, i, 0, h, ICE, 0.8, NAVY)
seviye1 = [0.5, 2.5, 4.5, 6.5]
for j, x in enumerate(seviye1):
    ok(ax, 2*j, 0.28, x, 1.02); ok(ax, 2*j+1, 0.28, x, 1.02)
    kutu(ax, x, 1.3, 'cift', NAVY, 1.4, fs=10)
seviye2 = [1.5, 5.5]
for j, x in enumerate(seviye2):
    ok(ax, seviye1[2*j], 1.58, x, 2.22); ok(ax, seviye1[2*j+1], 1.58, x, 2.22)
    kutu(ax, x, 2.5, 'dortlu', NAVY, 1.8, fs=10)
ok(ax, 1.5, 2.78, 3.5, 3.42); ok(ax, 5.5, 2.78, 3.5, 3.42)
kutu(ax, 3.5, 3.7, 'sekizli', NAVY, 2.0, fs=10)
kutu(ax, 3.5, 4.6, 'sonraki harf', GOLD, 2.4)
ok(ax, 3.5, 3.98, 3.5, 4.32)
for y, t in [(1.3, '(B,4,68)'), (2.5, '(B,2,68)'), (3.7, '(B,68)')]:
    ax.text(7.9, y, t, fontsize=9.5, color=GRAY, va='center', family='monospace')
ax.set_title('WaveNet: ikiser ikiser, 3 katmanda', fontsize=13, color=NAVY, fontweight='bold')
ax.set_xlim(-0.8, 9.2); ax.set_ylim(-0.6, 5.1); ax.axis('off')
plt.tight_layout()
plt.savefig('wavenet_agac.png', dpi=120, bbox_inches='tight')
plt.close()

# ---------- 2) BatchNorm eksen diyagrami ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for k, (ax, baslik, eksen) in enumerate(zip(axes,
        ['HATALI: mean(0) -> her pozisyona ayri ortalama', 'DOGRU: mean((0,1)) -> kanal basina tek ortalama'],
        ['0', '(0,1)'])):
    B, T = 6, 4
    for b in range(B):
        for t in range(T):
            renk = [ICE, '#F2D9A7', '#D9E8C8', '#E8C9D9'][t] if k == 0 else ICE
            ax.add_patch(plt.Rectangle((t, B - b - 1), 0.92, 0.85, fc=renk, ec='white'))
    if k == 0:
        for t in range(T):
            ax.text(t + 0.46, -0.6, f'ort{t+1}', ha='center', fontsize=10, color=NAVY, fontweight='bold')
    else:
        ax.text(T/2, -0.6, 'tek ortalama (6 x 4 = 24 ornek)', ha='center', fontsize=10, color=NAVY, fontweight='bold')
    ax.text(-0.35, B/2, 'batch', rotation=90, va='center', fontsize=10, color=GRAY)
    ax.text(T/2, B + 0.25, 'pozisyon (cift 1..4)', ha='center', fontsize=10, color=GRAY)
    ax.set_title(baslik, fontsize=11.5, color=NAVY, fontweight='bold', pad=22)
    ax.set_xlim(-0.6, T + 0.2); ax.set_ylim(-1.0, B + 0.6); ax.axis('off')
plt.tight_layout()
plt.savefig('batchnorm_eksen.png', dpi=120, bbox_inches='tight')
plt.close()

# ---------- 3) karsilastirma cubuk grafigi ----------
modeller = ['Baglam 3\nduz MLP', 'Baglam 8\nduz MLP', 'Baglam 8\nWaveNet']
dev = [2.1030, 2.0101, 1.9942]
param = [20875, 44875, 76579]
fig, ax = plt.subplots(figsize=(8, 4.2))
cub = ax.bar(modeller, dev, color=[GRAY, ICE, NAVY], width=0.55)
for c, d, p in zip(cub, dev, param):
    ax.text(c.get_x() + c.get_width()/2, d + 0.004, f'{d:.4f}', ha='center', fontsize=12, fontweight='bold', color=NAVY)
    ax.text(c.get_x() + c.get_width()/2, 1.955, f'{p:,} param'.replace(',', '.'), ha='center', fontsize=9.5,
            color='white' if c.get_facecolor()[0] < 0.3 else NAVY)
ax.axhline(1.993, color=GOLD, ls='--', lw=1.5)
ax.text(2.33, 1.996, 'Karpathy 1.993', color=GOLD, fontsize=10, ha='right')
ax.set_ylim(1.94, 2.12)
ax.set_ylabel('dev loss')
ax.set_title('Embedding 24, 60.000 adim', color=NAVY, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('karsilastirma.png', dpi=120, bbox_inches='tight')
plt.close()
print("wavenet_agac.png, batchnorm_eksen.png, karsilastirma.png kaydedildi")
