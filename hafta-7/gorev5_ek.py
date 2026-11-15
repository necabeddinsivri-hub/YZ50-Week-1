# Gorev 5 (ek) - (a) attention isi haritasi  +  (b) tril maskesini kaldirmak
# YZ50 Hafta 7

import torch
from ortak import encode, decode, loss_olc, egit, BLOCK
from model import AttentionDilModeli

# ==========================================================
# (a) EGITILMIS MODELDE BIR CUMLENIN ATTENTION AGIRLIKLARI
# ==========================================================
print("=" * 66)
print("(a) ATTENTION ISI HARITASI")
print("=" * 66)
torch.manual_seed(1337)
model = AttentionDilModeli()
egit(model, adim=5000, lr=1e-3, sessiz=True)
s = loss_olc(model)
print(f"  egitilmis model: val {s['val']:.4f}")
print()

cumle = "First Ci"          # tam olarak block_size = 8 karakter
idx = torch.tensor([encode(cumle)])
model.eval()
with torch.no_grad():
    model(idx)
wei = model.sa_head.son_wei[0]       # (8, 8)

print(f"  cumle: {cumle!r}")
print()
print("      " + "".join(f"{c:>7}" for c in cumle))
for i, c in enumerate(cumle):
    satir = "".join(f"{wei[i, j].item():>7.3f}" if j <= i else f"{'·':>7}" for j in range(8))
    print(f"  {c!r:>4}{satir}")
print()
print("  (satirlar: 'ben hangi token'im', sutunlar: 'kime bakiyorum')")
print("  (· = gelecek, tril maskesiyle kapali)")
print()

# metin tabanli isi haritasi
print("  Isi haritasi (karakter yogunlugu agirligi gosteriyor):")
semalar = " .:-=+*#%@"
print("      " + "".join(f"{c:>3}" for c in cumle))
for i, c in enumerate(cumle):
    satir = ""
    for j in range(8):
        if j > i:
            satir += "   "
        else:
            k = int(wei[i, j].item() * (len(semalar) - 1) / max(wei[i].max().item(), 1e-9))
            satir += f"  {semalar[k]}"
    print(f"  {c!r:>4}{satir}")
print()

for satir_no in [3, 7]:
    w = wei[satir_no, :satir_no + 1]
    en = w.argmax().item()
    print(f"  {satir_no}. token ({cumle[satir_no]!r}) en cok {en}. token'a bakiyor "
          f"({cumle[en]!r}, agirlik {w[en]:.3f}); duz ortalama olsaydi {1/(satir_no+1):.3f} olurdu")
print()

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(wei.numpy(), cmap='Blues', vmin=0, vmax=wei.max().item())
    ax.set_xticks(range(8), list(cumle)); ax.set_yticks(range(8), list(cumle))
    ax.set_xlabel("kime bakiyor"); ax.set_ylabel("hangi token")
    for i in range(8):
        for j in range(8):
            if j <= i:
                ax.text(j, i, f"{wei[i, j]:.2f}", ha='center', va='center', fontsize=8,
                        color='white' if wei[i, j] > wei.max() * 0.6 else '#00285A')
    ax.set_title("Attention agirliklari — 'First Ci'", color='#00285A')
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout(); plt.savefig('attention_isi.png', dpi=120); plt.close()
    print("  attention_isi.png kaydedildi")
except ImportError:
    print("  (matplotlib yok)")
print()


# ==========================================================
# (b) TRIL MASKESINI KALDIRMAK
# ==========================================================
print("=" * 66)
print("(b) TRIL MASKESI KALDIRILIRSA")
print("=" * 66)
torch.manual_seed(1337)
maskesiz = AttentionDilModeli(maske=False)
egit(maskesiz, adim=5000, lr=1e-3, sessiz=True)
s2 = loss_olc(maskesiz)
print(f"  maskeli   : val {s['val']:.4f}")
print(f"  maskesiz  : val {s2['val']:.4f}")
print(f"  fark      : {s['val'] - s2['val']:+.4f}")
print()

gg = torch.Generator().manual_seed(42)
print("  maskesiz modelin urettigi 250 karakter:")
print(decode(maskesiz.generate(torch.zeros((1, 1), dtype=torch.long), 250, g=gg)[0].tolist()))
print()

print("""  YORUM

  Maskesiz model egitimde DAHA DUSUK loss aliyor. Sasirtici degil:
  7. token'in hedefi 8. karakter ve maskesiz modelde 7. token butun
  parcayi, yani 8. karakteri de goruyor. Cevabi gormek kolay.

  Ama bu loss yalan. Model sinavda kopya cekiyor.

  Uretim zamaninda gelecek diye bir sey yok: her adimda sadece
  o ana kadar uretilmis karakterler var. Yani model egitimde
  alistigi bilgiden yoksun kaliyor ve urettigi metin bozuluyor.

  Bu, Hafta 6'da train/dev/test ayriminda ogrendigim seyin baska
  bir hali: olcumu bozan bir sizinti varsa dusen loss ilerleme
  anlamina gelmiyor.

  tril maskesi tam olarak bu sizintiyi kapatiyor. 'Decoder-only'
  mimari denmesinin sebebi de bu. Maske kaldirilirsa model bir
  dil modeli olmaktan cikip BERT tarzi bir encoder'a donusuyor;
  o da metin uretmek icin degil, metni anlamak icin kullaniliyor.
""")
