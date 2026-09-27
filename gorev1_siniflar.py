# Gorev 1 - Dagınık kodu siniflara toplamak
# YZ50 Hafta 6 (1:40 - 17:11)
#
# Hafta 4'teki MLP + BatchNorm, bu sefer tek bir Sequential olarak.
# Egitim dongusu katmanlarin adini bilmiyor: sadece model(x) ve
# model.parameters() cagiriyor.

import torch
from katmanlar import Linear, BatchNorm1d, Tanh, Embedding, Flatten, Sequential
from ortak import bolumle, egit, degerlendir, isim_uret, son_katmani_kucult, parametre_sayisi

BAGLAM, EMB, GIZLI = 3, 10, 200
(Xtr, Ytr), (Xdev, Ydev), _, stoi, itos = bolumle('names.txt', BAGLAM)
V = len(stoi)

torch.manual_seed(42)
model = Sequential([
    Embedding(V, EMB),
    Flatten(),
    Linear(BAGLAM * EMB, GIZLI, bias=False), BatchNorm1d(GIZLI), Tanh(),
    Linear(GIZLI, V),
])
son_katmani_kucult(model)

print("--- model ---")
for katman in model.layers:
    ps = katman.parameters()
    n = sum(p.nelement() for p in ps)
    print(f"  {katman.__class__.__name__:<14} {n:>6} parametre")
print(f"  TOPLAM         {parametre_sayisi(model):>6}")
print()

print("--- egitim (30.000 adim) ---")
kayit = egit(model, Xtr, Ytr, adim=30000)
print()

tr = degerlendir(model, Xtr, Ytr)
dv = degerlendir(model, Xdev, Ydev)
print(f"train loss = {tr:.4f}")
print(f"dev   loss = {dv:.4f}")
print("(Hafta 4'te ayni model, dagınık kodla: 2.1486)")
print()


# --- loss egrisini duzeltmek ---
# Ham minibatch loss'u cok gurultulu, egri 'hokey sopasi' gibi titriyor.
# Cozum: 1000 adimlik bloklarin ortalamasini al.
kayit = torch.tensor(kayit)
duzgun = kayit.view(-1, 1000).mean(1)
print("--- loss egrisi: ham vs 1000'lik blok ortalamasi (log10) ---")
print(f"  ham kayit sayisi     : {len(kayit)}")
print(f"  ham std (gurultu)    : {kayit[-5000:].std().item():.4f}")
print(f"  duzgun nokta sayisi  : {len(duzgun)}")
print("  duzgun egri (her 1000 adim):")
for i in range(0, len(duzgun), 5):
    print(f"    {i*1000:>6}: {10**duzgun[i].item():.4f}")
print()

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(kayit.numpy(), linewidth=0.3, color='gray')
    ax[0].set_title("Ham minibatch loss (log10) - okunmuyor")
    ax[1].plot(duzgun.numpy(), color='navy', linewidth=2)
    ax[1].axvline(22.5, color='goldenrod', linestyle='--', label='lr 0.1 -> 0.01')
    ax[1].set_title("1000 adimlik ortalama (log10) - okunuyor")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('loss_egrisi.png', dpi=110)
    print("loss_egrisi.png kaydedildi")
except ImportError:
    print("(matplotlib yok, grafik atlandi)")
print()

print("--- urettigi isimler ---")
print("  " + ", ".join(isim_uret(model, itos, BAGLAM)))
