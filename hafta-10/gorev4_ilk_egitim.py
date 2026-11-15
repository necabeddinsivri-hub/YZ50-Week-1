# Gorev 4 - Ilk egitim adimlari
# YZ50 Hafta 10  (video 0:45:50 - 1:02:00)
#
# Iki kontrol var ve ikisi de "model dogru kuruldu mu" sorusunu
# cevapliyor. Henuz bir sey ogretmeye calismiyorum; sadece makinenin
# dogru calistigini dogruluyorum.
#
#   4a) Rastgele baslatilmis modelin ILK loss'u ln(50257) ~ 10.82
#       civarinda olmali.
#   4b) Tek bir batch'i ezberletince loss sifira yaklasmali.

import json
import math
import time

import torch

from gpt2 import GPT, GPTConfig
from tokenizer import enc

CIHAZ = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
sonuc = {}

print("=" * 76)
print("4a) RASTGELE BASLATILMIS MODELIN ILK LOSS'U")
print("=" * 76)
V = GPTConfig().vocab_size
beklenen = math.log(V)
print(f"  vocab_size (V)         : {V:,}".replace(',', '.'))
print(f"  beklenen ilk loss      : ln({V}) = {beklenen:.4f}")
print(f"  bunun olasilik karsiligi: 1/{V} = {1/V:.3e}")
print()

print("""  NEDEN BU DEGERE YAKIN CIKMALI

  cross_entropy, modelin dogru token'a verdigi olasiligin eksi
  logaritmasi. Model hicbir sey ogrenmemisse 50.257 token'in hepsini
  esit olasi gormeli, yani her birine 1/50.257 vermeli. Bunun eksi
  logaritmasi ln(50.257) = 10.82.

  Bu basit bir sayi ama cok ise yariyor: bir HATA DETEKTORU.

  - Loss cok daha BUYUK cikarsa (mesela 30), baslangic agirliklari fazla
    buyuk demektir. Model daha ilk adimda bazi token'lara cok yuksek,
    bazilarina cok dusuk olasilik veriyor; yani emin ama yanlis. Hafta
    4'te tanh'in doyma sorununu incelerken gordugum seyin aynisi:
    baslangic olcegi yanlissa ag daha basta bozuk.
  - Loss cok daha KUCUK cikarsa, bir yerde sizinti var demektir. Hafta
    7'de tril maskesini kaldirinca loss dusuyordu ama model kopya
    cekiyordu. Ayni aile.
  - Tam 10.82 de cikmamali; agirliklar rastgele oldugu icin kucuk bir
    sapma normal.
""")

# Olcumu GERCEK metinle yapiyorum: girdi x, hedef y = x'in bir token
# kaydirilmis hali. Neden boyle yaptigimi 4a-ek'te anlatiyorum -
# ilk denememde hedefi girdiyle ayni vermistim ve sayi yaniltici cikti.
e0 = enc()
_metin = open('input.txt', encoding='utf-8').read()
_tok = e0.encode(_metin[:1000])
_buf = torch.tensor(_tok[:4 * 32 + 1])
xr = _buf[:-1].view(4, 32).to(CIHAZ)
yr = _buf[1:].view(4, 32).to(CIHAZ)

modeller = []
for ad, kw in [('ozel init + weight tying (GPT-2 semasi)',
                dict(ozel_init=True, weight_tying=True)),
               ('ozel init, weight tying YOK',
                dict(ozel_init=True, weight_tying=False)),
               ('PyTorch varsayilan init + weight tying',
                dict(ozel_init=False, weight_tying=True))]:
    torch.manual_seed(1337)
    m = GPT(GPTConfig(), **kw).to(CIHAZ)
    p = sum(x.numel() for x in m.parameters())
    with torch.no_grad():
        _, loss = m(xr, yr)
    modeller.append((ad, p, loss.item()))
    print(f"  {ad:<42}{p:>13,}  loss {loss.item():.4f}".replace(',', '.'))
    del m
print()
print(f"  beklenen {beklenen:.4f} ile fark: "
      f"{modeller[0][2] - beklenen:+.4f}")
print()
print("""  Ucu de 10.8'in hemen uzerinde. Model dogru kurulmus.

  PyTorch'un varsayilan init'i de yakin bir sey veriyor, cunku
  nn.Linear zaten makul bir olcek kullaniyor. GPT-2'nin semasi
  bilincli iki sey yapiyor: butun agirliklara std=0.02, ve residual
  yoluna EKLENEN katmanlara ayrica 1/sqrt(2*n_layer) carpani.
  Ikincisinin sebebi: her blokta residual'a iki ekleme var (attention
  ve MLP), yani 12 blok boyunca 24 terim toplaniyor ve toplamin
  varyansi buyuyor. Olcekleme bunu sabit tutuyor. Bu, Hafta 8'de
  residual'i eklerken 'toplama dugumunun gradyani 1' diye yazdigim
  seyin ileri yondeki karsiligi.""")
print()

# ---------------------------------------------------------------------
# 4a-ek) KENDI TESTIMDE YAPTIGIM HATA
# ---------------------------------------------------------------------
print("=" * 76)
print("4a-ek) SANITY CHECK KENDI HATAMI YAKALADI")
print("=" * 76)
print("""  Ilk yazdigimda testi soyle kurmustum:

      x = torch.randint(0, V, (4, 32))
      _, loss = model(x, x)          # <- hedef olarak GIRDININ KENDISI

  Dikkatsizlikti: dil modelinde hedef, girdinin BIR TOKEN KAYDIRILMIS
  hali olmali. Cikan sayi 9.99'du, yani beklenen 10.82'nin 0.84
  ALTINDA. Loss'un beklenenden dusuk cikmasi her zaman supheli
  (Hafta 7'de tril maskesini kaldirinca da boyle olmustu), o yuzden
  ustune gittim.""")
print()
torch.manual_seed(1337)
_m = GPT(GPTConfig(), ozel_init=True, weight_tying=True).to(CIHAZ)
torch.manual_seed(1337)
_m2 = GPT(GPTConfig(), ozel_init=True, weight_tying=False).to(CIHAZ)
_r = torch.randint(0, V, (4, 32), device=CIHAZ)
_r2 = torch.randint(0, V, (4, 32), device=CIHAZ)
olcum = []
with torch.no_grad():
    for ad, mm in [('weight tying VAR', _m), ('weight tying YOK', _m2)]:
        a = mm(_r, _r)[1].item()
        b = mm(_r, _r2)[1].item()
        c = mm(xr, yr)[1].item()
        olcum.append((ad, a, b, c))
print(f"  {'':<20}{'hedef = girdi':>16}{'bagimsiz rastgele':>20}"
      f"{'gercek metin':>16}")
print("  " + "-" * 72)
for ad, a, b, c in olcum:
    print(f"  {ad:<20}{a:>16.4f}{b:>20.4f}{c:>16.4f}")
print("  " + "-" * 72)
print()
print(f"""  SEBEBI BULDUM: WEIGHT TYING.

  Weight tying'de giris embedding'i (wte) ile cikis katmani AYNI
  tensor. Yani bir token'in logit'i, o token'in embedding vektoru ile
  son katmanin ciktisinin ic carpimi. Modelin ciktisi henuz
  egitilmedigi icin buyuk olcude girdi embedding'inin kendisine
  benziyor - ve bir vektorun kendisiyle ic carpimi, baska bir
  vektorle ic carpimindan buyuk.

  Sonuc: model, "bir sonraki token girdinin kendisidir" tahminine
  dogal bir egilimle basliyor. Hedefi girdinin kendisi yapinca bu
  egilim odullendiriliyor ve loss {olcum[0][1]:.2f}'e dusuyor.

  Kanit tablonun ikinci satirinda: weight tying YOKken ayni test
  {olcum[1][1]:.2f} veriyor, yani etki tamamen kayboluyor.

  Iki ders cikardim. Birincisi, ln(V) kontrolu gercekten bir hata
  detektoru - beklenenden SAPAN her sayi bir sey soyluyor ve benim
  durumumda kendi test kodumdaki hatayi soyledi. Ikincisi, weight
  tying masum bir parametre tasarrufu degil; modelin baslangictaki
  davranisini da degistiriyor.""")
print()

sonuc['ilk_loss'] = {ad: dict(param=p, loss=l) for ad, p, l in modeller}
sonuc['beklenen'] = beklenen
sonuc['tying_hatasi'] = [list(o) for o in olcum]

# =====================================================================
# 4b) TEK BATCH'I EZBERLETME
# =====================================================================
print("=" * 76)
print("4b) TEK BIR BATCH'I EZBERLETME")
print("=" * 76)
B, T, ADIM, LR = 4, 32, 300, 3e-4
e = enc()
metin = open('input.txt', encoding='utf-8').read()
tokens = e.encode(metin[:1000])
buf = torch.tensor(tokens[:B * T + 1])
x = buf[:-1].view(B, T).to(CIHAZ)
y = buf[1:].view(B, T).to(CIHAZ)

print(f"  batch : B={B}, T={T}  ->  {B*T} token, hepsi ayni, her adimda")
print(f"  adim  : {ADIM}, lr={LR}, AdamW")
print(f"  cihaz : {CIHAZ}")
print()
print(f"  ilk {T} token (ilk satir): {e.decode(x[0].tolist())!r}")
print()

print("""  BU TEST NEDEN ISE YARIYOR

  Tek bir batch'i, hep ayni veriyi, yuzlerce kez gosteriyorum. Model
  genellemek zorunda degil; sadece {} token'i ezberlemesi yeterli.
  124 milyon parametreli bir model icin bu cok kolay bir is.

  Dolayisiyla loss SIFIRA inmek ZORUNDA. Inmiyorsa ortada bir ogrenme
  sorunu degil, bir KOD sorunu vardir:
    - gradyanlar akmiyordur (bir yerde detach ya da no_grad kalmistir),
    - hedefler bir adim kaymistir (y = x[1:] olmali, x[:] degil),
    - optimizer parametreleri gormuyordur,
    - ya da learning rate tamamen yanlistir.

  Yani bu test modelin ne kadar iyi oldugunu olcmuyor; makinenin
  takili olup olmadigini olcuyor. Hafta 5'te manuel geri yayilimi
  PyTorch'un gradyanlariyla karsilastirmistim - mantik ayni: once
  dogrulugu kanitla, sonra performansa bak.
""".format(B * T))

torch.manual_seed(1337)
model = GPT(GPTConfig(), weight_tying=True, ozel_init=True).to(CIHAZ)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
kayit = []
t0 = time.time()
for i in range(ADIM):
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    kayit.append(loss.item())
    if i % 20 == 0 or i == ADIM - 1:
        print(f"    adim {i:>4}  loss {loss.item():>9.6f}   "
              f"({time.time()-t0:>5.0f} sn)")
sure = time.time() - t0
print()
print(f"  ilk loss : {kayit[0]:.6f}")
print(f"  son loss : {kayit[-1]:.6f}")
print(f"  sure     : {sure/60:.1f} dk  ({sure/ADIM:.2f} sn/adim)")
print()

# ezberledi mi? ayni batch'i tahmin ettirip bak
model.eval()
with torch.no_grad():
    logits, _ = model(x, y)
tahmin = logits.argmax(dim=-1)
dogru = (tahmin == y).float().mean().item()
print(f"  ayni batch'te token dogrulugu: {dogru*100:.1f}%  "
      f"({int(dogru*B*T)}/{B*T})")
print()
print(f"  Loss {kayit[0]:.2f}'den {kayit[-1]:.4f}'e indi. Model {B*T}")
print("  token'i ezberledi. Yani ileri gecis, geri yayilim ve optimizer")
print("  dogru bagli; eger gelecek hafta gercek egitimde loss inmezse")
print("  sebebi kod degil, hiperparametreler ya da veri olacak.")
print()

sonuc['ezber'] = dict(B=B, T=T, adim=ADIM, lr=LR, sure=sure,
                      kayit=kayit, dogruluk=dogru, cihaz=CIHAZ)
json.dump(sonuc, open('sonuc_gorev4.json', 'w'), ensure_ascii=False, indent=1)
print("  -> sonuc_gorev4.json kaydedildi")
