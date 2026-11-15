# YZ50 Full Track: Sıfırdan Sinir Ağları ve GPT

Necabeddin Sivri'nin YZ50 Full Track çalışmaları. Program Andrej Karpathy'nin "Neural Networks: Zero to Hero" serisini takip ediyor: tek bir nörondan başlayıp kendi yazdığım GPT-2 (124M) modelini eğitip HellaSwag ile ölçmeye kadar. Her hafta kendi klasöründe; her klasörde kod, gerçek çalıştırmalardan çıkan sonuçlar ve o haftanın README'si var. Hafta 5 ve sonrasındaki çoğu haftada teknik not (Word) ve sunum (PowerPoint) da var.

## Haftalar

| Hafta | Konu | Klasör |
|---|---|---|
| 1 | Tek nöron: ağırlıklı toplam, bias, sigmoid ve tanh | [hafta-1](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/hafta-1) |
| 2 | micrograd: Value sınıfı, otomatik türev, küçük bir ağı eğitmek | [hafta-2](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/hafta-2) |
| 3 | makemore 1: sayım ile bigram, örnekleme, NLL, sinir ağı ile bigram, Türkçe isimler, trigram | [hafta-3](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/hafta-3) |
| 4 | makemore 2-3: MLP, embedding, başlangıç ağırlıkları, BatchNorm, Türkçe isimler | [Week4](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/Week4) |
| 5 | Backprop Ninja: geri yayılımı elle yazmak ve PyTorch ile doğrulamak | [Week5](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/Week5) |
| 6 | Kendi torch.nn katmanlarım ve WaveNet (bağlam 3'ten 8'e, hiyerarşik birleştirme) | [hafta-6](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/hafta-6) |
| 7 | GPT'ye giriş: bigram dil modeli ve self-attention | [hafta-7](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/hafta-7) |
| 8 | Çok başlı dikkat, residual, LayerNorm, dropout: tam Transformer | [hafta-8](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/hafta-8) |
| 9 | Tokenizer: BPE'yi sıfırdan yazmak, GPT-2 tokenizer ile karşılaştırma | [hafta-9](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/hafta-9) |
| 10 | GPT-2 (124M) modelini kendi kodumla kurmak ve OpenAI ağırlıklarını yüklemek | [hafta-10](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/hafta-10) |
| 11 | Hızlandırma ve GPT-3 eğitim ayarları: veri yükleyici, weight tying, init, bf16, torch.compile, AdamW, LR planı, gradient accumulation | [hafta-11](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/hafta-11) |
| 12 | Final: FineWeb-Edu ile eğitim, doğrulama kaybı, HellaSwag (kendi modelim ve OpenAI GPT-2), Türkçe ek çalışma | [hafta-12](https://github.com/necabeddinsivri-hub/YZ50-Week-1/tree/main/hafta-12) |

## Çalışma ilkeleri

- Tablolardaki ve grafiklerdeki her sayı gerçek bir çalıştırmadan geliyor; tahminlerim çalıştırmadan önce yazıldı ve tutmayanlar da raporlandı.
- Eğitimler GPU olmadan, CPU üzerinde yapıldı; bu yüzden ölçekler Karpathy'nin derslerinden küçük ve her haftanın README'sinde ayrıca belirtiliyor.
- Klasörler kendi başına çalışacak şekilde düzenlendi; gereken küçük veri dosyaları (isim listeleri, tokenizer dosyası) klasörün içinde.
