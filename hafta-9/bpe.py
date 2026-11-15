# bpe.py - Byte Pair Encoding, sifirdan
# YZ50 Hafta 9
#
# Uc sinif var, her biri bir oncekinin uzerine biniyor:
#   BasicTokenizer  - ham byte dizisi uzerinde BPE (0:23:50 - 0:57:36)
#   RegexTokenizer  - once GPT-4'un regex deseniyle bol, sonra BPE (0:57:36)
#   GPT4Tokenizer   - GPT-4'un hazir birlestirmelerini yukleyip kullan (1:11:38)
#
# Temel fikir cok basit: metni byte'lara cevir (0-255 arasi sayilar),
# en sik gecen ardisik IKILIYI bul, ona yeni bir numara ver (256, 257, ...),
# metinde o ikiliyi tek sayiyla degistir. Istenen vocab boyutuna gelene
# kadar tekrarla.

import regex as re


# =====================================================================
# TEMEL YARDIMCILAR
# =====================================================================
def istatistik(ids, sayac=None):
    """Ardisik ikililerin kac kere gectigini sayar.

    [1,2,3,1,2] -> {(1,2): 2, (2,3): 1, (3,1): 1}

    sayac verilirse uzerine ekler. Boylece birden fazla parcanin
    sayimlarini tek sozlukte toplayabiliyorum (RegexTokenizer icin).
    """
    sayac = {} if sayac is None else sayac
    for ikili in zip(ids, ids[1:]):
        sayac[ikili] = sayac.get(ikili, 0) + 1
    return sayac


def birlestir(ids, ikili, yeni):
    """ids icindeki her 'ikili' gecisini 'yeni' ile degistirir.

    [1,2,3,1,2], (1,2), 99 -> [99,3,99]
    """
    cikti = []
    i = 0
    while i < len(ids):
        # i ve i+1 birlikte aradigim ikiliyi olusturuyor mu?
        if i < len(ids) - 1 and ids[i] == ikili[0] and ids[i + 1] == ikili[1]:
            cikti.append(yeni)
            i += 2
        else:
            cikti.append(ids[i])
            i += 1
    return cikti


def goster(b):
    """Byte dizisini okunabilir hale getirir (yazdirilamayan karakterleri kacar)."""
    s = b.decode('utf-8', errors='replace')
    return ''.join(ch if ch.isprintable() or ch == ' ' else
                   '\\u{:04x}'.format(ord(ch)) for ch in s)


# =====================================================================
# 1) TEMEL TOKENIZER - ham byte dizisi uzerinde
# =====================================================================
class BasicTokenizer:
    def __init__(self):
        self.merges = {}        # (int, int) -> int
        self.vocab = {}         # int -> bytes
        self._vocab_kur()

    def _vocab_kur(self):
        # 0-255: tek byte'lar. Her metin bunlarla ifade edilebilir,
        # yani "bilinmeyen karakter" diye bir sorun hic olmuyor.
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), yeni in self.merges.items():
            self.vocab[yeni] = self.vocab[a] + self.vocab[b]

    def train(self, metin, vocab_size, sessiz=True, kayit=None):
        assert vocab_size >= 256
        n = vocab_size - 256
        ids = list(metin.encode('utf-8'))
        self.merges = {}
        for i in range(n):
            sayac = istatistik(ids)
            if not sayac:
                break
            ikili = max(sayac, key=sayac.get)      # en sik gecen ikili
            yeni = 256 + i
            adet = sayac[ikili]
            ids = birlestir(ids, ikili, yeni)
            self.merges[ikili] = yeni
            if kayit is not None:
                kayit.append(dict(sira=i + 1, ikili=ikili, yeni=yeni,
                                  adet=adet, kalan=len(ids)))
            if not sessiz and i < 20:
                print(f"    {i+1:>3}. birlestirme: {ikili} -> {yeni}"
                      f"   ({adet} kere)")
        self._vocab_kur()
        return self

    def decode(self, ids):
        """token id'lerden metne.

        errors='replace' ONEMLI: her byte dizisi gecerli UTF-8 degil.
        Gorev 3'te bunu ayrica tartisiyorum.
        """
        b = b''.join(self.vocab[i] for i in ids)
        return b.decode('utf-8', errors='replace')

    def encode(self, metin):
        """metinden token id'lere."""
        ids = list(metin.encode('utf-8'))
        return self._encode_ids(ids)

    def _encode_ids(self, ids):
        while len(ids) >= 2:
            sayac = istatistik(ids)
            # EN ERKEN ogrenilmis (en kucuk numarali) birlestirmeyi sec.
            # Sira onemli: egitimdeki sirayi birebir tekrar etmeliyim,
            # yoksa ayni metin baska token'lara ayrilir.
            ikili = min(sayac, key=lambda p: self.merges.get(p, float('inf')))
            if ikili not in self.merges:
                break                      # yapilacak birlestirme kalmadi
            ids = birlestir(ids, ikili, self.merges[ikili])
        return ids


# =====================================================================
# 2) REGEX TOKENIZER - once bol, sonra BPE
# =====================================================================
# GPT-4'un deseni (cl100k_base). Ne yaptigi:
#   (1) Ingilizce kisaltma ekleri ('s, 't, 're, 've, 'm, 'll, 'd)
#   (2) (opsiyonel bir harf-disi karakter +) harf dizisi
#   (3) en fazla 3 basamakli sayi
#   (4) (opsiyonel bosluk +) noktalama dizisi
#   (5) satir sonlari ve bosluklar
GPT4_DESEN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|"""
              r"""\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|"""
              r"""\s+(?!\S)|\s+""")

# GPT-2'nin deseni, karsilastirma icin
GPT2_DESEN = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| """
              r"""?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class RegexTokenizer(BasicTokenizer):
    """Metni once regex ile parcalara boler, BPE'yi HER PARCADA ayri calistirir.

    Neden: ham BPE'de 'kedi' ve 'kedi.' farkli yerlerde farkli birlesebilir;
    daha kotusu, 'kopek' ve ' kopek' ve 'kopek,' ayri ayri ogrenilir ve
    vocab noktalama cesitleriyle sisip gider. Regex once kelimeyi
    noktalamadan ayiriyor, BPE sadece kelimenin icinde calisiyor.
    Yani bir token asla kelime sinirini gecip noktalamayi icine almiyor.
    """

    def __init__(self, desen=None):
        self.desen = desen or GPT4_DESEN
        self._re = re.compile(self.desen)
        super().__init__()

    def train(self, metin, vocab_size, sessiz=True, kayit=None):
        assert vocab_size >= 256
        n = vocab_size - 256
        parcalar = self._re.findall(metin)
        ids_listesi = [list(p.encode('utf-8')) for p in parcalar]
        self.merges = {}
        for i in range(n):
            sayac = {}
            for ids in ids_listesi:
                istatistik(ids, sayac)       # sayimlar tek sozlukte toplaniyor
            if not sayac:
                break
            ikili = max(sayac, key=sayac.get)
            yeni = 256 + i
            adet = sayac[ikili]
            ids_listesi = [birlestir(ids, ikili, yeni) for ids in ids_listesi]
            self.merges[ikili] = yeni
            if kayit is not None:
                kayit.append(dict(sira=i + 1, ikili=ikili, yeni=yeni,
                                  adet=adet,
                                  kalan=sum(len(x) for x in ids_listesi)))
            if not sessiz and i < 20:
                print(f"    {i+1:>3}. birlestirme: {ikili} -> {yeni}"
                      f"   ({adet} kere)")
        self._vocab_kur()
        return self

    def encode(self, metin):
        ids = []
        for parca in self._re.findall(metin):
            ids.extend(self._encode_ids(list(parca.encode('utf-8'))))
        return ids


# =====================================================================
# 3) GPT4 TOKENIZER - GPT-4'un kendi birlestirmeleriyle
# =====================================================================
def _bpe_coz(ranks, parcalar, max_rank):
    """Verilen parcalari, rank'i max_rank'ten KUCUK olan birlestirmeleri
    uygulayarak birlestirir. max_rank'e gelince durur.

    Bu, 'bu token olusmadan hemen onceki durum neydi' sorusunun cevabi.
    """
    while True:
        if len(parcalar) == 1:
            break
        en_iyi, en_iyi_rank = None, None
        for i in range(len(parcalar) - 1):
            r = ranks.get(parcalar[i] + parcalar[i + 1])
            if r is not None and (en_iyi_rank is None or r < en_iyi_rank):
                en_iyi, en_iyi_rank = i, r
        if en_iyi is None or (max_rank is not None and en_iyi_rank >= max_rank):
            break
        parcalar = (parcalar[:en_iyi]
                    + [parcalar[en_iyi] + parcalar[en_iyi + 1]]
                    + parcalar[en_iyi + 2:])
    return parcalar


def gpt4_merges(mergeable_ranks):
    """tiktoken'in rank sozlugunden (bytes->int) merges sozlugu uretir.

    tiktoken sadece 'token -> rank' veriyor, birlestirme agacini vermiyor.
    Agaci geri kurmak icin her token'i tek tek byte'larina ayirip, SADECE
    o token'dan once gelmis birlestirmeleri uyguluyorum. Geriye tam iki
    parca kaliyor: o token'i olusturan ikili.

    Ilk denememde 'ilk gecerli bolme noktasini al' demistim; yanlisti ve
    KeyError ile cokuyordu. Sebep: bir token birden fazla yerden
    bolunebiliyor ve dogru olani her zaman ilki degil.
    """
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue                      # 0-255 taban token'lari
        parcalar = _bpe_coz(ranks=mergeable_ranks,
                            parcalar=[bytes([b]) for b in token],
                            max_rank=rank)
        assert len(parcalar) == 2, (token, parcalar)
        sol, sag = parcalar
        merges[(mergeable_ranks[sol], mergeable_ranks[sag])] = rank
    return merges


class GPT4Tokenizer(RegexTokenizer):
    """GPT-4'un birlestirmelerini yukler; tiktoken ile ayni token'lari uretmeli.

    Bir puruz var: GPT-4, 0-255 taban byte'larini KARISTIRIYOR. Yani
    tiktoken'de 0 numarali token b'!' , 1 numarali b'"' ... seklinde.
    O yuzden encode'dan once byte'lari bu siraya cevirip, decode'da geri
    aliyorum (byte_shuffle).
    """

    def __init__(self, mergeable_ranks, desen=None):
        super().__init__(desen)
        self.merges = gpt4_merges(mergeable_ranks)
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.ters_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        self.vocab = {i: bytes([self.ters_shuffle[i]]) for i in range(256)}
        for (a, b), yeni in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[yeni] = self.vocab[a] + self.vocab[b]

    def encode(self, metin):
        ids = []
        for parca in self._re.findall(metin):
            b = parca.encode('utf-8')
            karisik = [self.byte_shuffle[x] for x in b]
            ids.extend(self._encode_ids(karisik))
        return ids

    def train(self, *a, **k):
        raise NotImplementedError('GPT4Tokenizer egitilmez, yuklenir')
