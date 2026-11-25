# ===================== 5. SAN: Script-Aware Normalization =====================
import unicodedata

INVISIBLE = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD}
BIDI = {0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069, 0x200E, 0x200F}

# Kirill ichidagi lotinga o'xshash harflar (lotinga qaytarish uchun)
CYR_FOLD = dict(CYR2LAT_HG)
CYR_FOLD.update({"ѕ": "s", "ј": "j", "і": "i", "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w",
                 "Ѕ": "S", "Ј": "J", "І": "I"})
# Lotin ichidagi kirillga o'xshash harflar (kirillga qaytarish uchun)
LAT_FOLD = {k: v for k, v in LAT2CYR_HG.items() if v not in ("і", "ѕ", "ј")}

# Til profillari: qonuniy kirill harflari (o'zbek + rus); qozoq uchun "kk" qo'shiladi
LEGIT_CYR = {"uz_ru": set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяўқғҳ"),
             "kk": set("әғқңөұүһі")}
UZ_ONLY_CYR = set("ўқғҳ")

def _script(c):
    n = unicodedata.name(c, "")
    return "L" if "LATIN" in n else "C" if "CYRILLIC" in n else None

def decode_hidden(text):
    """Ko'rinmas kanallarni oshkor qiladi: Unicode tag va variation selector ichidagi matn
    dekodlanadi (LLM uni o'qiydi, shuning uchun detektor ham ko'rishi kerak), bidi hal qilinadi,
    zero-width belgilar olib tashlanadi. Statistika qaytariladi."""
    st = {"tag": 0, "vs_run": 0, "bidi_override": 0, "zw": 0}
    # 1) bidi override: RLO ... PDF ichini teskari qilib qo'yish
    out, i = [], 0
    while i < len(text):
        if text[i] == "‮":
            j = text.find("‬", i + 1)
            j = len(text) if j == -1 else j
            out.append(text[i+1:j][::-1]); st["bidi_override"] += 1; i = j + 1
        else:
            out.append(text[i]); i += 1
    text = "".join(out)
    # 2) tag, variation selector, zero-width, qolgan bidi
    res, run = [], []
    def flush():
        if len(run) >= 2:
            try:
                res.append(bytes(run).decode("utf-8", errors="ignore")); st["vs_run"] += 1
            except Exception:
                pass
        run.clear()
    for ch in text:
        cp = ord(ch)
        if 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF:
            run.append(cp - 0xFE00 if cp <= 0xFE0F else cp - 0xE0100 + 16); continue
        flush()
        if 0xE0020 <= cp <= 0xE007E:
            res.append(chr(cp - 0xE0000)); st["tag"] += 1
        elif 0xE0000 <= cp <= 0xE007F:
            continue
        elif cp in INVISIBLE:
            st["zw"] += 1
        elif cp in BIDI:
            continue
        else:
            res.append(ch)
    flush()
    return "".join(res), st

def _dominant(letters):
    """Ishonchli (o'xshashi yo'q) harflar soni bo'yicha asosiy yozuvni aniqlash."""
    lat = sum(1 for c in letters if _script(c) == "L" and c not in LAT_FOLD)
    cyr = sum(1 for c in letters if _script(c) == "C" and c not in CYR_FOLD)
    return "L" if lat > cyr else "C" if cyr > lat else None

def fold_word(w, sent_dom, profile=("uz_ru",)):
    legit = set().union(*[LEGIT_CYR[p] for p in profile])
    letters = [c for c in w if c.isalpha()]
    scr = {_script(c) for c in letters} - {None}
    lower = [c.lower() for c in letters]
    uz_cyr = any(c in UZ_ONLY_CYR for c in lower)     # ў қ ғ ҳ: shubhasiz o'zbek kirilli -> lotinga qaytarilmaydi
    if scr == {"C"}:
        if uz_cyr:
            return w, False
        foreign = [c for c in lower if c not in legit]
        if foreign and all(c in CYR_FOLD for c in letters if c.lower() not in legit):
            return "".join(CYR_FOLD.get(c, c) for c in w), True        # і, ѕ, ј ...
        if sent_dom == "L" and len(letters) >= 3 and all(c in CYR_FOLD for c in letters):
            return "".join(CYR_FOLD.get(c, c) for c in w), True        # butunlay soxta so'z
        return w, False
    if scr == {"L", "C"}:
        dom = "C" if uz_cyr else (_dominant(letters) or sent_dom)
        if dom == "L":
            minority = [c for c in letters if _script(c) == "C"]
            if all(c in CYR_FOLD for c in minority):
                return "".join(CYR_FOLD.get(c, c) for c in w), True
        elif dom == "C":
            minority = [c for c in letters if _script(c) == "L"]
            if all(c in LAT_FOLD for c in minority):
                return "".join(LAT_FOLD.get(c, c) for c in w), True
        return w, False                                  # masalan: Telegramʼда (brend + qo'shimcha)
    return w, False

def san(text, profile=("uz_ru",), obf_min=1):
    """SAN: (1) yashirin kanallarni dekodlash, (2) NFKC, (3) apostrof, (4) so'z darajasida
    yozuvni hisobga olgan gomoglif tuzatish. Qaytaradi: kanonik matn, obfuskatsiya signali, statistika."""
    t, st = decode_hidden(text)
    t2 = unicodedata.normalize("NFKC", t)
    st["nfkc"] = sum(1 for a, b in zip(t, t2) if a != b) + abs(len(t) - len(t2))
    words = t2.split(" ")
    sent_dom = _dominant([c for c in t2 if c.isalpha()])
    folded, nf = [], 0
    for w in words:
        fw, f = fold_word(w, sent_dom, profile)
        folded.append(fw); nf += f
    st["fold"] = nf
    canon = norm_apos(" ".join(folded))      # apostrof gomoglif tuzatilgandan keyin
    obf = ((st["tag"] > 0) + (st["vs_run"] > 0) + (st["bidi_override"] > 0) +
           (st["zw"] >= 3) + (nf >= 2) + (st["nfkc"] >= 5))
    st["obf"] = obf
    return canon, int(obf >= obf_min), st

# ----- Baseline normalizatsiyalar (N0–N4) -----
def n0(t): return t
def n1(t): return unicodedata.normalize("NFKC", t)
def n2(t):   # NFKC + ko'rinmaslarni olib tashlash (dekodlamasdan)
    t = n1(t)
    return "".join(c for c in t if ord(c) not in INVISIBLE and ord(c) not in BIDI
                   and not (0xE0000 <= ord(c) <= 0xE01EF) and not (0xFE00 <= ord(c) <= 0xFE0F))
def n3(t):   # N2 + barcha kirill o'xshashlarini lotinga (UTS #39 skeleton uslubi, yozuvni hisobga olmaydi)
    return "".join(CYR_FOLD.get(c, c) for c in n2(t))
def mixed_word_flag(t):   # N4 uchun: biror so'zda lotin+kirill aralashsa -> bloklash
    return int(any({_script(c) for c in w if c.isalpha()} - {None} == {"L", "C"} for w in t.split()))

# ----- SAN-T: rasmiy o'zbek kirill -> lotin transliteratsiyasi (yozuvni detektor "tushunadigan" ko'rinishga keltirish) -----
C2L = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ё": "yo", "ж": "j", "з": "z", "и": "i",
       "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
       "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
       "ъ": TUTUQ, "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya", "ў": "o" + OKINA,
       "қ": "q", "ғ": "g" + OKINA, "ҳ": "h"}
CYR_VOWELS = set("аеёиоуэюяў")

def cyr2lat_uz(word):
    out = []
    for i, ch in enumerate(word):
        lo = ch.lower()
        if lo == "е":
            prev = word[i-1].lower() if i > 0 else ""
            rep = "ye" if (i == 0 or not prev.isalpha() or prev in CYR_VOWELS or prev in "ъь") else "e"
        elif lo in C2L:
            rep = C2L[lo]
        else:
            out.append(ch); continue
        if ch.isupper() and rep:
            nb = [c for c in (word[i-1:i] + word[i+1:i+2]) if c.isalpha()]
            all_up = bool(nb) and all(c.isupper() for c in nb)
            rep = rep.upper() if all_up else rep[0].upper() + rep[1:]
        out.append(rep)
    return "".join(out)

def san_t(text, profile=("uz_ru",), obf_min=1):
    """SAN + qonuniy o'zbekcha kirill so'zlarni rasmiy lotin yozuviga o'tkazish.
    Hujumchi injection'ni kirillda yozsa ham, detektor uni lotin ko'rinishida baholaydi."""
    canon, obf, st = san(text, profile, obf_min)
    words = canon.split(" ")
    st["translit"] = sum(any(_script(c) == "C" for c in w) for w in words)
    return " ".join(cyr2lat_uz(w) for w in words), obf, st
