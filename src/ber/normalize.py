"""Text normalisation for business names and addresses.

All rules are generic, language-level canonicalisations (legal-form and street-type
abbreviations, accent folding, transliteration). No external lookup of any business
identity is performed. An optional token dictionary learned *from the training
pairs only* (see translit.py) maps Indic-script tokens to their Latin equivalents.
"""
import re
import unicodedata

from unidecode import unidecode

INDIC_RE = re.compile(r"[\u0900-\u0DFF]")
INDIC_KEY_RE = re.compile(r"[^\u0900-\u0DFF]")
NONWORD_RE = re.compile(r"[^a-z0-9 ]+")
SPACE_RE = re.compile(r"\s+")
WEB_TOKEN_RE = re.compile(r"\S*www\.\S*|\S+\.(?:com|net|org|in|co|biz|info|fr|us)\b\S*")
DIGITS_RE = re.compile(r"\d+")
WEB_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|co|biz|info|fr|us)\b")

# Canonical legal forms (US / India / France and generic).
LEGAL_MAP = {
    "private": "pvt", "pvt": "pvt", "pvte": "pvt", "pte": "pvt", "prvt": "pvt", "priv": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd", "limitd": "ltd",
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "cos": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "pa": "pa",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "ei": "ei",
    "sci": "sci", "snc": "snc", "scop": "scop", "gmbh": "gmbh",
}
# multi-token legal forms collapsed before token mapping
LEGAL_PHRASES = [
    (re.compile(r"\bl\s?l\s?c\b"), " llc "), (re.compile(r"\bl\s?l\s?p\b"), " llp "),
    (re.compile(r"\bp\s?l\s?c\b"), " plc "), (re.compile(r"\bs\s?a\s?r\s?l\b"), " sarl "),
    (re.compile(r"\bs\s?a\s?s\b"), " sas "), (re.compile(r"\be\s?u\s?r\s?l\b"), " eurl "),
    (re.compile(r"\bpvt\s?ltd\b"), " pvt ltd "),
]
HONORIFICS = {"sri", "shri", "shree", "smt", "m s", "ms", "messrs", "mr", "mrs", "dr", "the", "and", "of",
              "et", "de", "la", "le", "les", "du", "des"}
LEGAL_TOKENS = set(LEGAL_MAP.values())

ADDR_MAP = {
    "road": "rd", "rd": "rd", "street": "st", "st": "st", "str": "st", "avenue": "ave", "ave": "ave",
    "av": "ave", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd", "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "place": "pl", "pl": "pl", "circle": "cir",
    "cir": "cir", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy", "terrace": "ter",
    "ter": "ter", "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq", "suite": "ste", "ste": "ste",
    "apartment": "apt", "apt": "apt", "floor": "fl", "fl": "fl", "building": "bldg", "bldg": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "near": "nr", "nr": "nr", "opposite": "opp",
    "opp": "opp", "number": "no", "no": "no", "nos": "no", "house": "h", "sector": "sec", "sec": "sec",
    "nagar": "ngr", "ngr": "ngr", "colony": "col", "col": "col", "marg": "marg", "chowk": "chk",
    "rue": "rue", "r": "rue", "chemin": "ch", "ch": "ch", "route": "rte", "rte": "rte", "allee": "all",
    "impasse": "imp", "imp": "imp", "cours": "crs", "quai": "qu", "faubourg": "fbg", "fbg": "fbg",
    "saint": "st", "sainte": "ste", "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "point": "pt",
    "pt": "pt", "unit": "unit", "po": "po", "box": "box", "block": "blk", "blk": "blk",
}
ADDR_STOP = {"no", "nr", "opp", "h", "the", "of", "de", "du", "des", "la", "le", "near", "c", "o", "at", "post",
             "po", "dist", "district", "tal", "taluka", "via", "na", "n a"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "ladakh": "la",
}
_STATES = {**US_STATES, **IN_STATES}
STATE_RE = re.compile(r"\b(" + "|".join(sorted(_STATES, key=len, reverse=True)) + r")\b")

_translit = {}


def set_translit_dict(d):
    """Install a learned Indic-token -> Latin-token dictionary (from training pairs)."""
    global _translit
    _translit = dict(d)


def fold(s):
    """Unicode NFKC, learned Indic token mapping, accent/script folding, lowercase."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    if not s.isascii():
        if _translit and INDIC_RE.search(s):
            s = " ".join(_translit.get(INDIC_KEY_RE.sub("", t), t) if INDIC_RE.search(t) else t
                         for t in s.split())
        s = unidecode(s)
    return s.lower()


def basic(s):
    """fold + '&'->and + punctuation removal (keeps digits/letters), collapsed spaces."""
    s = fold(s).replace("&", " and ").replace("'", "")
    s = NONWORD_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


def norm_name(raw):
    """Return (canonical name string, core tokens list, legal tokens list, web flag)."""
    f = fold(raw).strip()
    web = 0
    if "." in f and " " in f:  # drop embedded web domains ("Acme Ltd | www.acme.com")
        g = WEB_TOKEN_RE.sub(" ", f).replace("|", " ").strip()
        if g:
            f, web = g, 2
    m = WEB_RE.match(f.replace(" ", "")) if "." in f else None
    if m and " " not in f.strip():
        f = m.group(1)
        web = 1
    s = f.replace("&", " and ").replace("'", "")
    s = s.replace(".", "")  # "p.v.t." -> "pvt", "l.l.c" -> "llc"
    s = NONWORD_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    for rx, rep in LEGAL_PHRASES:
        s = rx.sub(rep, s)
    toks = s.split()
    core, legal = [], []
    for t in toks:
        c = LEGAL_MAP.get(t)
        if c is not None:
            legal.append(c)
        elif t in HONORIFICS:
            continue
        else:
            core.append(t)
    canon = " ".join(core + legal)
    return canon, core, legal, web


def norm_addr(raw):
    """Return (canonical address string, tokens list, numeric tokens list)."""
    s = fold(raw.replace("°", " ")).replace("#", " ").replace("#", " ").replace("'", "")
    s = s.replace(".", " ")
    s = NONWORD_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    if not s:
        return "", [], []
    s = STATE_RE.sub(lambda m: _STATES[m.group(1)], s)
    out, nums = [], []
    for t in s.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ADDR_MAP.get(t, t)
        out.append(t)
        if not t.isalpha():
            nums.extend(d.lstrip("0") or "0" for d in DIGITS_RE.findall(t))
    return " ".join(out), out, nums


_PHON_SUBS = [("ph", "f"), ("sh", "s"), ("ch", "k"), ("x", "ks"), ("q", "k"), ("c", "k"), ("w", "v"),
              ("z", "j"), ("h", "")]
_VOWELS_RE = re.compile(r"[aeiouy]")
_REPEAT_RE = re.compile(r"(.)\1+")


def phon(tok):
    """Consonant-skeleton phonetic key: robust to transliteration/vowel noise.

    e.g. apex/epeks -> pks, properties/puraapprttttiis -> prprts, first/phrstt -> frst."""
    if tok.isdigit():
        return tok
    t = tok
    for a, b in _PHON_SUBS:
        t = t.replace(a, b)
    t = _REPEAT_RE.sub(r"\1", _VOWELS_RE.sub("", t))
    return t or tok[:1]
