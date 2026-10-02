"""Learn an Indic-script token -> Latin token dictionary from TRAINING match pairs only.

For every training ground-truth pair (S1 record, S2/S3 record) where the S2/S3 record
contains Indic-script tokens, we count co-occurrences between each Indic token and the
canonicalised Latin tokens of the S1 record. Each Indic token is mapped to the Latin
token with the highest Dice coefficient (subject to support thresholds). Tokens never
seen in training fall back to unidecode transliteration.
"""
import re
from collections import Counter

import pandas as pd

from .normalize import INDIC_KEY_RE, INDIC_RE, LEGAL_MAP, basic


def _indic_list(text):
    out = []
    for t in text.split():
        if INDIC_RE.search(t):
            k = INDIC_KEY_RE.sub("", t)
            if k:
                out.append(k)
    return out


def _dice_map(c_t, c_l, c_tl, min_count, min_dice):
    best = {}
    for (t, l), c in c_tl.items():
        if c < min_count:
            continue
        d = 2 * c / (c_t[t] + c_l[l])
        if d >= min_dice and d > best.get(t, (None, 0))[1]:
            best[t] = (l, d)
    return {t: l for t, (l, d) in best.items()}


def learn_translit(s1, others, pairs, min_count=3, min_dice=0.3, min_pos=2, min_pos_share=0.5):
    """s1/others: DataFrames indexed by entity_id; pairs: DataFrame[source1_entity_id, ids].

    Name tokens are aligned only against S1 *name* tokens and address tokens only against
    S1 *address* tokens (mixing the two maps e.g. the Odia word for "Private" to the
    state name, because the state co-occurs with every record). For names whose token
    count equals the S1 name's, tokens are aligned by position (strong evidence); the
    remaining tokens use Dice co-occurrence.
    """
    pat = "[\u0900-\u0DFF]"  # literal characters (pyarrow regex has no \u escapes)
    oth = others.loc[others.business_name.str.contains(pat) | others.business_address.str.contains(pat)]
    p = pairs[pairs.ids.isin(oth.index)]
    s1n = s1.business_name.to_dict()
    s1a = s1.business_address.to_dict()
    on = oth.business_name.to_dict()
    oa = oth.business_address.to_dict()
    nt, nl, ntl, pos = Counter(), Counter(), Counter(), Counter()
    at, al, atl = Counter(), Counter(), Counter()
    for a, b in zip(p.source1_entity_id.values, p.ids.values):
        raw = on[b].split()
        it = _indic_list(on[b])
        if it:
            lt = [LEGAL_MAP.get(t, t) for t in basic(s1n[a]).split()]
            if len(it) == len(raw) == len(lt):
                for x, y in zip(it, lt):
                    pos[(x, y)] += 1
            su, sl = set(it), set(lt)
            nt.update(su); nl.update(sl)
            for x in su:
                for y in sl:
                    ntl[(x, y)] += 1
        ia = set(_indic_list(oa[b]))
        if ia:
            la = set(basic(s1a[a]).split())
            at.update(ia); al.update(la)
            for x in ia:
                for y in la:
                    atl[(x, y)] += 1
    name_map = _dice_map(nt, nl, ntl, min_count, min_dice)
    tot = Counter()
    for (x, y), c in pos.items():
        tot[x] += c
    best = {}
    for (x, y), c in pos.items():
        if c >= min_pos and c / tot[x] >= min_pos_share and c > best.get(x, (None, 0))[1]:
            best[x] = (y, c)
    name_map.update({x: y for x, (y, c) in best.items()})
    addr_map = _dice_map(at, al, atl, min_count, min_dice)
    # names take priority (legal words); address-only tokens (states, cities) added after
    out = dict(addr_map)
    out.update(name_map)
    return out


def save_dict(d, path):
    pd.DataFrame(sorted(d.items()), columns=["indic", "latin"]).to_csv(path, sep="\t", index=False)


def load_dict(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return dict(zip(df.indic, df.latin))
