"""
Build an English minimal-pair add-on set from CLMET 3.1 (Corpus of Late Modern English Texts).

For each word pair (neutral -> loaded, plus neutral -> neutral controls) we sample real
sentences from CLMET that contain the neutral word, then create the exchange sentence by
swapping in the other word. Output has the same columns as the PolBiX dataset.csv, so the
same prompting script can run it (with --lang en).

CLMET has no truth labels: golden_truthfulness_base_claim is left empty and only
within-pair measures (change in P("true"), verdict flips) are analysed.

    python3 src/build_clmet_pairs.py --per_pair 12   # -> data/clmet/clmet_candidates.csv
"""
import argparse
import glob
import os
import random
import re

import pandas as pd

# (neutral forms, loaded forms, judgmental_exchange). Forms are aligned: singular, plural.
# None = that form is not used (e.g. adjective/verb ambiguity).
PAIRS = [
    (("crowd", "crowds"), ("mob", "mobs"), 1),
    (("multitude", "multitudes"), ("rabble", "rabbles"), 1),
    (("follower", "followers"), ("minion", "minions"), 1),
    (("writer", "writers"), ("scribbler", "scribblers"), 1),
    (("lawyer", "lawyers"), ("pettifogger", "pettifoggers"), 1),
    (("physician", "physicians"), ("quack", "quacks"), 1),
    (("banker", "bankers"), ("usurer", "usurers"), 1),
    (("speech", "speeches"), ("harangue", "harangues"), 1),
    (("enthusiasm", None), ("fanaticism", None), 1),
    (("religion", "religions"), ("superstition", "superstitions"), 1),
    (("frugality", None), ("stinginess", None), 1),
    (("plan", "plans"), ("scheme", "schemes"), 1),
    ((None, "Catholics"), (None, "papists"), 1),
    ((None, "officials"), (None, "placemen"), 1),
    ((None, "soldiers"), (None, "hirelings"), 1),
    ((None, "peasants"), (None, "yokels"), 1),
    ((None, "taxes"), (None, "exactions"), 1),
    (("statesman", "statesmen"), ("demagogue", "demagogues"), 1),
    ((None, "foreigners"), (None, "aliens"), 1),
    (("reformer", "reformers"), ("agitator", "agitators"), 1),
    (("negotiation", "negotiations"), ("intrigue", "intrigues"), 1),
    (("government", "governments"), ("regime", "regimes"), 1),
    ((None, "servants"), (None, "menials"), 1),
    ((None, "labourers"), (None, "drudges"), 1),
    ((None, "natives"), (None, "savages"), 1),
    (("merchant", "merchants"), ("huckster", "hucksters"), 1),
    (("the poor", None), ("the paupers", None), 1),
    # controls: neutral -> neutral synonym
    (("ship", "ships"), ("vessel", "vessels"), 0),
    (("answer", "answers"), ("reply", "replies"), 0),
    (("gift", "gifts"), ("present", "presents"), 0),
    (("friend", "friends"), ("companion", "companions"), 0),
    (("house", "houses"), ("dwelling", "dwellings"), 0),
    (("stream", "streams"), ("brook", "brooks"), 0),
    (("road", "roads"), ("highway", "highways"), 0),
    (("shop", "shops"), ("store", "stores"), 0),
]

# "the poor" only when used as a noun (not "the poor man")
POOR = r"\bthe poor\b(?=\s*(?:[,.;:!?)]|and\b|who\b|are\b|were\b|have\b|had\b|of\b|in\b|to\b|is\b|was\b))"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`'\"])")


def load_texts(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)) if os.path.isdir(path) else [path]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    text_col = "text" if "text" in df.columns else df.select_dtypes("object").columns[0]
    id_col = next((c for c in ["id", "text_id", "file"] if c in df.columns), None)
    period_col = next((c for c in ["period", "decade", "year"] if c in df.columns), None)
    return [(str(r[id_col]) if id_col else str(i), str(r[period_col]) if period_col else "", str(r[text_col]))
            for i, r in df.iterrows()]


def clean(text):
    """CLMET 'plain' is tokenised (spaces before punctuation, `` '' quotes): undo that."""
    text = re.sub(r"<[^>]+>", " ", text)          # stray markup
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+('s|'d|'ll|'re|'ve|'m|n't)\b", r"\1", text)
    text = re.sub(r"\s+-\s+", " - ", text)
    return text.strip()


ABBREV = re.compile(r"\b(Mr|Mrs|Dr|St|Messrs|Esq|Rev|Capt|Col|Gen|Sir|i\.e|e\.g|viz|No|Vol|p|pp|ch)\.$")


def good_sentence(s):
    n = len(s.split())
    return (8 <= n <= 30
            and s[0].isupper() and s[-1] in ".!?"
            and not re.search(r"``|''|`|\"|--|[\[\]{}_|=*]|\d{3,}", s)
            and not s.startswith("'")
            and s.count(",") <= 3 and s.count(";") <= 1
            and not ABBREV.search(s)
            and sum(w.isupper() and len(w) > 1 for w in s.split()) <= 1)


def fix_article(sentence, start):
    """Adjust a/an directly before position `start` to the following word."""
    m = re.search(r"\b([Aa]n?) $", sentence[:start])
    if not m:
        return sentence
    nxt = sentence[start:start + 1].lower()
    art = m.group(1)
    new = ("an" if nxt in "aeiou" else "a")
    new = new.capitalize() if art[0].isupper() else new
    return sentence[:m.start(1)] + new + sentence[m.end(1):]


DETS = set("the a an this that these those his her their our my your its every each no any some one "
           "such another other whole great large vast small old own first last same".split())
AFTER_OK = set("of and or but was were is are had has have who whom which that to in with at for on by from as "
               "would could should will may might must did said then when where than whose into upon".split())
CAPITAL = {"catholics"}
MASS = {"enthusiasm", "frugality", "religion"}
PRONOUNS = set("he she it they we i you who which that".split())


def noun_use(s, m, plural):
    """Keep only clear, common-noun uses of the target word: lower-case, preceded by a determiner
    (singular) or not by a pronoun (plural), and not the first part of a compound noun."""
    word = s[m.start():m.end()]
    if m.start() > 0 and word[0].isupper() and not word.istitle() or (m.start() > 0 and word[0].isupper() and word.lower() not in CAPITAL):
        return False                                   # capitalised mid-sentence: name / title
    if (m.start() > 0 and s[m.start() - 1] in "-'") or s[m.end():m.end() + 1] in ("-", "'"):
        return False                                   # hyphenated compound or possessive
    before = s[:m.start()].split()
    prev = before[-1].lower().strip(",;:(") if before else ""
    after = s[m.end():].lstrip()
    nxt = re.match(r"[A-Za-z]+", after)
    if nxt and not re.match(r"^[,.;:!?)]", after) and nxt.group(0).lower() not in AFTER_OK:
        return False                                   # likely "X gazette", "X vessel" compound
    if before and before[-1][:1].isupper() and len(before) > 1:
        return False                                   # "Camden Road", "Salvation Army"
    if word.lower().startswith("the "):
        return True
    if plural or word.lower() in MASS:
        return prev not in PRONOUNS and prev != "to"
    return prev in DETS


def make_pairs(texts, per_pair, seed):
    rng = random.Random(seed)
    sentences = []
    for tid, period, t in texts:
        for s in SENT_SPLIT.split(clean(t)):
            if good_sentence(s):
                sentences.append((tid, period, s))
    print(f"{len(sentences):,} candidate sentences from {len(texts)} texts")

    rows = []
    for neutral, loaded, judg in PAIRS:
        forms = [(a, b) for a, b in zip(neutral, loaded) if a]
        pats = [(re.compile(POOR if a == "the poor" else rf"\b{re.escape(a)}\b", re.I), a, b) for a, b in forms]
        hits = []
        for tid, period, s in sentences:
            found = [(p, a, b) for p, a, b in pats if len(p.findall(s)) == 1]
            if len(found) != 1:
                continue
            if any(re.search(rf"\b{re.escape(b)}\b", s, re.I) for _, b in forms):
                continue
            if not noun_use(s, found[0][0].search(s), plural=found[0][1] != forms[0][0] or neutral[0] is None):
                continue
            hits.append((tid, period, s, found[0]))
        rng.shuffle(hits)
        chosen, used_texts = [], set()
        for h in hits:                        # at most one sentence per text, for spread
            if h[0] in used_texts:
                continue
            chosen.append(h); used_texts.add(h[0])
            if len(chosen) == per_pair:
                break
        base_word = neutral[0] or neutral[1]
        exch_word = loaded[0] or loaded[1]
        for k, (tid, period, s, (p, a, b)) in enumerate(chosen, 1):
            m = p.search(s)
            word = s[m.start():m.end()]
            repl = b[0].upper() + b[1:] if word[0].isupper() else b
            ex = s[:m.start()] + repl + s[m.end():]
            ex = fix_article(ex, m.start())
            rows.append(dict(
                id=f"clmet_{base_word.replace(' ', '-')}_{exch_word.replace(' ', '-')}_{k}",
                base_word=base_word, exchange_word=exch_word, base_claim=s, exchange_claim=ex,
                golden_truthfulness_base_claim="", shift="none", axis="none",
                judgmental_base=0, judgmental_exchange=judg,
                origin_dataset=f"CLMET3.1:{tid}:{period}"))
        print(f"{base_word:>14} -> {exch_word:<14} {len(hits):6d} hits, {len(chosen)} used")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--clmet", default=os.path.join(root, "data", "clmet", "raw", "plain"))
    ap.add_argument("--out", default=os.path.join(root, "data", "clmet", "clmet_candidates.csv"))
    ap.add_argument("--per_pair", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = make_pairs(load_texts(a.clmet), a.per_pair, a.seed)
    df.to_csv(a.out, index=False)
    print(f"\nWrote {len(df)} pairs ({(df.judgmental_exchange == 1).sum()} loaded, "
          f"{(df.judgmental_exchange == 0).sum()} control) to {a.out}")
