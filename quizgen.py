#!/usr/bin/env python3
"""
Automatic Quiz Generation from Text -- classical NLP only (no LLMs, no neural text generation).

Pipeline
--------
1. Preprocessing      : whitespace normalisation, rule-based sentence segmentation
                        (abbreviation / initial aware), regex word tokenisation.
2. Linguistic cues    : heuristic POS filtering (stop-words, verb/adverb suffix rules),
                        capitalisation-based proper-noun chunking (light NER), regex
                        YEAR / NUMBER detection.
3. Importance scoring : TF-IDF over sentences, TextRank (graph centrality on cosine
                        similarity) for sentence selection, TF-IDF weights for keywords.
4. Distractors        : LSA (TruncatedSVD on TF-IDF) context vectors -> semantically
                        close but distinct terms from the same text; numeric
                        perturbation for years / quantities.
5. Question building  : cloze, multiple-choice, True/False, and rule-based WH
                        questions (subject-aux inversion templates).

Dependencies: numpy, scikit-learn (nothing else).
"""
from __future__ import annotations

import argparse
import difflib
import itertools
import json
import random
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------- #
# Lexical resources
# --------------------------------------------------------------------------- #
STOP = set("""a about above after again against all almost also although always am among an and
another any are around as at be because been before being below between both but by can could did
do does doing down during each either else even ever every few for from further had has have having
he her here hers herself him himself his how however i if in into is it its itself just least less
like many may me might more most much must my myself no nor not now of off often on once only or
other our ours ourselves out over own per rather same shall she should since so some such than that
the their theirs them themselves then there these they this those through thus to too under until up
upon us very was we were what when where whether which while who whom whose why will with within
without would yet you your yours yourself yourselves""".split())

# Frequent verbs / light words that should not form part of a noun-phrase term.
VERBISH = set("""use uses used make makes made produce produces convert converts contain contains
include includes become becomes called known help helps allow allows create creates take takes give
gives show shows cause causes form forms provide provides require requires need needs begin begins
began get gets got go goes went see sees saw find finds found keep keeps held hold holds run runs ran
work works build builds built lead leads led move moves play plays turn turns bring brings""".split())

MONTHS_DAYS = set("""january february march april june july august september october november december
monday tuesday wednesday thursday friday saturday sunday""".split())

CONNECT = {"of", "the", "de", "von", "van", "for"}
UNITS = {"percent", "million", "billion", "trillion", "thousand", "hundred"}
BAD_STARTS = {"he", "she", "it", "they", "this", "these", "those", "that", "however", "moreover",
              "also", "but", "and", "thus", "therefore", "such", "their", "his", "her", "its",
              "furthermore", "meanwhile", "then", "so", "yet", "instead", "nevertheless"}

# Adjectives / ordinals / determiner-like words that should not start a noun-phrase term.
GENERIC = set("""first second third last next previous new old many several various large small big little
great modern early late single different general same main major certain complete abstract powerful further
other another own whole entire real common recent current future possible important similar particular
specific total half full high low long short best better most least basic key""".split())
GENERIC_SUFFIXES = ("ous", "ful", "ive", "able", "ible", "less")

NATIONALITY = set("""german british french american english russian chinese japanese indian european african
asian italian spanish greek roman dutch swiss swedish australian canadian""".split())

IRREGULAR_VERBS = set("""wrote written brought built made held led ran began begun took taken gave given saw seen found
knew known became came went gone got said told thought kept left meant met paid sent set spent stood understood
won drew drawn grew grown chose chosen broke broken rose risen fell fallen flew sang spoke swam sat lay""".split())

# Clearly verbal base forms (noun/verb-ambiguous words such as "design" or "process" are deliberately absent).
VERB_BASE = set("""according remain stretch describe introduce replace demonstrate observe appear bring allow learn share propose
rely define regard manipulate intend write break enable involve represent consist depend determine perform operate
store weigh double become believe exist occur grow increase decrease reduce improve develop discover invent
complete publish establish announce expand extend connect combine separate divide measure compare explain
suggest prove solve compute calculate transmit receive deliver achieve reach rely emerge evolve spread arrive
declare adopt reject accept apply attach attract avoid belong carry choose claim collect compose conduct
consider continue contribute control cover decide deliver depict destroy detect direct display distribute
enter equal examine expect experience explore express fail fill follow generate identify imagine indicate
influence inform inspire install interpret investigate join launch maintain manage mention name notice obtain
offer open organize participate pass pay place predict prefer prepare present prevent protect publish pull
push raise realize recognize record refer reflect relate release rely remove repeat replace report require
research respond restore retain return reveal rule save select send serve settle simulate specify support
supply surround survive test transfer transform translate treat trigger understand unite update utilize
visit wait watch win""".split())


def verbish(w: str) -> bool:
    w = w.lower()
    if w in IRREGULAR_VERBS or w in VERBISH:
        return True
    forms = set()
    if w.endswith("ies"):
        forms.add(w[:-3] + "y")
    if w.endswith("es"):
        forms.add(w[:-2])
    if w.endswith("s"):
        forms.add(w[:-1])
    if w.endswith("ed"):
        forms |= {w[:-2], w[:-1], w[:-3] + "y" if w.endswith("ied") else w[:-2]}
        if len(w) > 4 and w[-3] == w[-4]:
            forms.add(w[:-3])
    return bool(forms & VERB_BASE) or w in VERB_BASE


ORG_WORDS = {"university", "institute", "company", "corporation", "inc", "ltd", "association", "bank",
             "ministry", "party", "council", "society", "agency", "committee", "department", "school",
             "college", "laboratory", "lab", "foundation", "organization", "organisation", "union",
             "army", "navy", "museum", "academy", "club", "group", "press", "prize", "award", "act",
             "treaty", "war", "empire", "dynasty", "project", "program", "programme", "system"}
PLACE_WORDS = {"river", "mountain", "mountains", "city", "island", "islands", "sea", "ocean", "republic",
               "kingdom", "valley", "lake", "desert", "bay", "gulf", "coast", "province", "state", "county",
               "africa", "asia", "europe", "america", "antarctica", "australia", "india", "china", "japan",
               "france", "germany", "england", "britain", "russia", "canada", "brazil", "mexico", "italy",
               "spain", "egypt", "paris", "london", "berlin", "tokyo", "rome", "york", "delhi", "bengaluru",
               "beijing", "moscow", "united", "states", "earth", "pacific", "atlantic"}

AUX = r"is|are|was|were|has|have|had|can|could|will|would|should|may|might"
WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?:[.,]\d+)*%?")
BLANK = "_____"

ABBR = ["Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "vs", "etc", "e.g", "i.e", "Fig", "No",
        "Inc", "Ltd", "Co", "U.S", "U.K"]


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    text: str
    start: int
    end: int
    kind: str          # YEAR | NUMBER | PROPER | TERM
    score: float = 0.0


@dataclass
class Sentence:
    idx: int
    text: str
    cands: list = field(default_factory=list)
    score: float = 0.0


@dataclass
class Question:
    qtype: str                       # mcq | tf | wh | cloze
    question: str
    answer: str
    options: Optional[list] = None   # mcq only
    answer_letter: Optional[str] = None
    source: str = ""
    sentence_idx: int = -1

    def to_dict(self):
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [])}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> list:
    t = re.sub(r"\s+", " ", text.strip())
    for a in ABBR:
        t = re.sub(rf"\b{re.escape(a)}\.", lambda m: m.group(0)[:-1].replace(".", "<D>") + "<P>", t)
    t = re.sub(r"\b([A-Z])\.(?=\s[A-Z])", r"\1<P>", t)          # initials: "J. K. Rowling"
    parts = re.split(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])', t)
    out = []
    for p in parts:
        p = p.replace("<D>", ".").replace("<P>", ".").strip()
        if len(p.split()) >= 4:
            out.append(p)
    return out


def analyze(sentence: str) -> list:
    """Analyzer for TF-IDF: lower-cased content words (stop-words removed)."""
    return [w.lower() for w in re.findall(r"[A-Za-z]+", sentence) if len(w) > 2 and w.lower() not in STOP]


def fit_case(opt: str, at_start: bool) -> str:
    return opt[:1].upper() + opt[1:] if at_start and opt else opt


def norm_surface(text: str, kind: str) -> str:
    """Normalise surface form so sentence-initial capitals don't leak the answer."""
    if kind == "TERM" and text[:1].isupper() and not text.isupper() and text[1:].islower():
        return text[:1].lower() + text[1:]
    return text


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
class QuizGenerator:
    def __init__(self, text: str, seed: int = 42):
        self.rng = random.Random(seed)
        raw = split_sentences(text)
        if len(raw) < 4:
            raise ValueError("Need at least 4 sentences of input text to generate a meaningful quiz.")
        self.sentences = [Sentence(i, s) for i, s in enumerate(raw)]
        self.fulltext = " ".join(s.text for s in self.sentences).lower()
        self._build_models()
        self._extract_all_candidates()
        self._score_sentences()
        self._build_pools()

    # ---------------- statistical models ---------------- #
    def _build_models(self):
        self.vec = TfidfVectorizer(analyzer=analyze)
        self.X = self.vec.fit_transform([s.text for s in self.sentences])
        self.vocab = self.vec.vocabulary_
        w = np.asarray(self.X.sum(axis=0)).ravel()
        w = w / w.max() if w.max() > 0 else w
        self.weight = {t: float(w[j]) for t, j in self.vocab.items()}
        # LSA embeddings for distractor similarity
        n, v = self.X.shape
        k = min(20, n - 1, v - 1)
        if k >= 2:
            E = TruncatedSVD(n_components=k, random_state=0).fit_transform(self.X)
        else:
            E = self.X.toarray()
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        self.E = E / np.where(norms == 0, 1, norms)
        # words seen capitalised in non-initial position => genuine proper nouns
        self.doc_caps = set()
        for s in self.sentences:
            for k2, m in enumerate(WORD.finditer(s.text)):
                w0 = m.group()
                if k2 > 0 and w0[0].isupper() and w0.lower() not in STOP:
                    self.doc_caps.add(w0.lower())

    def _tok_weight(self, phrase: str) -> float:
        parts = [p.lower() for p in re.split(r"[\s-]+", phrase) if p.isalpha() and len(p) > 2 and p.lower() not in STOP]
        vals = [self.weight.get(p, 0.0) for p in parts]
        return float(np.mean(vals)) if vals else 0.0

    # ---------------- candidate extraction (heuristic NER/chunking) ---------------- #
    def _extract(self, s: Sentence) -> list:
        text = s.text
        toks = list(WORD.finditer(text))
        n = len(toks)
        cands, covered = [], []

        def gap_ok(a, b):
            return text[toks[a].end():toks[b].start()] == " "

        # numbers and years
        for i, t in enumerate(toks):
            w = t.group()
            if not re.match(r"^\d", w):
                continue
            if re.fullmatch(r"(1\d{3}|20\d{2})", w):
                cands.append(Candidate(w, t.start(), t.end(), "YEAR"))
            else:
                end = t.end()
                if i + 1 < n and gap_ok(i, i + 1) and toks[i + 1].group().lower() in UNITS:
                    end = toks[i + 1].end()
                cands.append(Candidate(text[t.start():end], t.start(), end, "NUMBER"))
            covered.append((t.start(), cands[-1].end))

        # proper-noun phrases
        def is_cap(j):
            w = toks[j].group()
            lw = w.lower()
            if not w[0].isupper() or lw in STOP or lw in MONTHS_DAYS or not w.isalpha() and "-" not in w:
                return False
            if j == 0:
                return (j + 1 < n and toks[j + 1].group()[0].isupper() and gap_ok(j, j + 1)) or lw in self.doc_caps
            return True

        i = 0
        while i < n:
            if is_cap(i):
                j = i
                while True:
                    k = j + 1
                    if k < n and gap_ok(j, k) and is_cap(k):
                        j = k
                        continue
                    if (k + 1 < n and toks[k].group().lower() in CONNECT and gap_ok(j, k)
                            and gap_ok(k, k + 1) and is_cap(k + 1)):
                        j = k + 1
                        continue
                    break
                a, b = toks[i].start(), toks[j].end()
                cands.append(Candidate(text[a:b], a, b, "PROPER"))
                covered.append((a, b))
                i = j + 1
            else:
                i += 1

        # common-noun terms: runs of content words, windows of 1..3 tokens
        def in_cov(t):
            return any(a <= t.start() < b for a, b in covered)

        def ok(t, nxt=None):
            """Can token t belong to a noun-phrase term? (heuristic POS filter)"""
            w = t.group()
            lw = w.lower()
            if not (w.replace("-", "").isalpha()) or len(w) < 3 or lw in STOP or in_cov(t):
                return False
            if lw.endswith("ly") or verbish(lw):
                return False
            if lw.endswith("ed") and len(lw) > 4:
                # participial adjective ("integrated circuits") only if directly followed by a content word
                return nxt is not None and text[t.end():nxt.start()] == " " and nxt.group().isalpha() \
                    and nxt.group().lower() not in STOP and not verbish(nxt.group().lower())
            return True

        runs, run = [], []
        for i, t in enumerate(toks):
            nxt = toks[i + 1] if i + 1 < n else None
            good = ok(t, nxt)
            if good and (not run or text[run[-1].end():t.start()] == " "):
                run.append(t)
            else:
                if run:
                    runs.append(run)
                run = [t] if good else []
        if run:
            runs.append(run)
        for r in runs:
            # English noun phrases are head-final: a term must be a suffix of the run
            while r and (r[0].group().lower() in GENERIC or r[0].group().lower().endswith(GENERIC_SUFFIXES)) and len(r) > 1:
                r = r[1:]
            if r and r[-1].group().lower().endswith("ed"):
                continue                                   # run cannot end in a participle
            if len(r) == 1 and (r[0].group().lower() in GENERIC or r[0].group().lower().endswith(GENERIC_SUFFIXES)):
                continue
            for size in (1, 2, 3):
                if size <= len(r):
                    seg = r[len(r) - size:]
                    st, en = seg[0].start(), seg[-1].end()
                    cands.append(Candidate(norm_surface(text[st:en], "TERM"), st, en, "TERM"))
        # drop nationality adjectives and other non-entity proper phrases
        cands = [c for c in cands if not (c.kind == "PROPER" and c.text.lower() in NATIONALITY)]
        return cands

    def _extract_all_candidates(self):
        for s in self.sentences:
            s.cands = self._extract(s)
        weights = [self._tok_weight(c.text) for s in self.sentences for c in s.cands if c.kind == "TERM"]
        pos = [w for w in weights if w > 0]
        base = float(np.median(pos)) if pos else 0.2
        for s in self.sentences:
            for c in s.cands:
                if c.kind == "YEAR":
                    c.score = base * 1.25
                elif c.kind == "NUMBER":
                    c.score = base * 1.05
                elif c.kind == "PROPER":
                    c.score = self._tok_weight(c.text) * 1.15 + 0.05
                else:
                    nt = len(c.text.split())
                    freq = len(re.findall(rf"\b{re.escape(c.text.lower())}(?:s|es)?\b", self.fulltext))
                    recur = 1 + 0.35 * np.log(max(freq, 1)) if freq > 1 else 0.75   # recurring phrases = key terms
                    c.score = self._tok_weight(c.text) * (1 + 0.3 * (nt - 1)) * recur
            s.cands = [c for c in s.cands if c.score > 0]

    # ---------------- sentence importance (TextRank + keyword density) ---------------- #
    def _score_sentences(self, d=0.85, iters=50):
        S = (self.X @ self.X.T).toarray()
        np.fill_diagonal(S, 0)
        rs = S.sum(axis=1, keepdims=True)
        P = np.divide(S, rs, out=np.zeros_like(S), where=rs > 0)
        r = np.full(len(S), 1.0 / len(S))
        for _ in range(iters):
            r = (1 - d) / len(S) + d * P.T @ r
        r = r / r.max()
        for s in self.sentences:
            top = sorted((c.score for c in s.cands), reverse=True)[:3]
            kd = float(np.mean(top)) if top else 0.0
            pos = 1.0 / (1 + 0.25 * s.idx)
            s.score = 0.6 * float(r[s.idx]) + 0.3 * kd + 0.1 * pos

    def _usable(self, s: Sentence) -> bool:
        words = s.text.split()
        if not (8 <= len(words) <= 35) or "?" in s.text:
            return False
        return words[0].lower().strip(",") not in BAD_STARTS

    # ---------------- distractor pools ---------------- #
    def _build_pools(self):
        self.pool = {"TERM": {}, "PROPER": {}, "YEAR": set(), "NUMBER": set()}
        term_scores = [c.score for s in self.sentences for c in s.cands if c.kind == "TERM"]
        thr = float(np.percentile(term_scores, 40)) if term_scores else 0.0
        for s in self.sentences:
            for c in s.cands:
                if c.kind == "YEAR":
                    self.pool["YEAR"].add(int(c.text))
                elif c.kind in ("TERM", "PROPER"):
                    if c.kind == "TERM" and c.score < thr:
                        continue
                    e = self.pool[c.kind].setdefault(c.text.lower(), {"surface": c.text, "sents": set()})
                    e["sents"].add(s.idx)

    def _vec(self, entry):
        v = self.E[sorted(entry["sents"])].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v

    def distractors(self, s: Sentence, c: Candidate, k: int = 3) -> Optional[list]:
        if c.kind == "YEAR":
            return self._year_distractors(int(c.text), k)
        if c.kind == "NUMBER":
            return self._number_distractors(c.text, k)
        key = c.text.lower()
        a_tokens = set(re.split(r"[\s-]+", key))
        sent_low = s.text.lower()
        ref = self.pool[c.kind].get(key) or {"sents": {s.idx}}
        av = self._vec(ref)
        nwords = len(c.text.split())
        acat = self._pcat(c.text) if c.kind == "PROPER" else None
        ranked = []
        for k2, e in self.pool[c.kind].items():
            if k2 == key or set(re.split(r"[\s-]+", k2)) & a_tokens or k2 in sent_low or key in k2:
                continue
            if acat and self._pcat(e["surface"]) != acat:      # keep entity types consistent
                continue
            sim = float(self._vec(e) @ av)
            if sim > 0.97:                      # near-duplicate context => may also be correct
                continue
            diff = abs(len(k2.split()) - nwords)
            if diff > 1:
                continue
            bonus = 0.05 if (k2.endswith("s") == key.endswith("s")) else 0.0
            ranked.append((sim + bonus - 0.2 * diff + self.rng.random() * 0.01, e["surface"]))
        ranked.sort(reverse=True)
        out, used = [], set(a_tokens)
        for _, sf in ranked:                    # options must not share words with each other
            toks_ = set(re.split(r"[\s-]+", sf.lower()))
            stems = {t.rstrip("s") for t in toks_}
            if stems & {u.rstrip("s") for u in used}:
                continue
            out.append(norm_surface(sf, c.kind))
            used |= toks_
            if len(out) == k:
                break
        return out if len(out) >= k else None

    def _year_distractors(self, y: int, k: int) -> Optional[list]:
        near = sorted((v for v in self.pool["YEAR"] if v != y), key=lambda v: abs(v - y))[:k - 1]
        out = list(near)
        offsets = [-20, -10, -5, -3, -2, 2, 3, 5, 10, 15, 25]
        self.rng.shuffle(offsets)
        for o in offsets:
            if len(out) >= k:
                break
            cand = y + o
            if cand not in out and cand != y and 0 < cand <= 2100:
                out.append(cand)
        return [str(v) for v in out[:k]] if len(out) >= k else None

    def _number_distractors(self, txt: str, k: int) -> Optional[list]:
        m = re.match(r"^([\d.,]+)(.*)$", txt)
        if not m:
            return None
        num, suffix = m.group(1), m.group(2)
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            return None
        has_comma, is_int = "," in num, "." not in num.replace(",", "") or val.is_integer()
        percent = "percent" in suffix or "%" in suffix or "%" in num
        factors = [0.5, 0.75, 1.25, 1.5, 2, 3, 0.1, 10]
        self.rng.shuffle(factors)
        out = []
        for f in factors:
            v = val * f
            if percent and v > 100:
                continue
            if is_int:
                v = int(round(v))
                s = f"{v:,}" if has_comma else str(v)
            else:
                s = f"{v:.1f}"
            full = s + suffix
            if v > 0 and full != txt and full not in out:
                out.append(full)
            if len(out) == k:
                return out
        return None

    # ---------------- question builders ---------------- #
    def _blank(self, s: Sentence, c: Candidate) -> str:
        stem = re.sub(rf"\b{re.escape(c.text)}\b", BLANK, s.text, flags=re.I) if c.text[0].isalnum() else \
            s.text[:c.start] + BLANK + s.text[c.end:]
        return stem

    def _decap(self, phrase: str) -> str:
        w = phrase.split()
        if not w:
            return phrase
        f = w[0]
        if f.lower() in STOP:
            return f[0].lower() + f[1:]
        if f.isupper() or f.lower() in self.doc_caps or (len(w) > 1 and w[1][:1].isupper()):
            return phrase
        return f[0].lower() + phrase[1:]

    def _pcat(self, ans: str) -> str:
        """Coarse entity type for a proper-noun phrase: place / org / person / other."""
        low = {w.lower() for w in ans.split()}
        if low & PLACE_WORDS:
            return "place"
        if low & ORG_WORDS:
            return "org"
        if self._is_person(ans) or (len(low) == 1 and low & self.person_tokens):
            return "person"
        return "other"

    @property
    def person_tokens(self) -> set:
        if not hasattr(self, "_ptoks"):
            self._ptoks = {w.lower() for s in self.sentences for c in s.cands
                           if c.kind == "PROPER" and self._is_person(c.text) for w in c.text.split()}
        return self._ptoks

    def _is_person(self, ans: str) -> bool:
        ws = ans.split()
        low = {w.lower() for w in ws}
        return 2 <= len(ws) <= 3 and all(w[0].isupper() and w.isalpha() for w in ws) \
            and not (low & ORG_WORDS) and not (low & PLACE_WORDS)

    def _is_place(self, ans: str) -> bool:
        return bool({w.lower() for w in ans.split()} & PLACE_WORDS)

    def make_cloze(self, s, c):
        stem = self._blank(s, c)
        return Question("cloze", f"Fill in the blank: {stem}", norm_surface(c.text, c.kind), source=s.text, sentence_idx=s.idx)

    def make_mcq(self, s, c):
        d = self.distractors(s, c)
        if not d:
            return None
        at_start = c.start == 0
        answer = fit_case(norm_surface(c.text, c.kind), at_start)
        opts = [answer] + [fit_case(x, at_start) for x in d]
        self.rng.shuffle(opts)
        letter = "ABCD"[opts.index(answer)]
        stem = self._blank(s, c)
        return Question("mcq", f"Choose the option that best completes the sentence: {stem}", answer,
                        options=opts, answer_letter=letter, source=s.text, sentence_idx=s.idx)

    def make_tf(self, s, c):
        if self.rng.random() < 0.5:
            return Question("tf", s.text, "True", source=s.text, sentence_idx=s.idx)
        d = self.distractors(s, c)
        if not d:
            return None
        rep = fit_case(d[0], c.start == 0)
        false_stmt = s.text[:c.start] + rep + s.text[c.end:]
        return Question("tf", false_stmt, "False", source=s.text, sentence_idx=s.idx)

    def make_wh(self, s, c):
        text = s.text.rstrip(" .!")
        m = re.match(rf"^(?P<subj>.+?)\s+(?P<aux>{AUX})\s+(?P<rest>.+)$", text)
        if not m or len(m["subj"].split()) > 8:
            return None
        subj, aux, rest, ans = m["subj"], m["aux"], m["rest"], c.text
        strip_art = lambda x: re.sub(r"^(the|a|an)\s+", "", x.strip(), flags=re.I).lower()
        q = None
        if not m:
            return None
        if strip_art(subj) == strip_art(ans) and c.kind in ("TERM", "PROPER") and len(rest.split()) >= 3:
            if c.kind == "PROPER":
                wh = "Who" if self._is_person(ans) else ("What" if (self._is_place(ans) or
                     {w.lower() for w in ans.split()} & ORG_WORDS) else None)
            else:
                wh = "What"
            if wh:
                q = f"{wh} {aux} {rest}?"
        else:
            idx = rest.find(ans)
            if idx < 0:
                return None
            before, after = rest[:idx].rstrip(), rest[idx + len(ans):].lstrip()
            bw = before.split()
            prep = bw[-1].lower() if bw else ""
            before_wo = " ".join(bw[:-1]) if prep in {"in", "on", "at", "by", "since", "until", "during", "around"} else before
            rest_wo = f"{before_wo} {after}".strip()
            rest_wo = re.sub(r"\s+([,;])", r"\1", rest_wo)
            if c.kind == "YEAR":
                if prep in {"in", "on", "during", "around"}:
                    q = f"When {aux} {self._decap(subj)} {rest_wo}?"
                elif prep in {"since", "by", "until"}:
                    q = f"{prep.capitalize()} which year {aux} {self._decap(subj)} {rest_wo}?"
            elif c.kind == "PROPER":
                if prep in {"in", "at"} and self._is_place(ans):
                    q = f"Where {aux} {self._decap(subj)} {rest_wo}?"
                elif prep == "by" and self._is_person(ans):
                    q = f"By whom {aux} {self._decap(subj)} {rest_wo}?"
            elif c.kind == "NUMBER" and subj.lower() == "there" and before == "" and re.fullmatch(r"\d+", ans):
                q = f"How many {after} {aux} there?"
        if not q or ans.lower() in q.lower() or len(q.split()) < 4:
            return None
        return Question("wh", q, norm_surface(ans, c.kind), source=s.text, sentence_idx=s.idx)

    # ---------------- orchestration ---------------- #
    def generate(self, n: int = 10, types=("mcq", "tf", "wh", "cloze")) -> list:
        builders = {"mcq": self.make_mcq, "tf": self.make_tf, "wh": self.make_wh, "cloze": self.make_cloze}
        plan = itertools.cycle(["mcq", "wh", "tf", "mcq", "cloze"] if set(types) == set(builders) else list(types))
        ranked = sorted((s for s in self.sentences if self._usable(s) and s.cands), key=lambda s: -s.score)
        used_answers, out = set(), []

        for _ in range(2):                                           # second pass reuses sentences
            for s in ranked:
                if len(out) >= n:
                    break
                if any(q.sentence_idx == s.idx for q in out) and _ == 0:
                    continue
                cands = [c for c in sorted(s.cands, key=lambda c: -c.score)
                         if c.text.lower() not in used_answers and len(c.text.split()) <= 3
                         and len(re.findall(rf"\b{re.escape(c.text.lower())}", s.text.lower())) == 1
                         and not (c.kind == "TERM" and len(c.text) < 4)]
                want = next(plan)
                chain = [want] + [t for t in ("mcq", "cloze") if t != want and t in types]
                made = None
                for qt in chain:
                    for c in cands[:6]:
                        q = builders[qt](s, c)
                        if q:
                            made, used = q, c
                            break
                    if made:
                        break
                if made:
                    out.append(made)
                    used_answers.add(used.text.lower())
            if len(out) >= n:
                break
        return out


# --------------------------------------------------------------------------- #
# Rendering & CLI
# --------------------------------------------------------------------------- #
def to_markdown(qs: list) -> str:
    lines = ["# Generated Quiz", ""]
    label = {"mcq": "Multiple choice", "tf": "True / False", "wh": "Short answer", "cloze": "Fill in the blank"}
    for i, q in enumerate(qs, 1):
        lines.append(f"**Q{i}. ({label[q.qtype]})** {q.question}")
        if q.qtype == "mcq":
            lines += [f"   - {'ABCD'[j]}) {o}" for j, o in enumerate(q.options)]
        elif q.qtype == "tf":
            lines.append("   - True / False")
        lines.append("")
    lines += ["---", "", "## Answer key", ""]
    for i, q in enumerate(qs, 1):
        ans = f"{q.answer_letter}) {q.answer}" if q.qtype == "mcq" else q.answer
        lines.append(f"{i}. **{ans}**  \n   _Source:_ {q.source}")
    return "\n".join(lines) + "\n"


def run_interactive(qs: list) -> None:
    score = 0
    for i, q in enumerate(qs, 1):
        print(f"\nQ{i}. {q.question}")
        if q.qtype == "mcq":
            for j, o in enumerate(q.options):
                print(f"   {'ABCD'[j]}) {o}")
            ok = input("Your answer (A-D): ").strip().upper()[:1] == q.answer_letter
        elif q.qtype == "tf":
            ok = input("True or False (t/f): ").strip().lower()[:1] == q.answer[0].lower()
        else:
            g = re.sub(r"^(the|a|an)\s+", "", input("Your answer: ").strip().lower())
            a = re.sub(r"^(the|a|an)\s+", "", q.answer.lower())
            ok = difflib.SequenceMatcher(None, g, a).ratio() >= 0.85
        score += ok
        print("Correct!" if ok else f"Not quite. Answer: {q.answer}")
    print(f"\nScore: {score}/{len(qs)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Automatic quiz generation from text using classical NLP.")
    ap.add_argument("input", nargs="?", help="text file (default: stdin)")
    ap.add_argument("-n", "--num", type=int, default=10, help="number of questions")
    ap.add_argument("--types", default="mcq,tf,wh,cloze", help="comma list of: mcq,tf,wh,cloze")
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--out", help="write output to this file")
    ap.add_argument("-i", "--interactive", action="store_true", help="take the quiz in the terminal")
    a = ap.parse_args(argv)

    text = open(a.input, encoding="utf-8").read() if a.input else sys.stdin.read()
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    qs = QuizGenerator(text, seed=a.seed).generate(a.num, types)
    if a.interactive:
        return run_interactive(qs)
    out = json.dumps([q.to_dict() for q in qs], indent=2, ensure_ascii=False) if a.format == "json" else to_markdown(qs)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(out)
        print(f"Wrote {len(qs)} questions to {a.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
