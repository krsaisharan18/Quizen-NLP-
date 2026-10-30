#!/usr/bin/env python3
"""
Automatic Quiz Generation from Text -- classical NLP only (no LLMs, no neural text generation).

Pipeline
--------
1. Linguistic analysis (nlp.py)
     tokenise -> POS-tag -> chunk noun phrases -> parse each clause into
     lead adverbial / subject / verb group / voice / objects / PPs / appositive
     -> type entities (PERSON, ORG, PLACE, NAMED, YEAR, QUANTITY, CONCEPT)
     -> resolve pronouns ("It" -> "The ENIAC") and bare surnames ("Turing" -> "Alan Turing").
2. Meaning-driven question frames
     "A <invented> B in <YEAR>"          -> Who / When / Where questions
     "X was <verb-ed> by Y"              -> Who ... ?  /  What replaced ... ?
     "X used <N> things"                 -> How many ... ?  /  How much ... ?
     "X is a Y that ..."                 -> definition question (which term?)
     "..., an observation known as Z"    -> What is the observation ... known as?
     any sentence                        -> True/False (role-aware false swaps) and cloze
3. Importance ranking
     TF-IDF + TextRank sentence centrality; keyphrase recurrence.
4. Distractors
     same entity type (person / place / term / year / number), LSA-similar context,
     never present in the source sentence, no shared words between options.
5. Selection
     score x diversity (type mix, one question per answer, spread over the text).

Dependencies: numpy, scikit-learn.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp import ANAPHORIC, Analyzer, MEASURE_VERBS, ORG_WORDS, PLACE_WORDS, UNITS, past_tense, third_person

STOP = set("""a about above after again against all almost also although always am among an and
another any are around as at be because been before being below between both but by can could did
do does doing down during each either else even ever every few for from further had has have having
he her here hers herself him himself his how however i if in into is it its itself just least less
like many may me might more most much must my myself no nor not now of off often on once only or
other our ours ourselves out over own per rather same shall she should since so some such than that
the their theirs them themselves then there these they this those through thus to too under until up
upon us very was we were what when where whether which while who whom whose why will with within
without would yet you your yours yourself yourselves""".split())

GENERIC_ADJ = set("first second third last next many several various single other another such same certain own few much "
                  "more most some".split())
CLAIM_NOUNS = set("""observation idea principle rule law theory claim concept model hypothesis finding proposal conjecture
effect phenomenon method technique approach argument assumption notion""".split())
CLAIM_VERBS = set("observe propose state claim suggest note discover argue find formulate hypothesise hypothesize "
                  "predict show conclude".split())
DISCOURSE_PRON = {"it", "they", "them", "he", "she", "him", "her", "this", "these", "those", "its", "their", "his"}
HUMAN_NOUNS = set("""engineers scientists researchers mathematicians physicists chemists biologists doctors teachers students
people workers soldiers farmers artists writers poets philosophers historians scholars officials leaders kings queens
emperors founders inventors programmers developers designers architects explorers settlers team teams colleagues
astronomers economists scientist engineer researcher mathematician physicist chemist biologist doctor teacher
physicians priests monks merchants citizens women men children crowd mob army forces troops government
rebels protesters parisians""".split())
DISC = {"this", "these", "those", "such", "former", "latter", "aforementioned"}
BLANK = "_____"
ABBR = ["Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "vs", "etc", "e.g", "i.e", "Fig", "No",
        "Inc", "Ltd", "Co", "U.S", "U.K"]
WEIGHT = {"WHO_AGENT": 1.0, "NAMING": 1.0, "DEFINITION": 1.0, "WHO_SUBJ": 0.95, "WHEN": 0.95, "WHAT_AGENT": 0.95,
          "QUANT": 0.9, "WHERE": 0.85, "TF": 0.7, "HOW": 0.75, "WHAT_SUBJ": 0.6, "CLOZE": 0.4}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Question:
    qtype: str                       # mcq | tf | wh | cloze
    question: str
    answer: str
    options: Optional[list] = None
    answer_letter: Optional[str] = None
    accept: Optional[list] = None    # alternative acceptable answers (short-answer grading)
    about: str = ""                  # what the question is about (context)
    kind: str = ""                   # frame that produced it (WHO_AGENT, WHEN, ...)
    source: str = ""
    sentence_idx: int = -1
    score: float = field(default=0.0, repr=False)

    def to_dict(self):
        d = asdict(self)
        d.pop("score", None)
        return {k: v for k, v in d.items() if v not in (None, [], "")}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> list:
    t = re.sub(r"\s+", " ", text.strip())
    for a in ABBR:
        t = re.sub(rf"\b{re.escape(a)}\.", lambda m: m.group(0)[:-1].replace(".", "<D>") + "<P>", t)
    t = re.sub(r"\b([A-Z])\.(?=\s[A-Z])", r"\1<P>", t)
    parts = re.split(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])', t)
    out = []
    for p in parts:
        p = p.replace("<D>", ".").replace("<P>", ".").strip()
        if len(p.split()) >= 4:
            out.append(p)
    return out


def analyze(sentence: str) -> list:
    return [w.lower() for w in re.findall(r"[A-Za-z]+", sentence) if len(w) > 2 and w.lower() not in STOP]


def words(s: str) -> set:
    return {w for w in re.split(r"[\s'’-]+", s.lower()) if len(w) > 2}


def cap1(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
class QuizGenerator:
    def __init__(self, text: str, seed: int = 42):
        self.rng = random.Random(seed)
        self.texts = split_sentences(text)
        if len(self.texts) < 4:
            raise ValueError("Need at least 4 sentences of input text to generate a meaningful quiz.")
        self.A = Analyzer(self.texts)
        self.sentences = self.A.parsed
        self.low = [s.lower() for s in self.texts]
        self._build_models()
        self._inventory()
        self._importance()

    # ---------------- statistical models ---------------- #
    def _build_models(self):
        self.vec = TfidfVectorizer(analyzer=analyze)
        self.X = self.vec.fit_transform(self.texts)
        w = np.asarray(self.X.sum(axis=0)).ravel()
        w = w / w.max() if w.max() > 0 else w
        self.weight = {t: float(w[j]) for t, j in self.vec.vocabulary_.items()}
        n, v = self.X.shape
        k = min(20, n - 1, v - 1)
        E = TruncatedSVD(n_components=k, random_state=0).fit_transform(self.X) if k >= 2 else self.X.toarray()
        nr = np.linalg.norm(E, axis=1, keepdims=True)
        self.E = E / np.where(nr == 0, 1, nr)

    def _tok_weight(self, phrase: str) -> float:
        vals = [self.weight.get(p, 0.0) for p in analyze(phrase)]
        return float(np.mean(vals)) if vals else 0.0

    def _ctx(self, idxs):
        v = self.E[sorted(idxs)].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v

    def _importance(self, d=0.85, iters=50):
        S = (self.X @ self.X.T).toarray()
        np.fill_diagonal(S, 0)
        rs = S.sum(axis=1, keepdims=True)
        P = np.divide(S, rs, out=np.zeros_like(S), where=rs > 0)
        r = np.full(len(S), 1.0 / len(S))
        for _ in range(iters):
            r = (1 - d) / len(S) + d * P.T @ r
        r = r / r.max()
        self.imp = []
        for i, p in enumerate(self.sentences):
            pos = 1.0 / (1 + 0.2 * i)
            has = 1.0 if p.frame else 0.5
            self.imp.append(0.65 * float(r[i]) + 0.2 * has + 0.15 * pos)
        m = max(self.imp)
        self.imp = [x / m for x in self.imp]

    # ---------------- entity inventory (for distractors & keyphrases) ---------------- #
    def _core(self, p, np_):
        """(char_start, char_end, text) of the NP core: no determiner and no generic leading adjectives."""
        toks = p.toks[np_.cs:np_.e]
        while len(toks) > 1 and toks[0].tag == "ADJ" and toks[0].low in GENERIC_ADJ:
            toks = toks[1:]
        st, en = toks[0].start, p.toks[np_.e - 1].end
        txt = p.text[st:en]
        t0 = toks[0]
        if t0.i == 0 and t0.tag != "PROPN" and np_.etype in ("CONCEPT", "QUANTITY") and not txt[:2].isupper():
            txt = txt[0].lower() + txt[1:]
        return st, en, txt

    def _inventory(self):
        self.pool = {"PERSON": {}, "PLACE": {}, "NAMED": {}, "CONCEPT": {}}
        self.years, self.decades = set(), set()
        counts = {}
        for p in self.sentences:
            for np_ in p.nps:
                et = np_.etype
                if et == "PRONOUN":
                    continue
                _, _, txt = self._core(p, np_)
                key = txt.lower()
                if et == "YEAR":
                    m = re.match(r"(\d{4})(s?)$", txt)
                    if m:
                        (self.decades if m.group(2) else self.years).add(int(m.group(1)))
                    continue
                if et == "QUANTITY":
                    continue
                if et == "PERSON":
                    full = self.A.surnames.get(key, txt) if len(txt.split()) == 1 else txt
                    key, txt = full.lower(), full
                    bucket = "PERSON"
                elif et in ("PLACE", "ORG"):
                    bucket = "PLACE"
                elif et == "NAMED":
                    bucket = "NAMED"
                else:
                    head = p.toks[np_.head]
                    if head.tag not in ("NOUN", "ING") or len(txt.split()) > 3 or self._tok_weight(txt) <= 0:
                        continue
                    bucket = "CONCEPT"
                e = self.pool[bucket].setdefault(key, {"surface": txt, "sents": set()})
                e["sents"].add(p.idx)
                counts[(bucket, key)] = counts.get((bucket, key), 0) + 1
            f = p.frame
            for fr in p.frames:  # all-caps names after at/in are places/organisations (CERN)
                for sg in fr.segs:
                    if sg.kind == "pp" and sg.prep in ("at", "in") and sg.np and sg.np.etype == "NAMED":
                        _, _, t = self._core(p, sg.np)
                        if t.isupper() and len(t) > 2:
                            self.pool["PLACE"].setdefault(t.lower(), {"surface": t, "sents": set()})["sents"].add(p.idx)
        # keep only informative concepts
        sc = [self._tok_weight(v["surface"]) for v in self.pool["CONCEPT"].values()]
        thr = float(np.percentile(sc, 50)) if sc else 0.0
        self.pool["CONCEPT"] = {k: v for k, v in self.pool["CONCEPT"].items()
                                if self._tok_weight(v["surface"]) >= thr and not (set(k.split()) & ANAPHORIC)}
        # key concepts (shown in the UI)
        scored = []
        for (b, k), c in counts.items():
            e = self.pool[b].get(k)
            if not e:
                continue
            w = self._tok_weight(e["surface"]) + 0.05
            scored.append((c * w * (1.4 if b in ("NAMED", "PERSON") else 1.0), e["surface"], b))
        scored.sort(reverse=True)
        seen, self.key_concepts = set(), []
        for _, sf, b in scored:
            ws = words(sf)
            if ws and not (ws & seen):
                self.key_concepts.append(cap1(sf))
                seen |= ws
            if len(self.key_concepts) >= 8:
                break

    # ---------------- distractors ---------------- #
    def _pick(self, bucket: str, answer: str, sidx: int, k: int = 3, extra_pool=None) -> Optional[list]:
        pool = dict(self.pool[bucket])
        for b in extra_pool or []:
            pool.update(self.pool[b])
        akey = answer.lower()
        atoks = words(akey)
        av = self._ctx({sidx})
        scored = []
        for key, e in pool.items():
            if key == akey or key in self.low[sidx] or (words(key) & atoks):
                continue
            wd = abs(len(key.split()) - len(akey.split()))
            if wd > 1:
                continue
            sim = float(self._ctx(e["sents"]) @ av)
            if sim > 0.96:
                continue
            scored.append((sim - 0.15 * wd + self.rng.random() * 0.01, e["surface"]))
        scored.sort(reverse=True)
        out, used = [], {w.rstrip("s") for w in atoks}
        for _, sf in scored:
            ws = {w.rstrip("s") for w in words(sf)}
            if ws & used:
                continue
            out.append(sf)
            used |= ws
            if len(out) == k:
                return out
        return None

    def _year_distractors(self, txt: str, k: int = 3) -> Optional[list]:
        m = re.match(r"(\d{4})(s?)$", txt)
        if not m:
            return None
        y, dec = int(m.group(1)), m.group(2)
        near_pool = self.decades if dec else self.years
        out = sorted((v for v in near_pool if v != y), key=lambda v: abs(v - y))[:k - 1]
        offs = [-30, -20, -10, 10, 20, 30] if dec else [-20, -10, -5, -3, -2, 2, 3, 5, 10, 15, 25]
        self.rng.shuffle(offs)
        for o in offs:
            if len(out) >= k:
                break
            if y + o not in out and 0 < y + o <= 2100:
                out.append(y + o)
        return [f"{v}{dec}" for v in out[:k]] if len(out) >= k else None

    def _num_distractors(self, num: str, unit: str = "", k: int = 3) -> Optional[list]:
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            return None
        is_int = float(val).is_integer() and "." not in num
        factors = [0.5, 0.75, 1.5, 2, 3, 0.1, 10, 1.25]
        self.rng.shuffle(factors)
        out = []
        for f in factors:
            v = val * f
            s = (f"{int(round(v)):,}" if "," in num else str(int(round(v)))) if is_int else f"{v:.1f}"
            full = (s + (" " + unit if unit else ""))
            if v > 0 and s != num and full not in out and not (is_int and int(round(v)) == int(val)):
                out.append(full)
            if len(out) == k:
                return out
        return None

    # ---------------- text helpers ---------------- #
    def _leaks(self, stem: str, answer: str) -> bool:
        low = " " + re.sub(r"[^\w\s'-]", " ", stem.lower()) + " "
        toks = [w for w in re.findall(r"[\w'-]+", answer.lower()) if len(w) > 3 and w not in STOP]
        if not toks:
            toks = [answer.lower()]
        return any(f" {t} " in low or f" {t}s " in low or f" {t}'s " in low for t in toks)

    def _clip(self, p, s, e) -> str:
        return p.span(s, e)

    def _lead_tail(self, p, f) -> str:
        if not f.lead:
            return ""
        t0 = p.toks[0]
        if t0.tag != "PREP":
            return ""
        txt = p.span(*f.lead)
        return txt[0].lower() + txt[1:] if p.toks[1].tag != "PROPN" else txt[0].lower() + txt[1:]

    def _others(self, p, f, skip=(), skip_lead=False, skip_kinds=("tail", "coord")) -> Optional[str]:
        parts = []
        for sg in f.segs:
            if sg.kind in skip_kinds or id(sg) in skip:
                continue
            if sg.kind == "pp" and sg.np is None:
                return None
            parts.append(p.span(sg.s, sg.e))
        if not skip_lead:
            lt = self._lead_tail(p, f)
            if lt:
                parts.append(lt)
        txt = " ".join(x for x in parts if x).strip()
        return txt if len(txt.split()) <= 20 else None

    @staticmethod
    def _aux(f) -> str:
        return {"past": "did", "3sg": "does"}.get(f.verb.vform, "do")

    def _group(self, p, f):
        return p.toks[f.vg[0]:f.vg[1]]

    def _simple_active(self, p, f) -> bool:
        g = self._group(p, f)
        return (not f.passive and not f.copula and f.verb.tag == "VERB" and f.verb.vform in ("past", "3sg", "base")
                and all(t.tag == "ADV" or t is f.verb for t in g) and not f.neg and self._subj_ok(f))

    def _simple_passive(self, p, f) -> bool:
        g = [t for t in self._group(p, f) if t.tag != "ADV"]
        return f.passive and len(g) == 2 and g[0].tag == "BE" and g[1] is f.verb and not f.neg and self._subj_ok(f)

    @staticmethod
    def _subj_ok(f) -> bool:
        return not (f.subj.etype == "PRONOUN" and f.subj.resolved == f.subj.text) and not f.anaphoric

    def _mcq(self, stem, answer, distractors, **kw) -> Optional[Question]:
        if not distractors or len(distractors) < 3:
            return None
        opts = [answer] + list(distractors[:3])
        opts = [cap1(o) for o in opts]
        ans = opts[0]
        if len({o.lower() for o in opts}) < 4:
            return None
        self.rng.shuffle(opts)
        return Question("mcq", stem, ans, options=opts, answer_letter="ABCD"[opts.index(ans)], **kw)

    # ---------------- frame -> question candidates ---------------- #
    def _frame_cands(self, p, f) -> list:
        out, S = [], p.text
        if not self._subj_ok(f):
            return out
        sm = f.subj.mid or f.subj.text
        about = cap1(self._core(p, f.subj)[2]) if f.subj.etype != "PRONOUN" else ""
        if f.subj.resolved != f.subj.text:
            about = cap1(f.subj.resolved)
        base = dict(source=S, sentence_idx=p.idx)

        def add(q, kind, answers_for_leak=None):
            if not q:
                return
            q.kind = kind
            if not q.about:
                q.about = about
            if len(q.question.split()) < 5 or self._leaks(q.question, q.answer):
                return
            if set(re.findall(r"[a-z]+", q.question.lower())) & DISC:
                return
            q.score = WEIGHT[kind] * (0.55 + 0.45 * self.imp[p.idx])
            out.append(q)

        sa, sp = self._simple_active(p, f), self._simple_passive(p, f)
        aux = self._aux(f)
        lem = f.verb.lemma

        # ---------- WHEN ----------
        times = []
        if f.lead_np is not None and f.lead_np.etype == "YEAR" and p.toks[0].low in ("in", "by", "around", "since", "until", "during"):
            times.append((None, f.lead_np, p.toks[0].low))
        for sg in f.segs:
            if sg.kind == "pp" and sg.np and sg.np.etype == "YEAR" and sg.prep in ("in", "by", "around", "since", "until"):
                times.append((sg, sg.np, sg.prep))
        if len(times) == 1 and (sa or sp):
            sg, tnp, prep = times[0]
            ans = self._core(p, tnp)[2]
            unit = "decade" if ans.endswith("s") else "year"
            oth = self._others(p, f, skip={id(sg)} if sg else (), skip_lead=sg is None)
            if oth is not None:
                head = f"{cap1(prep)} which {unit}"
                if sa:
                    body = f"{aux} {sm} {lem}"
                else:
                    be = next(t.low for t in self._group(p, f) if t.tag == "BE")
                    body = f"{be} {sm} {f.verb.text}"
                stem = f"{head} {body} {oth}".strip() + "?"
                add(Question("wh", stem, ans, accept=[ans], **base), "WHEN")
                add(self._mcq(stem, ans, self._year_distractors(ans), **base), "WHEN")

        # ---------- WHERE ----------
        places = [sg for sg in f.segs if sg.kind == "pp" and sg.np and sg.prep in ("in", "at")
                  and (sg.np.etype in ("PLACE", "ORG") or (sg.np.etype == "NAMED" and self._core(p, sg.np)[2].isupper()))]
        if len(places) == 1 and (sa or sp):
            sg = places[0]
            ans = self._core(p, sg.np)[2]
            oth = self._others(p, f, skip={id(sg)})
            if oth is not None:
                if sa:
                    stem = f"Where {aux} {sm} {lem} {oth}".strip() + "?"
                else:
                    be = next(t.low for t in self._group(p, f) if t.tag == "BE")
                    stem = f"Where {be} {sm} {f.verb.text} {oth}".strip() + "?"
                acc = [ans, "the " + ans]
                add(Question("wh", stem, ans, accept=acc, **base), "WHERE")
                add(self._mcq(stem, ans, self._pick("PLACE", ans, p.idx), **base), "WHERE")

        # ---------- AGENT of a passive ----------
        by = [sg for sg in f.segs if sg.kind == "pp" and sg.prep == "by" and sg.np]
        if sp and len(by) == 1:
            agents = by[0].nps
            oth = self._others(p, f, skip={id(by[0])})
            past = past_tense(lem, f.verb.text)
            if oth is not None and all(a.etype == "PERSON" for a in agents):
                names = [self._core(p, a)[2] for a in agents]
                ans = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
                acc = [ans] + names + [n.split()[-1] for n in names]
                stem = f"Who {past} {sm} {oth}".strip() + "?"
                add(Question("wh", stem, ans, accept=acc, **base), "WHO_AGENT")
                if len(names) == 1:
                    add(self._mcq(stem, ans, self._pick("PERSON", ans, p.idx), **base), "WHO_AGENT")
            elif oth is not None and len(agents) == 1 and agents[0].etype == "CONCEPT" and not agents[0].plural \
                    and p.toks[agents[0].s].tag not in ("DET", "POSS") and p.toks[agents[0].head].low not in HUMAN_NOUNS:
                be = next(t.low for t in self._group(p, f) if t.tag == "BE")
                ans = self._core(p, agents[0])[2]
                stem = f"How {be} {sm} {f.verb.text} {oth}".strip() + "?"
                add(Question("wh", stem, ans, accept=[ans, "by " + ans], **base), "HOW")
            elif oth is not None and len(agents) == 1 and p.toks[agents[0].head].low in HUMAN_NOUNS:
                ans = self._core(p, agents[0])[2]
                stem = f"Who {past} {sm} {oth}".strip() + "?"
                add(Question("wh", stem, ans, accept=[ans, p.toks[agents[0].head].text], **base), "WHO_AGENT")
            elif oth is not None and len(agents) == 1 and agents[0].etype in ("CONCEPT", "NAMED", "ORG"):
                ans = self._core(p, agents[0])[2]
                bucket = "CONCEPT" if agents[0].etype == "CONCEPT" else "NAMED"
                wh = "Which organization" if agents[0].etype == "ORG" else "What"
                stem = f"{wh} {past} {sm} {oth}".strip() + "?"
                add(Question("wh", stem, ans, accept=[ans], **base), "WHAT_AGENT")
                add(self._mcq(stem, ans, self._pick(bucket, ans, p.idx), **base), "WHAT_AGENT")

        # ---------- WHO / WHAT as subject ----------
        rest_s, rest_e = f.vg[0], f.clause_end
        rest = p.span(rest_s, rest_e)
        lead_ok = (not f.lead) or p.toks[0].tag == "PREP"
        if rest and 3 <= len(rest.split()) <= 24 and lead_ok and not f.neg and not f.copula and f.verb.tag == "VERB":
            lead = p.span(*f.lead) if f.lead else ""
            def compose(wh):
                return (f"{lead}, {wh.lower()} {rest}?" if lead else f"{wh} {rest}?")
            hh = p.toks[f.subj.head]
            human = hh.low in HUMAN_NOUNS or (hh.text[0].isupper() and hh.low.endswith(("ians", "ites", "ese")))
            if human and f.subj.etype != "PERSON" and not f.passive and f.subj.resolved == f.subj.text \
                    and p.toks[f.subj.s].tag != "POSS":
                ans = self._core(p, f.subj)[2]
                if f.verb.vform == "base":      # plural subject -> singular "who"
                    st0 = p.toks[rest_s].start
                    vt = f.verb
                    rest = rest[:vt.start - st0] + third_person(vt.lemma) + rest[vt.end - st0:]
                add(Question("wh", compose("Who"), ans, accept=[ans, hh.text], **base), "WHO_SUBJ")
            elif f.subj.etype == "PERSON" and not f.passive:
                ans = f.subj.resolved
                acc = [ans, ans.split()[-1]]
                stem = compose("Who")
                add(Question("wh", stem, ans, accept=acc, **base), "WHO_SUBJ")
                add(self._mcq(stem, ans, self._pick("PERSON", ans, p.idx), **base), "WHO_SUBJ")
            elif f.subj.etype in ("CONCEPT", "NAMED", "ORG") and p.toks[f.subj.s].tag != "POSS" \
                    and len(f.subj.core.split()) <= 5 and f.subj.resolved == f.subj.text:
                ans = self._core(p, f.subj)[2]
                bucket = "CONCEPT" if f.subj.etype == "CONCEPT" else "NAMED"
                stem = compose("What")
                add(Question("wh", stem, ans, accept=[ans, f.subj.text], **base), "WHAT_SUBJ")
                add(self._mcq(stem, ans, self._pick(bucket, ans, p.idx), **base), "WHAT_SUBJ")

        # ---------- QUANTITY ----------
        if sa:
            for sg in f.segs:
                if sg.kind == "obj" and sg.np and sg.np.etype == "QUANTITY":
                    toks = p.toks[sg.np.cs:sg.np.e]
                    if not toks or toks[0].tag != "NUM" or not re.match(r"\d", toks[0].text) or len(toks) < 2:
                        continue
                    num, tail = toks[0].text, toks[1:]
                    unit = tail[-1].low if tail[-1].low in UNITS else None
                    oth = self._others(p, f, skip={id(sg)})
                    if oth is None:
                        continue
                    if unit and len(tail) == 1:
                        if lem not in MEASURE_VERBS:
                            continue
                        stem = f"{MEASURE_VERBS[lem]} {aux} {sm} {lem} {oth}".strip() + "?"
                        ans = f"{num} {unit}"
                        dis = self._num_distractors(num, unit)
                    else:
                        noun = p.span(tail[0].i, sg.np.e)
                        stem = f"How many {noun} {aux} {sm} {lem} {oth}".strip() + "?"
                        ans = num
                        dis = self._num_distractors(num)
                    acc = [ans, num]
                    add(Question("wh", stem, ans, accept=acc, **base), "QUANT")
                    add(self._mcq(stem, ans, dis, **base), "QUANT")

        # ---------- DEFINITION ----------
        objs = [sg for sg in f.segs if sg.kind == "obj"]
        if f.copula and not f.neg and objs and f.subj.etype in ("CONCEPT", "NAMED") and f.subj.resolved == f.subj.text:
            first = objs[0]
            comp = p.span(first.s, f.clause_end)
            if p.toks[first.s].tag == "DET" and 4 <= len(comp.split()) <= 26 and " one of " not in f" {comp.lower()} " \
                    and not first.np.core.lower().startswith(("first", "number")):
                be = next(t.low for t in self._group(p, f) if t.tag == "BE")
                ans = self._core(p, f.subj)[2]
                bucket = "CONCEPT" if f.subj.etype == "CONCEPT" else "NAMED"
                add(Question("wh", f"What {be} {comp}?", ans, accept=[ans, f.subj.text], **base), "DEFINITION")
                add(self._mcq(f"Which of the following {be} {comp}?", ans, self._pick(bucket, ans, p.idx), **base), "DEFINITION")

        # ---------- NAMING ("an observation known as Moore's Law") ----------
        if f.naming and f.appos:
            head = p.toks[f.appos.head].low
            that = [sg for sg in f.segs if sg.kind == "that"]
            if head in CLAIM_NOUNS and that and f.verb.lemma in CLAIM_VERBS:
                clause = p.span(that[0].s + 1, that[0].e)
                kw = next((t.low for t in p.toks[f.appos.e:] if t.low in ("known", "called", "termed", "named")), "known")
                tail_word = "called" if kw in ("called", "named", "termed") else "known as"
                stem = f"What is the {head} that {clause} {tail_word}?"
                ans = self._core(p, f.naming)[2]
                acc = [ans, re.sub(r"['’]s\b", "", ans)]
                add(Question("wh", stem, ans, accept=acc, about=ans, **base), "NAMING")
                add(self._mcq(stem, ans, self._pick("NAMED", ans, p.idx), about=ans, **base), "NAMING")
        return out

    # ---------------- resolved sentence, true/false, cloze ---------------- #
    def _reps_for_resolution(self, p, f) -> list:
        """Character replacements that make the sentence stand alone (pronoun / surname resolution)."""
        reps = []
        if f and f.subj.resolved and f.subj.resolved != f.subj.text:
            st, en = p.toks[f.subj.s].start, p.toks[f.subj.e - 1].end
            txt = f.subj.resolved if f.subj.s == 0 else (f.subj.mid or f.subj.resolved)
            reps.append((st, en, txt))
        return reps

    @staticmethod
    def _apply(text: str, reps: list) -> str:
        for st, en, new in sorted(reps, key=lambda r: -r[0]):
            text = text[:st] + new + text[en:]
        return text

    def _clean_for_statement(self, p, f) -> bool:
        if not f or not 7 <= len(p.toks) <= 34 or not self._subj_ok(f):
            return False
        # unresolved pronouns elsewhere make the statement ungrounded
        for t in p.toks[f.subj.e:]:
            if t.tag in ("PRON",) and t.low in DISCOURSE_PRON:
                return False
        if any(t.low in DISC for t in p.toks):
            return False
        return p.toks[0].low not in ("however", "thus", "therefore", "moreover", "also", "but")

    def _tf_cands(self, p) -> list:
        f = p.frame
        if not self._clean_for_statement(p, f):
            return []
        about = cap1(f.subj.resolved) if f.subj.etype != "PRONOUN" or f.subj.resolved != f.subj.text else ""
        base_reps = self._reps_for_resolution(p, f)
        true_text = self._apply(p.text, base_reps)
        w = WEIGHT["TF"] * (0.55 + 0.45 * self.imp[p.idx])
        out = [Question("tf", true_text, "True", about=about, kind="TF", source=p.text, sentence_idx=p.idx, score=w)]
        sw = self._swap(p, f, base_reps)
        if sw:
            new_text, what = sw
            out.append(Question("tf", new_text, "False", about=about, kind="TF", source=p.text,
                                sentence_idx=p.idx, score=w * 1.05))
        return out

    def _swap(self, p, f, base_reps) -> Optional[tuple]:
        """Make a plausible FALSE statement by replacing one role-bearing entity with a same-type one."""
        def reps_with(st, en, new):
            r = [x for x in base_reps if not (x[0] < en and st < x[1])]
            r.append((st, en, new))
            return self._apply(p.text, r)

        s_idx = p.idx
        nps = [(sg.np, sg.prep) for sg in f.segs if sg.np] + ([(f.lead_np, "")] if f.lead_np else [])
        # 1. year
        for np_, prep in nps:
            if np_.etype == "YEAR":
                txt = self._core(p, np_)[2]
                d = self._year_distractors(txt, 3)
                if d:
                    st, en, _ = self._core(p, np_)
                    return reps_with(st, en, d[0]), "year"
        # 2. person (subject or single agent)
        pers = [f.subj] if f.subj.etype == "PERSON" else []
        pers += [sg.np for sg in f.segs if sg.kind == "pp" and sg.prep == "by" and len(sg.nps) == 1 and sg.np.etype == "PERSON"]
        for np_ in pers:
            name = f.subj.resolved if np_ is f.subj else self._core(p, np_)[2]
            d = self._pick("PERSON", name, s_idx, 1)
            if d:
                if np_ is f.subj:
                    st, en = p.toks[np_.s].start, p.toks[np_.e - 1].end
                else:
                    st, en, _ = self._core(p, np_)
                return reps_with(st, en, d[0]), "person"
        # 3. place
        for np_, prep in nps:
            if prep in ("in", "at") and (np_.etype in ("PLACE", "ORG")):
                txt = self._core(p, np_)[2]
                d = self._pick("PLACE", txt, s_idx, 1)
                if d:
                    st, en, _ = self._core(p, np_)
                    return reps_with(st, en, d[0]), "place"
        # 4. quantity
        for np_, prep in nps:
            if np_.etype == "QUANTITY":
                toks = p.toks[np_.cs:np_.e]
                if toks and re.match(r"\d", toks[0].text):
                    d = self._num_distractors(toks[0].text)
                    if d:
                        return reps_with(toks[0].start, toks[0].end, d[0]), "quantity"
        # 5. named entity / concept object
        for np_, prep in nps:
            if np_.etype in ("NAMED", "CONCEPT") and np_.head >= 0:
                st, en, txt = self._core(p, np_)
                bucket = "NAMED" if np_.etype == "NAMED" else "CONCEPT"
                d = self._pick(bucket, txt, s_idx, 3)
                if not d:
                    continue
                det = p.toks[np_.s].low if p.toks[np_.s].tag == "DET" else ""
                if det in ("each", "every", "all", "some", "many", "no", "any", "both") or " of " in f" {txt.lower()} ":
                    continue
                for cand in d:
                    plural_c, plural_a = cand.lower().endswith("s"), np_.plural
                    if det in ("a", "an") and plural_c:
                        continue
                    if det in ("a", "an") and (cand[0].lower() in "aeiou") != (det == "an"):
                        continue
                    if bucket == "CONCEPT" and plural_c != plural_a:
                        continue
                    return reps_with(st, en, cand), "term"
        return None

    def _cloze_cands(self, p) -> list:
        f = p.frame
        if p.toks[0].low in ("however", "thus", "therefore", "moreover") and not f:
            return []
        if not (7 <= len(p.toks) <= 34):
            return []
        if f and not self._clean_for_statement(p, f):
            return []
        if not f and any(t.tag == "PRON" and t.low in DISCOURSE_PRON for t in p.toks):
            return []
        reps = self._reps_for_resolution(p, f)
        out = []
        for np_ in p.nps:
            if np_.etype in ("PRONOUN",):
                continue
            st, en, txt = self._core(p, np_)
            if any(r[0] < en and st < r[1] for r in reps):
                continue
            if len(txt.split()) > 3:
                continue
            if p.text.lower().count(txt.lower()) != 1:
                continue
            et = np_.etype
            head = p.toks[np_.head]
            if et == "CONCEPT" and (head.tag != "NOUN" or self._tok_weight(txt) < 0.4 or len(txt) < 5):
                continue
            boost = {"PERSON": 1.2, "NAMED": 1.2, "YEAR": 1.1, "QUANTITY": 1.0, "PLACE": 1.1, "ORG": 1.1, "CONCEPT": 1.0}[et]
            wt = 0.5 + (self._tok_weight(txt) if et == "CONCEPT" else 0.5)
            stem_text = self._apply(p.text, reps + [(st, en, BLANK)]) if not reps else None
            if stem_text is None:
                # blank in the resolved text: shift by the length change of earlier replacements
                shift = sum(len(r[2]) - (r[1] - r[0]) for r in reps if r[1] <= st)
                new = self._apply(p.text, reps)
                stem_text = new[:st + shift] + BLANK + new[en + shift:]
            ans = f.subj.resolved if (f and et == "PERSON" and np_ is f.subj) else txt
            q = Question("cloze", "Fill in the blank: " + stem_text, ans, accept=[ans], source=p.text, sentence_idx=p.idx,
                         about=cap1(f.subj.resolved) if f and f.subj.etype != "PRONOUN" else "", kind="CLOZE")
            q.score = WEIGHT["CLOZE"] * (0.55 + 0.45 * self.imp[p.idx]) * boost * wt
            out.append(q)
        out.sort(key=lambda q: -q.score)
        return out[:3]

    # ---------------- selection ---------------- #
    def generate(self, n: int = 10, types=("mcq", "tf", "wh", "cloze")) -> list:
        cands = []
        for p in self.sentences:
            for f in p.frames:
                cands += self._frame_cands(p, f)
            cands += self._tf_cands(p)
            cands += self._cloze_cands(p)
        cands = [q for q in cands if q.qtype in types]
        # drop duplicates
        seen, uniq = set(), []
        for q in sorted(cands, key=lambda q: -q.score):
            k = (q.question.lower(), q.qtype)
            if k not in seen:
                seen.add(k)
                uniq.append(q)
        cands = uniq
        order = ["mcq", "wh", "tf", "mcq", "wh", "tf", "mcq", "cloze"] if set(types) >= {"mcq", "tf", "wh", "cloze"} else list(types)
        chosen, used_ans, per_sent, per_kind = [], set(), {}, {}

        def usable(q, strict):
            a = q.answer.lower()
            if a in used_ans:
                return False
            if per_sent.get(q.sentence_idx, 0) >= (1 if strict else 2):
                return False
            return True

        for rnd in range(2):
            ti = 0
            while len(chosen) < n:
                want = order[ti % len(order)]
                ti += 1
                pool = [q for q in cands if q not in chosen and usable(q, rnd == 0) and q.qtype == want]
                if not pool:
                    pool = [q for q in cands if q not in chosen and usable(q, rnd == 0)]
                if not pool:
                    break
                best = max(pool, key=lambda q: q.score - 0.12 * per_kind.get(q.kind, 0))
                chosen.append(best)
                used_ans.add(best.answer.lower())
                per_sent[best.sentence_idx] = per_sent.get(best.sentence_idx, 0) + 1
                per_kind[best.kind] = per_kind.get(best.kind, 0) + 1
                if ti > 4 * n + 12:
                    break
            if len(chosen) >= n:
                break
        chosen.sort(key=lambda q: q.sentence_idx)
        return chosen


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
            ok = any(difflib.SequenceMatcher(None, g, re.sub(r"^(the|a|an)\s+", "", a.lower())).ratio() >= 0.85
                     for a in (q.accept or [q.answer]))
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
    gen = QuizGenerator(text, seed=a.seed)
    qs = gen.generate(a.num, types)
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
