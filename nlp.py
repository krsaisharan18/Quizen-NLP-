"""
nlp.py -- a small, dependency-free English analyser (no ML models, no downloads).

It gives the quiz generator a *linguistic* view of every sentence:

  tokenise -> POS-tag (lexicon + suffix rules + context rules) -> chunk noun phrases
  -> parse the clause (lead adverbial, subject, verb group, voice, objects, PPs, appositive)
  -> type the entities (PERSON / ORG / PLACE / NAMED / YEAR / QUANTITY / CONCEPT)
  -> resolve pronouns and surnames across sentences (light coreference).

It is intentionally rule-based and tuned for declarative, encyclopedic prose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*(?:%|s\b)?|[A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+)*|[^\w\s]")

DET = set("a an the this these those each every some any no all both another either neither such".split())
POSS = set("his her its their our my your".split())
PRON = set("it he she they we you i him them us me itself himself herself themselves".split())
REL = {"who", "whom", "which", "whose"}
PREP = set("""in on at by with from of for into onto over under between among during after before since until
through across against about around within without upon toward towards near per via like despite throughout
beyond along behind beneath above below as according including regarding concerning excluding depending""".split())
CONJ = {"and", "or", "but", "nor"}
SUB = set("because although though while whereas if when whenever unless so yet whether than how what where why".split())
BE = set("is are was were be been being am".split())
HAVE = set("has have had".split())
DO = set("do does did".split())
MODAL = set("can could will would shall should may might must".split())
ADVS = set("""not never also often then later still just only even already always usually generally widely roughly
approximately nearly almost far more most less very too quite rather soon eventually finally ago now today thus
therefore however moreover furthermore instead together again once back mainly largely primarily
significantly particularly especially currently previously recently originally initially subsequently
further simply actually really eventually altogether up down out off away forth apart""".split())
QUANT_ADV = set("about around roughly nearly almost approximately".split())
NUMWORDS = set("""one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion
trillion dozen""".split())
UNITS = set("""tons ton tonnes kilograms kg pounds lbs meters metres km kilometers kilometres miles years months days
hours minutes seconds percent percentage dollars euros degrees watts volts hertz bytes""".split())
ANAPHORIC = set("""reaction process result system method effect change step procedure approach event situation period work
idea concept phenomenon technique action activity development outcome mechanism problem issue case one latter former
stage cycle model theory""".split())
MEASURE_VERBS = {"weigh": "How much", "cost": "How much", "measure": "How long", "last": "How long"}

ADJ_WORDS = set("""first second third fourth fifth last next previous new old many several various large small big little
great modern early late single different general same main major certain complete abstract powerful further other own
whole entire real common recent current future possible important similar particular specific total full high low
long short best better basic key few much smaller faster larger greater higher lower longer shorter easier cheaper
stronger weaker worse bigger earlier newer older reliable simple complex huge tiny vast rapid slow quick free open
hard soft hot cold heavy light dark bright strong weak rich poor young ancient wide narrow deep thin thick""".split())
ADJ_SUFFIX = ("al", "ic", "ous", "ive", "able", "ible", "ful", "less", "ary", "ent", "ant", "ical", "ish")
NOUN_SUFFIX = ("tion", "sion", "ment", "ness", "ity", "ism", "ance", "ence", "ship", "ist", "ics", "ogy", "hood", "ure")
NATIONALITY = set("""german british french american english russian chinese japanese indian european african asian
italian spanish greek roman dutch swiss swedish australian canadian irish scottish welsh polish turkish arab arabic
egyptian korean mexican brazilian persian israeli soviet""".split())
MONTHS = set("""january february march april may june july august september october november december monday
tuesday wednesday thursday friday saturday sunday""".split())

ORG_WORDS = set("""university institute company corporation inc ltd association bank ministry party council society agency
committee department school college laboratory laboratories lab labs foundation organization organisation union army
navy museum academy club group press prize award act treaty empire dynasty project programme program parliament
congress senate court church temple hospital library observatory centre center""".split())
PLACE_WORDS = set("""river mountain mountains city island islands sea ocean republic kingdom valley lake desert bay gulf coast
province state county park street road bridge africa asia europe america antarctica australia india china japan france
germany england britain russia canada brazil mexico italy spain egypt paris london berlin tokyo rome york delhi
bengaluru beijing moscow earth pacific atlantic united states kingdom""".split())
FIRSTNAMES = set("""john william james george charles robert thomas david richard joseph henry edward arthur albert alan
alexander isaac nikola thomas michael peter paul mark stephen steven andrew daniel matthew anthony jack walter gordon
tim timothy benjamin samuel frank harold raymond louis leonard leo carl karl max niels erwin werner enrico ernest
francis frederick galileo johannes marie ada grace mary elizabeth anna margaret jane susan dorothy florence rosalind
lise barbara emmy katherine hedy virginia amelia sarah emily alice helen ruth rachel charlotte victoria catherine
bill steve larry sergey elon jeff mark linus dennis ken bjarne guido vint claude norbert john vannevar herman
tom jim joe bob ray sam ben dan nick luis jose juan pierre jean marco giovanni hans fritz ludwig wilhelm""".split())
PERSON_TITLES = {"dr", "mr", "mrs", "ms", "prof", "professor", "sir", "lady", "lord", "king", "queen", "president"}
PERSON_VERBS = set("""invent discover write propose observe introduce demonstrate design develop found create publish win
say argue believe describe formulate show prove establish pioneer coin lead build become work study teach born
state claim suggest note explain conduct perform receive award complete devise patent construct realize conceive
define originate help""".split())

NOUNISH = set("""design process study use work form plan name place change increase decrease record report research support
test rule order control result cause point model state view call need show walk run set play move build drive power
head offer answer experience focus aim claim access approach cost reach store limit share trade market mark rate rank
rise fall cover file code transfer turn train line list match mean surface issue image display document paper present
object subject end start step stage book fire water light sound note content figure influence function struggle
attempt benefit damage demand effect increase interest land lead lift mind object plant produce progress promise
question rest return sign speech stand stress surprise talk touch travel trial trust visit voice vote wave
welcome wish wonder""".split())

IRREG = {  # surface form -> (lemma, kind)
    "wrote": ("write", "past"), "written": ("write", "past"), "ran": ("run", "past"), "built": ("build", "past"),
    "made": ("make", "past"), "held": ("hold", "past"), "led": ("lead", "past"), "began": ("begin", "past"),
    "begun": ("begin", "past"), "took": ("take", "past"), "taken": ("take", "past"), "gave": ("give", "past"),
    "given": ("give", "past"), "saw": ("see", "past"), "seen": ("see", "past"), "found": ("find", "past"),
    "knew": ("know", "past"), "known": ("know", "past"), "became": ("become", "past"), "came": ("come", "past"),
    "went": ("go", "past"), "gone": ("go", "past"), "got": ("get", "past"), "said": ("say", "past"),
    "told": ("tell", "past"), "thought": ("think", "past"), "kept": ("keep", "past"), "left": ("leave", "past"),
    "meant": ("mean", "past"), "met": ("meet", "past"), "paid": ("pay", "past"), "sent": ("send", "past"),
    "spent": ("spend", "past"), "stood": ("stand", "past"), "understood": ("understand", "past"), "won": ("win", "past"),
    "drew": ("draw", "past"), "drawn": ("draw", "past"), "grew": ("grow", "past"), "grown": ("grow", "past"),
    "chose": ("choose", "past"), "chosen": ("choose", "past"), "broke": ("break", "past"), "broken": ("break", "past"),
    "rose": ("rise", "past"), "risen": ("rise", "past"), "fell": ("fall", "past"), "fallen": ("fall", "past"),
    "flew": ("fly", "past"), "flown": ("fly", "past"), "sang": ("sing", "past"), "spoke": ("speak", "past"),
    "spoken": ("speak", "past"), "sat": ("sit", "past"), "brought": ("bring", "past"), "bought": ("buy", "past"),
    "taught": ("teach", "past"), "caught": ("catch", "past"), "fought": ("fight", "past"), "sought": ("seek", "past"),
    "sold": ("sell", "past"), "shown": ("show", "past"), "lost": ("lose", "past"), "heard": ("hear", "past"),
    "put": ("put", "past"), "set": ("set", "past"), "read": ("read", "past"), "cut": ("cut", "past"),
    "hit": ("hit", "past"), "shut": ("shut", "past"), "ate": ("eat", "past"), "eaten": ("eat", "past"),
    "wore": ("wear", "past"), "worn": ("wear", "past"), "swept": ("sweep", "past"), "dealt": ("deal", "past"),
    "fed": ("feed", "past"), "forgot": ("forget", "past"), "forgotten": ("forget", "past"), "hid": ("hide", "past"),
    "lit": ("light", "past"), "rang": ("ring", "past"), "shook": ("shake", "past"), "slept": ("sleep", "past"),
    "stole": ("steal", "past"), "swam": ("swim", "past"), "threw": ("throw", "past"), "thrown": ("throw", "past"),
    "woke": ("wake", "past"), "wound": ("wind", "past"), "bore": ("bear", "past"), "born": ("bear", "past"),
    "does": ("do", "3sg"), "has": ("have", "3sg"), "goes": ("go", "3sg"),
}
PAST_OF = {}                                   # participle -> simple past (only where they differ)
for _form, (_lem, _k) in IRREG.items():
    pass
PAST_OF.update({"written": "wrote", "taken": "took", "given": "gave", "seen": "saw", "known": "knew",
                "begun": "began", "gone": "went", "drawn": "drew", "grown": "grew", "chosen": "chose",
                "broken": "broke", "risen": "rose", "fallen": "fell", "flown": "flew", "spoken": "spoke",
                "shown": "showed", "eaten": "ate", "worn": "wore", "forgotten": "forgot", "thrown": "threw",
                "born": "bore"})

VERBS = set("""remain stretch describe introduce replace demonstrate observe appear allow learn share propose rely define regard
manipulate intend enable involve represent consist depend determine perform operate store weigh double believe exist occur
increase decrease reduce improve develop discover invent complete publish establish announce expand extend connect combine
separate divide measure compare explain suggest prove solve compute calculate transmit receive deliver achieve reach emerge
evolve spread arrive declare adopt reject accept apply attach attract avoid belong carry claim collect compose conduct
consider continue contribute control cover decide depict destroy detect direct display distribute enter equal examine expect
experience explore express fail fill follow generate identify imagine indicate influence inform inspire install interpret
investigate join launch maintain manage mention name notice obtain offer open organize organise participate pass pay place predict
prefer prepare present prevent protect pull push raise realize realise recognize recognise record refer reflect relate release
remove repeat report require research respond restore retain return reveal rule save select send serve settle simulate specify
support supply surround survive test transfer transform translate treat trigger understand unite update utilize utilise visit
wait watch win use make produce convert contain include become call help create take give show cause form provide need begin
get go see find keep hold run work build lead move play turn bring write read teach study design plan change use grow draw
choose break rise fall fly sing speak sit buy catch fight seek sell lose hear put set cut hit eat wear feed hide light ring
shake sleep steal swim throw wake wind bear know think tell say come leave mean meet spend stand drive ride cost last happen
include begin cause lack need seem tend try turn want wish wonder mark rank rate limit link lift list match mine own pack
pair paint rely rent scan score sell shape sign skip sort spell spin split spot stage stand state stay stick store strike
stress stretch struggle submit succeed suffer suit supply swap switch tag tap target tie tip trace track trade train trap
travel trust tune twist upgrade vary view vote walk warn wash weave wrap yield accompany account accumulate acquire act
adapt add address adjust admit advance affect agree aim alter amount analyse analyze appeal approach approve argue arrange
assemble assess assign assist associate assume assure attempt attend attribute balance base bind blend boost borrow bound
branch breed bridge broadcast burn bury calculate capture cast cease celebrate challenge chart check circulate cite classify
clear climb close coin collapse colonize combine command commence commit communicate compete compile complicate comprise
conceive concentrate conclude confirm conquer conserve construct consume contact contrast convey cooperate coordinate copy
correspond count create cross cultivate cure damage deal debate decline dedicate defeat defend delay demand deny derive
describe deserve destroy devise devote differ digest discuss dismiss dispute dissolve distinguish dominate draft drain drop
dry earn educate elect eliminate embrace emit employ encode encounter encourage engage engineer ensure equip erect escape
estimate evaluate exceed exchange exclude execute exhibit expand expel exploit export expose fabricate face facilitate
feature fit fix flow focus fold force forge formulate found frame fuel function fund gain gather govern grant guide handle
harvest heat host house hunt hurt illustrate implement imply impose incorporate indicate induce infer inhabit inherit
initiate inject insert insist inspect integrate interact interfere introduce invade invest invite isolate issue judge
justify kill label land lay lease lecture legislate license lie lock locate log maintain map market master mature mediate
melt merge migrate mix modify monitor motivate mount name negotiate nominate normalise observe occupy operate oppose
order originate outline overcome overlap oversee own patent pattern permit persuade pioneer plant plot point pollute pose
possess pour power practise practice precede prescribe preserve press pretend prime print process proclaim profit program
progress project promote pronounce protest publish pump purchase pursue qualify quote range reach react realise rebuild
recall receive reckon recover recruit rectify redesign reduce reform refine refuse regard register regulate reinforce
relieve rely remember remind render renew replace reply represent reproduce request reserve reside resist resolve resume
retire reveal reverse review revise revolve reward rotate route run rush sample satisfy scatter schedule screen seal search
secure seek seize separate shift shine ship shoot shrink signal signify sketch slow smell solve sound sow span speak
sponsor spread stabilize steer stimulate stir stop strengthen stretch structure subdivide substitute suck suggest summarize
supervise supplement suppress surpass surrender suspect sustain sweep symbolize synthesize tackle taste tax tend terminate
testify theorize threaten tolerate total touch tour trace tolerate transport trigger triumph undergo undertake unify
upset urge utter validate value verify vibrate violate volunteer wage weigh welcome widen withdraw withstand witness
worship wound""".split())

DOUBLE = set("""stop plan drop ship rob tap trap trip skip slip step stir strip submit commit admit permit refer prefer
occur control omit transfer equip beg bat chat clap drag drum fit flip grab hug hum jog knit nod pat plug pop pot rub
sip sit slam spot swap tag tip wrap zip regret""".split())


def _inflect(base: str) -> dict:
    forms = {}
    b = base
    # 3rd person singular
    if re.search(r"(s|x|z|ch|sh|o)$", b):
        s3 = b + "es"
    elif re.search(r"[^aeiou]y$", b):
        s3 = b[:-1] + "ies"
    else:
        s3 = b + "s"
    # past / participle
    if b.endswith("e"):
        pa = b + "d"
    elif re.search(r"[^aeiou]y$", b):
        pa = b[:-1] + "ied"
    elif b in DOUBLE or re.fullmatch(r"[^aeiou]*[aeiou][^aeiouwxy]", b) and len(b) <= 4:
        pa = b + b[-1] + "ed"
    else:
        pa = b + "ed"
    # -ing
    if b.endswith("ie"):
        ing = b[:-2] + "ying"
    elif b.endswith("e") and not b.endswith(("ee", "ye", "oe")):
        ing = b[:-1] + "ing"
    elif b in DOUBLE or re.fullmatch(r"[^aeiou]*[aeiou][^aeiouwxy]", b) and len(b) <= 4:
        ing = b + b[-1] + "ing"
    else:
        ing = b + "ing"
    forms[b] = "base"
    forms[s3] = "3sg"
    forms[pa] = "past"
    forms[ing] = "ing"
    return forms


VERBFORMS: dict = {}
for _v in VERBS:
    for _f, _k in _inflect(_v).items():
        VERBFORMS.setdefault(_f, (_v, _k))
for _f, (_l, _k) in IRREG.items():
    VERBFORMS[_f] = (_l, _k)


def third_person(lemma: str) -> str:
    for form, kind in _inflect(lemma).items():
        if kind == "3sg":
            return form
    return lemma


def past_tense(lemma: str, participle: str) -> str:
    """Simple-past form given a participle (needed to turn passives into active questions)."""
    return PAST_OF.get(participle.lower(), participle)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Tok:
    i: int
    text: str
    low: str
    start: int
    end: int
    tag: str = "NOUN"
    lemma: str = ""
    vform: str = ""
    year: bool = False
    vk: Optional[tuple] = None      # candidate verb reading (lemma, kind)


@dataclass
class NP:
    s: int
    e: int                      # exclusive
    head: int
    etype: str = "CONCEPT"      # PERSON ORG PLACE NAMED YEAR QUANTITY CONCEPT PRONOUN
    text: str = ""
    core: str = ""              # without determiner / quantifier adverb
    plural: bool = False
    resolved: str = ""          # after coreference
    cs: int = 0                 # first token of the core (after determiner / quantifier adverb)
    mid: str = ""               # form usable mid-sentence (decapitalised when common noun)


@dataclass
class Seg:
    kind: str                   # obj | pp | adj | that | inf | tail | coord
    s: int
    e: int
    prep: str = ""
    np: Optional[NP] = None
    nps: list = field(default_factory=list)   # for coordinated agent lists


@dataclass
class Frame:
    lead: Optional[tuple]       # (s, e) token span of fronted adverbial (without comma)
    lead_np: Optional[NP]
    subj: NP
    vg: tuple                   # (s, e) verb group span
    verb: Tok
    passive: bool
    copula: bool
    neg: bool
    segs: list
    clause_end: int             # token index where the main clause stops (before tail comma / end)
    appos: Optional[NP] = None
    naming: Optional[NP] = None
    anaphoric: bool = False     # subject refers back to something not named in the sentence


@dataclass
class Parsed:
    idx: int
    text: str
    toks: list
    nps: list
    frame: Optional[Frame] = None
    frames: list = field(default_factory=list)

    def span(self, s: int, e: int) -> str:
        return self.text[self.toks[s].start:self.toks[e - 1].end] if e > s else ""


# --------------------------------------------------------------------------- #
# The analyser
# --------------------------------------------------------------------------- #
class Analyzer:
    def __init__(self, sentences: list):
        self.parsed = []
        raw = [[m for m in TOKEN_RE.finditer(s)] for s in sentences]
        self.seen_lower = set()
        for ms in raw:
            for k, m in enumerate(ms):
                w = m.group()
                if w[0].islower():
                    self.seen_lower.add(w.lower())
        for idx, (s, ms) in enumerate(zip(sentences, raw)):
            toks = [Tok(k, m.group(), m.group().lower(), m.start(), m.end()) for k, m in enumerate(ms)]
            self._tag(toks)
            p = Parsed(idx, s, toks, [])
            p.nps = self._all_nps(toks, p)
            self.parsed.append(p)
        self.persons: dict = {}           # lower full name -> surface
        self.surnames: dict = {}          # lower surname -> surface full name
        for p in self.parsed:
            p.frame = self._parse(p)
        self._infer_persons()
        for p in self.parsed:
            for np_ in p.nps:
                self._type(np_, p)
            for fr in getattr(p, "frames", []):
                for np_ in [fr.subj, fr.appos, fr.naming, fr.lead_np] + \
                           [x for sg in fr.segs for x in ([sg.np] if sg.np else []) + sg.nps]:
                    if np_:
                        self._type(np_, p)
        self._coref()

    # ----------------------------- tagging ----------------------------- #
    def _tag(self, toks: list):
        n = len(toks)
        for t in toks:
            self._lex(t, toks)
        # context pass: resolve verb candidates
        verb_seen = False
        for i, t in enumerate(toks):
            if t.tag in ("REL", "SUB", "THAT", "COMMA"):
                verb_seen = False                      # a new clause starts
            if t.tag in ("BE", "HAVE", "DO", "MODAL"):
                verb_seen = True
            if t.tag != "VCAND":
                if t.tag == "VERB":
                    verb_seen = True
                continue
            lemma, kind = t.vk
            j = i - 1
            while j >= 0 and toks[j].tag == "ADV":
                j -= 1
            prev = toks[j] if j >= 0 else None
            nxt = toks[i + 1] if i + 1 < n else None
            nxt_nounish = bool(nxt) and nxt.tag in ("NOUN", "ADJ", "PROPN", "ING", "NUM", "VCAND")
            nounish = lemma in NOUNISH
            later_verb = any(x.tag in ("BE", "HAVE", "DO", "MODAL", "VCAND") for x in toks[i + 1:])
            tag = "NOUN"
            if prev is None:
                tag = "ADJ" if kind == "past" and nxt_nounish else ("VERB" if kind == "past" and not nxt_nounish else "NOUN")
            elif prev.tag in ("TO", "MODAL", "DO"):
                tag = "VERB" if kind == "base" else "NOUN"
            elif prev.tag == "VERB" and kind == "base" and prev.vform == "past" and prev.lemma in ("help", "let", "make", "see", "watch", "hear"):
                tag = "VERB"
            elif prev.tag in ("BE", "HAVE"):
                tag = "VERB" if kind in ("past", "3sg", "base") else "ING"
                if kind == "ing":
                    tag = "ING"
            elif prev.tag in ("DET", "POSS", "ADJ", "PREP", "NUM"):
                if kind == "past" and nxt_nounish and prev.tag != "PREP":
                    tag = "ADJ"
                elif kind == "past" and nxt_nounish and prev.tag == "PREP":
                    tag = "ADJ"
                else:
                    tag = "NOUN"
            elif prev.tag in ("NOUN", "PROPN", "PRON", "ING", "REL", "THAT", "COMMA", "CONJ", "SUB"):
                if kind == "ing":
                    tag = "ING"
                elif verb_seen and prev.tag in ("NOUN", "PROPN", "ING") and nounish and kind != "past":
                    tag = "NOUN"
                elif nounish and not verb_seen and prev.tag in ("NOUN", "PROPN") and later_verb and kind != "past":
                    tag = "NOUN"
                elif kind == "past" and prev.tag in ("COMMA",) and nxt_nounish:
                    tag = "ADJ"
                elif prev.tag == "COMMA" and kind == "base" and nounish:
                    tag = "NOUN"
                else:
                    tag = "VERB"
            else:
                tag = "VERB" if kind in ("past",) else "NOUN"
            if kind == "ing" and tag != "ING" and tag == "VERB":
                tag = "ING"
            t.tag = tag
            if tag == "VERB":
                t.lemma, t.vform = lemma, kind
                verb_seen = True
        for t in toks:
            if t.tag == "VCAND":
                t.tag = "NOUN"
        # "to" is an infinitive marker only before a verb; otherwise a preposition ("according to rules")
        for i, t in enumerate(toks):
            if t.tag == "TO":
                j = i + 1
                while j < n and toks[j].tag == "ADV":
                    j += 1
                if not (j < n and (toks[j].tag in ("BE", "HAVE") or (toks[j].tag == "VERB" and toks[j].vform == "base"))):
                    t.tag = "PREP"
        # quantity adverbs before numbers
        for i, t in enumerate(toks[:-1]):
            if t.low in QUANT_ADV and toks[i + 1].tag == "NUM":
                t.tag = "ADV"

    def _lex(self, t: Tok, toks: list):
        w, low, i = t.text, t.low, t.i
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if not re.match(r"\w", w):
            t.tag = "COMMA" if w == "," else "PUNCT"
            return
        if re.fullmatch(r"\d+(?:[.,]\d+)*(?:%|s)?", w):
            t.tag = "NUM"
            t.year = bool(re.fullmatch(r"(1\d{3}|20\d{2})s?", w))
            return
        if low in NUMWORDS:
            t.tag = "NUM"
            return
        if low == "to":
            t.tag = "TO"
            return
        if low == "that":
            t.tag = "DET" if (i == 0 or toks[i - 1].tag in ("PUNCT", "COMMA")) else "THAT"
            return
        if low in BE:
            t.tag = "BE"; return
        if low in HAVE:
            t.tag = "HAVE"; t.lemma = "have"; return
        if low in DO:
            t.tag = "DO"; return
        if low in MODAL:
            t.tag = "MODAL"; return
        if low in DET:
            t.tag = "DET"; return
        if low in POSS:
            t.tag = "POSS"; return
        if low in PRON:
            t.tag = "PRON"; return
        if low in REL:
            t.tag = "REL"; return
        if low in CONJ:
            t.tag = "CONJ"; return
        if low in SUB and low != "than":
            t.tag = "SUB"; return
        if low in PREP:
            t.tag = "PREP"; return
        if low == "than":
            t.tag = "SUB"; return
        if low in ADVS or (low.endswith("ly") and len(low) > 4 and low not in ("family", "assembly", "italy", "supply", "apply", "reply", "ally", "july")):
            t.tag = "ADV"; return
        if low in MONTHS:
            t.tag = "NOUN"; return
        if low in NATIONALITY:
            t.tag = "ADJ"; return
        cap = w[0].isupper()
        if cap and i > 0 and low.endswith(("ian", "ish", "ese", "ean", "ic")) and nxt is not None \
                and nxt.text[0].islower() and nxt.low not in PREP and nxt.low not in BE and nxt.low not in CONJ:
            t.tag = "ADJ"
            return
        if cap and i > 0 and not (toks[i - 1].text in ('"', "“", "‘") and False):
            t.tag = "PROPN"; return
        if cap and i == 0:
            seen_l = low in self.seen_lower
            nxt_is = nxt is not None
            if not seen_l and not low.endswith(("ing", "ed", "ly")) and not low.endswith(ADJ_SUFFIX + NOUN_SUFFIX):
                nlow = nxt.low if nxt_is else ""
                if (nlow in BE or nlow in HAVE or nlow in DO or nlow in MODAL or nlow in VERBFORMS
                        or (nxt_is and (nxt.text[0].isupper() or nlow in ADVS))):
                    t.tag = "PROPN"; return
            if not seen_l and w.isupper() and len(w) > 1:
                t.tag = "PROPN"; return
        if low in VERBFORMS and low not in ADJ_WORDS:
            lemma, kind = VERBFORMS[low]
            t.vk = (lemma, kind)
            t.tag = "VCAND"
            return
        if low.endswith("ed") and len(low) > 4 and low not in ADJ_WORDS and not low.endswith(("eed", "ied")):
            stem = low[:-2]
            lemma = stem + "e" if (stem + "e") in VERBS else stem
            if re.search(r"([^aeiou])\1$", stem) and stem[:-1] in VERBS:
                lemma = stem[:-1]
            t.vk = (lemma, "past")
            t.tag = "VCAND"
            return
        if low.endswith("ing") and len(low) > 4:
            t.tag = "ING"; return
        if low in ADJ_WORDS or low.endswith(ADJ_SUFFIX) and not low.endswith(NOUN_SUFFIX):
            t.tag = "ADJ"; return
        t.tag = "NOUN"

    # ----------------------------- chunking ----------------------------- #
    def _chunk(self, toks: list, i: int, p: Parsed, allow_of: bool = True) -> Optional[NP]:
        n = len(toks)
        j = i
        if j < n and toks[j].tag == "ADV" and toks[j].low in QUANT_ADV and j + 1 < n and toks[j + 1].tag == "NUM":
            j += 1
        start_core = j
        if j < n and toks[j].tag == "PRON":
            return self._mk(p, i, j + 1, j, start_core)
        if j < n and toks[j].tag in ("DET", "POSS"):
            j += 1
            start_core = j
        k = j
        last_head = -1
        while k < n:
            tg = toks[k].tag
            if tg in ("ADJ", "NOUN", "PROPN", "NUM", "ING"):
                if tg != "ADJ":
                    last_head = k
                k += 1
            elif tg == "CONJ" and k > j and toks[k - 1].tag == "ADJ" and k + 1 < n and toks[k + 1].tag in ("ADJ", "NOUN", "PROPN"):
                k += 1                                # "British and Prussian forces"
            else:
                break
        if last_head < 0:
            return None
        end = last_head + 1
        # a trailing run of ADJ after the last head is not part of the NP (predicate adjectives)
        np_ = self._mk(p, i, end, last_head, start_core)
        if allow_of and end + 1 < n and toks[end].low == "of" and toks[end].tag == "PREP":
            nxt = self._chunk(toks, end + 1, p, allow_of=True)
            if nxt and not (toks[end + 1].tag == "PRON"):
                np_ = self._mk(p, i, nxt.e, nxt.head, start_core)
                np_.head = last_head if toks[last_head].tag == "NUM" else nxt.head
        return np_

    def _mk(self, p: Parsed, s: int, e: int, head: int, core_start: int) -> NP:
        toks = p.toks
        txt = p.span(s, e)
        core = p.span(core_start, e)
        plural = toks[head].low.endswith("s") and toks[head].tag == "NOUN" and not toks[head].low.endswith(("ss", "us", "is", "ics"))
        etype = "PRONOUN" if toks[head].tag == "PRON" else "CONCEPT"
        np_ = NP(s, e, head, etype, txt, core, plural)
        np_.cs = core_start
        return np_

    def _all_nps(self, toks: list, p: Parsed) -> list:
        out, i, n = [], 0, len(toks)
        while i < n:
            if toks[i].tag in ("DET", "POSS", "ADJ", "NOUN", "PROPN", "NUM", "ING", "PRON") or \
               (toks[i].tag == "ADV" and toks[i].low in QUANT_ADV):
                np_ = self._chunk(toks, i, p)
                if np_:
                    out.append(np_)
                    i = np_.e
                    continue
            i += 1
        return out

    # ----------------------------- clause parse ----------------------------- #
    def _parse(self, p: Parsed) -> Optional[Frame]:
        toks, n = p.toks, len(p.toks)
        if n < 4 or p.text.rstrip().endswith("?"):
            return None
        end = n - 1 if toks[-1].tag == "PUNCT" else n
        f = self._parse_clause(p, 0, end, None)
        p.frames = [f] if f else []
        # coordinated predicates sharing the subject: "It used X and weighed Y"
        while f and f.segs and f.segs[-1].kind == "coord":
            st = f.segs[-1].s + 1
            if st < end and toks[st].tag in ("BE", "HAVE", "DO", "MODAL", "VERB", "ADV"):
                f2 = self._parse_clause(p, st, end, f.subj)
                if not f2:
                    break
                f2.lead, f2.lead_np = f.lead, f.lead_np
                p.frames.append(f2)
                f = f2
            else:
                break
        return p.frames[0] if p.frames else None

    def _parse_clause(self, p: Parsed, i: int, end: int, subj_override) -> Optional[Frame]:
        toks, n = p.toks, len(p.toks)
        # strip leading discourse marker ("However,") and fronted adverbial ("In 1936,")
        lead, lead_np = None, None
        if subj_override is not None:
            pass
        elif toks[0].tag == "ADV" and n > 2 and toks[1].tag == "COMMA":
            i = 2
        elif toks[0].tag in ("PREP", "SUB"):
            comma = next((k for k in range(1, end) if toks[k].tag == "COMMA"), None)
            if comma is not None and not any(toks[k].tag in ("VERB", "BE", "HAVE", "DO", "MODAL") for k in range(1, comma)):
                lead = (0, comma)
                lead_np = self._chunk(toks, 1, p) if toks[0].tag == "PREP" else None
                i = comma + 1
        subj = self._chunk(toks, i, p) if subj_override is None else subj_override
        if not subj:
            return None
        j = subj.e if subj_override is None else i
        if subj_override is None:
            # subject post-modifiers: PPs ("the number of transistors on a chip")
            while j < end and toks[j].tag == "PREP" and toks[j].low != "by":
                nxt = self._chunk(toks, j + 1, p)
                if not nxt:
                    break
                ct = 1 if toks[subj.s].tag in ("DET", "POSS") else 0
                cs0 = subj.cs
                subj = NP(subj.s, nxt.e, subj.head, subj.etype, p.span(subj.s, nxt.e),
                          p.span(subj.s + ct, nxt.e), subj.plural)
                subj.cs = cs0
                j = nxt.e
            # skip a comma-delimited appositive: "Alan Turing, a mathematician, introduced ..."
            if j < end and toks[j].tag == "COMMA":
                k2 = next((q for q in range(j + 1, end) if toks[q].tag == "COMMA"), None)
                if k2 is not None and not any(toks[q].tag in ("VERB", "BE", "HAVE", "DO", "MODAL") for q in range(j + 1, k2)):
                    j = k2 + 1
        vs = j
        while j < end and toks[j].tag in ("BE", "HAVE", "DO", "MODAL", "VERB", "ADV"):
            j += 1
        vg = (vs, j)
        group = toks[vs:j]
        verbs = [t for t in group if t.tag == "VERB"]
        if not group or not any(t.tag in ("BE", "HAVE", "DO", "MODAL", "VERB") for t in group):
            return None
        if group[0].tag == "ADV" and not any(t.tag != "ADV" for t in group[:1]):
            pass
        auxes = [t for t in group if t.tag in ("BE", "HAVE", "DO", "MODAL")]
        neg = any(t.low == "not" for t in group)
        if len(verbs) > 1 and verbs[1].vform == "base" and verbs[0].lemma in ("help", "let", "make", "see", "watch", "hear"):
            j = verbs[0].i + 1
            vg = (vs, j)
            group = toks[vs:j]
            verbs = [verbs[0]]
        if verbs:
            verb = verbs[-1]
            copula = False
            passive = any(t.tag == "BE" for t in group) and verb.vform == "past" and not any(t.tag == "HAVE" and group.index(t) > max(group.index(b) for b in group if b.tag == "BE") for t in group)
        else:
            verb = [t for t in group if t.tag in ("BE", "HAVE", "DO", "MODAL")][-1]
            verb.lemma = verb.lemma or ("be" if verb.tag == "BE" else verb.low)
            copula = verb.tag == "BE"
            passive = False
        # rest of clause
        segs, clause_end = [], end
        k = j
        appos = naming = None
        while k < end:
            t = toks[k]
            if t.tag == "COMMA":
                clause_end = k
                tail_s = k + 1
                ap = self._chunk(toks, tail_s, p) if tail_s < end else None
                if ap and toks[tail_s].tag in ("DET", "POSS", "NUM", "PROPN"):
                    appos = ap
                for m in range(tail_s, end):
                    if toks[m].low in ("known", "called", "termed", "named", "referred") and m + 1 < end:
                        q = m + 1
                        if toks[q].low == "to" and q + 1 < end:
                            q += 1
                        if toks[q].low == "as" and q + 1 < end:
                            naming = self._chunk(toks, q + 1, p)
                            break
                segs.append(Seg("tail", tail_s, end))
                break
            if t.tag == "PREP":
                np_ = self._chunk(toks, k + 1, p)
                if np_:
                    sg = Seg("pp", k, np_.e, t.low, np_, [np_])
                    k = np_.e
                    if t.low == "by":  # coordinated agents: "A, B and C"
                        while k < end and toks[k].tag in ("COMMA", "CONJ"):
                            nx = self._chunk(toks, k + 1, p)
                            if nx and nx.etype != "PRONOUN" and toks[nx.head].tag == "PROPN":
                                sg.nps.append(nx)
                                sg.e = nx.e
                                k = nx.e
                            else:
                                break
                    segs.append(sg)
                    continue
                segs.append(Seg("pp", k, k + 1, t.low, None, []))
                k += 1
                continue
            if t.tag in ("DET", "POSS", "NUM", "NOUN", "PROPN", "ING", "PRON") or (t.tag == "ADV" and t.low in QUANT_ADV):
                np_ = self._chunk(toks, k, p)
                if np_:
                    segs.append(Seg("obj", k, np_.e, "", np_, [np_]))
                    k = np_.e
                    continue
            if t.tag == "ADJ":
                np_ = self._chunk(toks, k, p)
                if np_:
                    segs.append(Seg("obj", k, np_.e, "", np_, [np_]))
                    k = np_.e
                    continue
                m = k
                while m < end and toks[m].tag in ("ADJ", "ADV", "CONJ", "COMMA"):
                    m += 1
                segs.append(Seg("adj", k, m))
                k = m
                continue
            if t.tag in ("THAT", "SUB", "REL"):
                cut = end
                for q in range(k, end):
                    if toks[q].tag == "COMMA":
                        # trailing appositive/naming tail starts with a determiner NP
                        if q + 1 < end and toks[q + 1].tag in ("DET", "POSS"):
                            cut = q
                            break
                segs.append(Seg("that" if t.tag == "THAT" else "clause", k, cut))
                if cut < end:
                    clause_end = cut
                    ap = self._chunk(toks, cut + 2, p)
                    if ap and toks[cut + 1].tag in ("DET", "POSS"):
                        appos = self._chunk(toks, cut + 1, p)
                    for m in range(cut + 1, end):
                        if toks[m].low in ("known", "called", "termed", "named") and m + 1 < end:
                            q = m + 1
                            if toks[q].low == "as" and q + 1 < end:
                                naming = self._chunk(toks, q + 1, p)
                                break
                    segs.append(Seg("tail", cut + 1, end))
                k = end
                break
            if t.tag == "TO":
                cut = end
                for q in range(k, end):
                    if toks[q].tag == "COMMA":
                        cut = q
                        break
                segs.append(Seg("inf", k, cut))
                k = cut
                continue
            if t.tag == "CONJ":
                segs.append(Seg("coord", k, end))
                clause_end = k
                break
            k += 1
        return Frame(lead, lead_np, subj, vg, verb, passive, copula, neg, segs, clause_end, appos, naming)

    # ----------------------------- entities ----------------------------- #
    def _infer_persons(self):
        names = set()
        for p in self.parsed:
            for np_ in p.nps:
                toks = p.toks[np_.s:np_.e]
                pr = [t for t in toks if t.tag == "PROPN"]
                if len(pr) >= 2 and len(pr) == len([t for t in toks if t.tag not in ("DET", "POSS")]) and \
                        not ({t.low for t in pr} & (ORG_WORDS | PLACE_WORDS)):
                    if pr[0].low in FIRSTNAMES or (np_.s > 0 and p.toks[np_.s - 1].low.rstrip(".") in PERSON_TITLES):
                        names.add(" ".join(t.text for t in pr))
            for f in p.frames:
              if f.verb.tag == "VERB":
                lem = f.verb.lemma
                cands = []
                if not f.passive and lem in PERSON_VERBS:
                    cands.append(f.subj)
                if f.passive:
                    for sg in f.segs:
                        if sg.kind == "pp" and sg.prep == "by" and lem in PERSON_VERBS:
                            cands += sg.nps
                for np_ in cands:
                    toks = p.toks[np_.s:np_.e]
                    core = [t for t in toks if t.tag not in ("DET", "POSS")]
                    if 2 <= len(core) <= 3 and all(t.tag == "PROPN" for t in core) and \
                            not ({t.low for t in core} & (ORG_WORDS | PLACE_WORDS)):
                        names.add(" ".join(t.text for t in core))
        for nm in names:
            self.persons[nm.lower()] = nm
            self.surnames[nm.split()[-1].lower()] = nm
            self.surnames.setdefault(nm.split()[0].lower(), nm)

    def _type(self, np_: NP, p: Parsed):
        toks = p.toks[np_.s:np_.e]
        core = [t for t in toks if t.tag not in ("DET", "POSS") and not (t.tag == "ADV")]
        if np_.etype == "PRONOUN":
            return
        if len(core) == 1 and core[0].tag == "NUM":
            np_.etype = "YEAR" if core[0].year else "QUANTITY"
            return
        if core and core[0].tag == "NUM" and any(t.low == "of" for t in core):
            np_.etype = "CONCEPT"
            return
        if any(t.tag == "NUM" for t in core) and core[0].tag == "NUM":
            np_.etype = "QUANTITY"
            return
        head = p.toks[np_.head]
        text = " ".join(t.text for t in core)
        low = {t.low for t in core}
        if core and all(t.tag == "PROPN" for t in core) or (head.tag == "PROPN"):
            first = next((t for t in core if t.tag == "PROPN"), None)
            if text.lower() in self.persons or (len(core) == 1 and core[0].low in self.surnames):
                np_.etype = "PERSON"
            elif low & PLACE_WORDS:
                np_.etype = "PLACE"
            elif low & ORG_WORDS or (len(core) == 1 and core[0].text.isupper() and len(core[0].text) > 2):
                np_.etype = "ORG" if low & ORG_WORDS else "NAMED"
            elif first and all(t.tag == "PROPN" for t in core):
                np_.etype = "NAMED"
            else:
                np_.etype = "NAMED"
            if np_.etype != "PERSON" and head.tag == "PROPN" and any(t.tag != "PROPN" for t in core):
                np_.etype = "NAMED"
            return
        if any(t.tag == "PROPN" for t in core):
            np_.etype = "NAMED"       # e.g. "Turing machine", "Moore's Law"
            return
        np_.etype = "CONCEPT"

    # ----------------------------- coreference ----------------------------- #
    def _mid(self, p: Parsed, np_: NP, text: str) -> str:
        """Mid-sentence form: lower-case the first letter unless the phrase starts with a proper name."""
        t0 = p.toks[np_.s]
        if not text:
            return text
        if t0.tag in ("PROPN",) or np_.etype == "PERSON" or (len(text) > 1 and text[:2].isupper()):
            return text
        return text[0].lower() + text[1:]

    def _coref(self):
        last_person, last_thing = None, None          # (resolved, mid)
        for p in self.parsed:
            f = p.frame
            if not f:
                continue
            for fr in p.frames[1:]:
                fr.subj = f.subj
            subj = f.subj
            core = p.toks[subj.s:subj.e]
            first = p.toks[subj.s]
            if subj.etype == "PERSON" and len(core) == 1 and core[0].low in self.surnames:
                subj.resolved = self.surnames[core[0].low]
                subj.mid = subj.resolved
            elif subj.etype == "PRONOUN":
                w = p.toks[subj.head].low
                if w in ("he", "she") and last_person:
                    subj.resolved, subj.mid = last_person, last_person
                elif w in ("it", "they") and last_thing:
                    subj.resolved, subj.mid = last_thing
            elif first.tag == "POSS":
                w = first.low
                rest = p.span(subj.s + 1, subj.e)
                if w in ("his", "her") and last_person:
                    subj.resolved = f"{last_person}'s {rest}"
                    subj.mid = subj.resolved
                elif w in ("its", "their") and last_thing:
                    subj.resolved = f"{last_thing[1]}'s {rest}"
                    subj.mid = subj.resolved
            if not subj.resolved:
                subj.resolved = subj.text
                subj.mid = self._mid(p, subj, subj.text)
            if subj.etype == "PERSON":
                last_person = subj.resolved
            elif subj.etype in ("NAMED", "ORG", "PLACE", "CONCEPT"):
                last_thing = (subj.resolved, subj.mid)
            for fr in f and p.frames:
                for sg in fr.segs:
                    if sg.kind == "pp" and sg.prep == "by" and sg.np and sg.np.etype == "PERSON":
                        last_person = sg.np.text
        for p in self.parsed:
            for fr in p.frames:
                first = p.toks[fr.subj.s]
                head = p.toks[fr.subj.head].low
                fr.anaphoric = first.low in ("this", "these", "those", "such") or \
                    (first.low == "the" and head in ANAPHORIC and fr.subj.resolved == fr.subj.text)
        # make every NP carry a mid-sentence form
        for p in self.parsed:
            for fr in p.frames:
                for np_ in [fr.lead_np, fr.appos, fr.naming] + [x for sg in fr.segs for x in sg.nps]:
                    if np_ and not np_.mid:
                        np_.mid = self._mid(p, np_, np_.text)


def toks_have_det(p: Parsed, np_: NP) -> bool:
    return p.toks[np_.s].tag == "DET"
