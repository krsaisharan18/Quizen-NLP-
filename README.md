# Automatic Quiz Generation from Text

Generate multiple-choice, true/false, short-answer and fill-in-the-blank quizzes from any passage of text, using **classical Natural Language Processing only**. There is no LLM, neural network, pretrained model or external API; everything runs locally.

---

## Table of contents
1. [Features](#features)
2. [Quick start](#quick-start)
3. [Usage](#usage)
4. [How it works](#how-it-works)
5. [Question types](#question-types)
6. [NLP techniques used](#nlp-techniques-used)
7. [Project structure](#project-structure)
8. [Example](#example)
9. [Limitations](#limitations)
10. [Troubleshooting](#troubleshooting)
11. [Future work](#future-work)

---

## Features
- **Paste text or upload a file**: `.pdf`, `.docx`, `.txt`, `.md` (up to 5 MB).
- **Four question types**: multiple choice, true/false, short answer, fill in the blank. You choose which to include and how many (1-30).
- **Meaning-based questions**: asks *who / when / where / how many / what replaced / what is it called / which term fits this definition*, based on what each sentence actually says.
- **Context-aware**: resolves pronouns and surnames (*"It used…"* becomes *"The ENIAC used…"*, *"Turing"* becomes *"Alan Turing"*).
- **Smart distractors**: wrong options have the same type as the answer (people for "who", years for "when"), come from the same text, and never appear in the source sentence.
- **Interactive quiz UI**: progress bar, answer checking, score ring, reveal answers, retry, new questions, download as Markdown.
- **Short-answer grading** accepts alternatives and small typos (for example "Berners-Lee" for "Tim Berners-Lee").
- **Command-line mode** with Markdown, JSON or an interactive terminal quiz.
- Light and dark mode, works on phone screens.

## Quick start
Requires **Python 3.9+**.

```bash
# 1. unzip and enter the folder
cd quizgen

# 2. (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. start the web app
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

Dependencies: `flask`, `scikit-learn`, `numpy`, `pypdf` (PDF upload only).

## Usage

### Web UI
1. Choose **Paste text** or **Upload file** (or click **Try sample text**).
2. Set the number of questions and the question types.
3. Click **Generate quiz**, answer the questions, then **Check answers**.
4. Use **Show answers**, **Retry**, **New questions** (different selection from the same text) or **Download .md**.

### Command line
```bash
python quizgen.py sample.txt -n 10                    # quiz + answer key (Markdown)
python quizgen.py sample.txt -n 8 --format json       # JSON output
python quizgen.py sample.txt --types mcq,wh -i        # take the quiz in the terminal
python quizgen.py sample.txt -o quiz.md --seed 7      # save to file, different selection
cat notes.txt | python quizgen.py -n 5                # read from stdin
```
| Option | Meaning |
|---|---|
| `-n, --num` | number of questions (default 10) |
| `--types` | comma list of `mcq,tf,wh,cloze` |
| `--format` | `md` (default) or `json` |
| `--seed` | change to get a different selection |
| `-o, --out` | write to a file |
| `-i, --interactive` | answer in the terminal and get a score |

### As a library
```python
from quizgen import QuizGenerator

gen = QuizGenerator(open("sample.txt").read(), seed=42)
for q in gen.generate(n=8, types=("mcq", "wh", "tf")):
    print(q.qtype, q.question, "->", q.answer)
print(gen.key_concepts)
```

### HTTP API
`POST /api/generate` (multipart form): `text` or `file`, `n`, `types` (repeatable), optional `seed`.
Returns `{questions: [...], sentences: N, concepts: [...]}`.

## How it works

```
text / PDF / DOCX
      |
      v
 1. Preprocess ---------- clean PDF layout, split sentences, tokenise
      |
      v
 2. Linguistic analysis (nlp.py)
      POS tagging -> noun-phrase chunking -> clause parsing
      -> entity typing -> coreference
      |
      v
 3. Importance ranking --- TF-IDF + TextRank
      |
      v
 4. Question frames ------ meaning pattern -> question template
      |                    + distractors (same type, LSA-similar)
      v
 5. Selection ----------- score x diversity, no leaked answers
      |
      v
   quiz (Markdown / JSON / web UI)
```

**1. Preprocessing.** PDF text is repaired (hyphenated line breaks, hard-wrapped lines, page numbers). Sentences are split with abbreviation and initial awareness (`Dr.`, `J. K. Rowling`).

**2. Linguistic analysis** (`nlp.py`)
- *POS tagging* with closed-class lexicons, a verb lexicon with all inflections, suffix rules, and context rules (for example `designs` is a verb after a noun phrase but a noun after `the`; `to` is an infinitive marker only before a verb).
- *Chunking* of noun phrases: determiners, adjectives, proper-name runs, `X of Y`, coordinated adjectives (`British and Prussian forces`).
- *Clause parsing*: fronted adverbial (`In 1936,`), subject, verb group, **voice** (active or passive), objects, prepositional phrases (time, place, agent), appositive, `known as` names, and coordinated predicates (`used X and weighed Y`).
- *Entity typing*: PERSON, ORG, PLACE, NAMED thing, YEAR, QUANTITY, CONCEPT.
- *Coreference*: pronouns, possessives and bare surnames resolve to earlier mentions.

**3. Importance ranking.** TF-IDF weights words; TextRank (PageRank over a sentence-similarity graph) finds the sentences that matter most.

**4. Question frames.** The parsed sentence is matched to a meaning pattern and turned into a question (see below). Verb forms are converted correctly (`designed` becomes `did … design`; passive `was invented` becomes active `invented`).

**5. Selection.** Candidates are scored (frame weight x sentence importance), then picked in a mix of types with one question per answer and limited repetition per sentence. Questions are dropped if they leak the answer, depend on vague references (*"this process"*), or contain unresolved pronouns.

## Question types

| Sentence pattern | Question generated |
|---|---|
| `X designed Y in 1837` | *In which year did X design Y?* |
| `Y was invented at P by A, B and C` | *Who invented Y at P?* / *Where was Y invented?* |
| `crowd stormed the Bastille` (human subject) | *Who stormed the Bastille?* |
| `Y was executed by guillotine` (means, not agent) | *How was Y executed?* |
| `X were replaced by Y` | *What replaced X?* |
| `X used about 18,000 Ys` | *How many Ys did X use?* |
| `X weighed 30 tons` | *How much did X weigh?* |
| `X is a Y that does Z` | *Which of the following is a Y that does Z?* |
| `…, an observation known as Z` | *What is the observation that … known as?* |
| any sentence | **True/False**: true as written, false by swapping one entity for another of the same type (person, year, place, number, term) |
| any sentence | **Fill in the blank** on an important name, term, year or number |

## NLP techniques used

| Technique | Where |
|---|---|
| Sentence segmentation, tokenisation | `quizgen.split_sentences`, `nlp.TOKEN_RE` |
| Part-of-speech tagging (lexicon + suffix + context rules) | `nlp.Analyzer._tag` |
| Lemmatisation and verb inflection | `nlp._inflect`, `past_tense`, `third_person` |
| Noun-phrase chunking | `nlp.Analyzer._chunk` |
| Clause / shallow dependency parsing | `nlp.Analyzer._parse_clause` |
| Named-entity recognition | `nlp.Analyzer._type`, `_infer_persons` |
| Coreference resolution | `nlp.Analyzer._coref` |
| TF-IDF keyword weighting | `quizgen._build_models` |
| TextRank sentence importance | `quizgen._importance` |
| LSA (TruncatedSVD) similarity for distractors | `quizgen._pick` |
| Template-based natural-language generation | `quizgen._frame_cands` |

**Not NLP:** the Flask server, the HTML/JS interface, PDF/DOCX text extraction and answer shuffling.

## Project structure
```
quizgen/
├── app.py                 Flask server (upload handling, PDF/DOCX extraction, /api/generate)
├── quizgen.py             question generation: frames, distractors, selection, CLI
├── nlp.py                 tagger, chunker, clause parser, entity typing, coreference
├── templates/index.html   web interface (single page, no build step)
├── requirements.txt
├── sample.txt             computing history (example input)
├── sample_history.txt     French Revolution (example input)
├── sample_science.txt     photosynthesis (example input)
├── sample.pdf             PDF version of sample.txt
├── ui.png                 screenshot
├── README.md
└── HOW_TO_RUN.md          short run guide
```

## Example
Input:
> Gordon Moore observed that the number of transistors on a chip doubles roughly every two years, an observation known as Moore's Law. The World Wide Web was proposed by Tim Berners-Lee at CERN in 1989.

Output:
- *What is the observation that the number of transistors on a chip doubles roughly every two years known as?*  ->  **Moore's Law**
- *Who proposed the World Wide Web at CERN in 1989?*  ->  **Tim Berners-Lee**
- *Where was the World Wide Web proposed by Tim Berners-Lee in 1989?*  ->  **CERN**

## Limitations
- **Rule-based, not statistically trained.** It is less accurate than a trained parser (spaCy, Stanza) on messy, very long or unusual sentences. Sentences it cannot parse confidently are skipped instead of guessed.
- **Best on factual, declarative prose** such as encyclopedia, textbook or lecture-note text, ideally 10+ sentences. Opinion, dialogue, poetry and lists work poorly.
- **Hand-written word lists** (verbs, first names, organisation/place cue words) mean rare words may be mis-typed.
- **Fixed question wording**: questions come from templates, not free-form generation.
- **English only.** Scanned PDFs (images) are not supported because there is no OCR.
- Some questions (especially fill-in-the-blank) can still be weak; review before high-stakes use.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | run `pip install -r requirements.txt` |
| "PDF support needs `pip install pypdf`" | `pip install pypdf` |
| "No readable text found in this PDF" | the PDF is a scan; use a text-based PDF or paste the text |
| "Need at least 4 sentences" | provide a longer passage |
| "Couldn't build questions" | text is too short or not factual prose; try more declarative sentences |
| Port 5000 already in use | edit the `port=5000` line at the bottom of `app.py` |

## Future work
- Optional spaCy/NLTK back-end (`doc.noun_chunks`, `doc.ents`, WordNet) behind the same question frames.
- More frames: cause and effect, comparisons, processes and sequences.
- OCR for scanned PDFs; multilingual support.
- Difficulty levels and exporting to Moodle/Kahoot formats.
