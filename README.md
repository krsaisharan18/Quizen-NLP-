# Automatic Quiz Generation from Text (classical NLP only)

No LLMs, no neural text generation, no model downloads. Dependencies: `numpy`, `scikit-learn`, `flask` (UI), `pypdf` (PDF upload).

## How it understands a sentence (`nlp.py`)
1. **Tokenise + POS-tag** – lexicons, suffix rules and context rules (e.g. *"designs"* is a verb after a noun phrase but a noun after *"the"*; *"to"* is an infinitive marker only before a verb).
2. **Chunk noun phrases** – determiners, adjectives, proper-name runs, *"X of Y"*, coordinated adjectives (*"British and Prussian forces"*).
3. **Parse each clause** – fronted adverbial (*"In 1936,"*), subject, verb group, **voice** (active/passive), objects, prepositional phrases (time / place / agent), appositive, *"known as"* names, and coordinated predicates (*"used X and weighed Y"*).
4. **Type entities** – PERSON, ORG, PLACE, NAMED thing, YEAR, QUANTITY, CONCEPT.
5. **Coreference** – *"It used…"* -> *"The ENIAC used…"*, *"Her notes"* -> *"Ada Lovelace's notes"*, *"Turing"* -> *"Alan Turing"*.

## How it makes questions (`quizgen.py`)
Questions come from the **meaning frame** of each sentence, not from random keyword blanking:

| Sentence pattern | Question |
|---|---|
| `X <verb> Y in 1837` | *In which year did X <verb> Y?* |
| `X was <verb>-ed at P by A` | *Who <verb>-ed X?* / *Where was X <verb>-ed?* |
| `V-ed by guillotine` (means, not agent) | *How was X <verb>-ed?* |
| `X replaced by Y` | *What replaced X?* |
| `X used about 18,000 Ys` | *How many Ys did X use?* |
| `X is a Y that …` | *Which of the following is a Y that …?* |
| `…, an observation known as Z` | *What is the observation that … known as?* |
| any sentence | True/False (role-aware false swap) and fill-in-the-blank |

Ranking uses TF-IDF + TextRank. **Distractors** have the same entity type (person / place / term / year / number), come from the same text, are LSA-similar in context, never appear in the source sentence and never share words with each other. Questions that leak the answer, depend on vague references (*"this process"*), or contain unresolved pronouns are rejected.

## Run
    pip install -r requirements.txt
    python app.py                                   # web UI -> http://127.0.0.1:5000
    python quizgen.py sample.txt -n 10              # command line
    python quizgen.py sample.txt --format json -i   # JSON / interactive

## Limitations
- The tagger/parser are rule-based and tuned for declarative, encyclopedic prose (10+ factual sentences). Unusual grammar can be mis-parsed; such sentences are skipped rather than guessed.
- spaCy/NLTK were not installable in the build environment. With them, `nlp.py`'s tagger/chunker could be swapped for `doc.noun_chunks` / `doc.ents` and the same question frames would still apply.
