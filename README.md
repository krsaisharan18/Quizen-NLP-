# Automatic Quiz Generation from Text (classical NLP only)

No LLMs or neural text generation. Dependencies: `numpy`, `scikit-learn`.

## Run
    python quizgen.py sample.txt -n 10                  # Markdown quiz + answer key
    python quizgen.py sample.txt -n 8 --format json     # machine-readable
    python quizgen.py sample.txt --types mcq,tf -i      # take the quiz in the terminal

## Pipeline
1. **Preprocessing** – whitespace normalisation, abbreviation-aware sentence splitting, regex tokenisation.
2. **Linguistic analysis** – stop-word + suffix/lexicon POS filtering, capitalisation-based proper-noun chunking
   (lightweight NER: person / place / organisation), regex YEAR and NUMBER detection, head-final noun-phrase extraction.
3. **Importance scoring** – TF-IDF weights for keywords, TextRank (PageRank over a cosine-similarity sentence graph)
   for sentence selection, phrase-recurrence boost.
4. **Distractors** – LSA (TruncatedSVD of TF-IDF) context vectors give same-type, topically close terms; options may
   not share words; years/numbers use numeric perturbation.
5. **Question builders** – cloze, multiple choice, True/False (answer substitution for the false items), and
   rule-based WH questions (Who/What/Where/When via subject–auxiliary templates).

## Limitations
- spaCy / NLTK could not be installed (package index blocked in this environment), so POS and NER are heuristic.
  With spaCy, swap `_extract` for `doc.noun_chunks` / `doc.ents` and use WordNet for distractors.
- Works best on encyclopedic, declarative prose (>= ~10 sentences). Some terms/distractors are imperfect; review output.

## Web UI
    pip install flask scikit-learn numpy      # 
    python app.py                             # open http://127.0.0.1:5000
Paste text or upload .txt/.md/.docx/.pdf, pick question count and types, then take the quiz in the browser (check, reveal, regenerate, download .md).
