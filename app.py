#!/usr/bin/env python3
"""Web UI for the NLP quiz generator.  Run:  python app.py  ->  http://127.0.0.1:5000"""
import io
import re
import zipfile

from flask import Flask, jsonify, render_template, request

from quizgen import QuizGenerator

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB


def clean_pdf_text(t: str) -> str:
    """Undo PDF layout artefacts: hyphenated line breaks, hard-wrapped lines, page numbers, headers."""
    t = t.replace("\r", "")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)                       # de-hyphenate
    lines = [l.strip() for l in t.split("\n")]
    lines = [l for l in lines if not re.fullmatch(r"(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?", l, re.I)]
    out, buf = [], []
    for l in lines:
        if not l:
            if buf: out.append(" ".join(buf)); buf = []
        else:
            buf.append(l)
    if buf: out.append(" ".join(buf))
    # headings without end punctuation would glue onto the next sentence: end them with a full stop
    out = [o if re.search(r"[.!?:\"')]$", o) or len(o.split()) > 12 else o + "." for o in out]
    return re.sub(r"[ \t]+", " ", "\n\n".join(out))


def read_upload(f) -> str:
    name = (f.filename or "").lower()
    data = f.read()
    if name.endswith(".docx"):  # a .docx is a zip; pull text out of word/document.xml (no extra libs)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
        xml = re.sub(r"</w:p>", "\n", xml)
        return re.sub(r"<[^>]+>", "", xml)
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF support needs `pip install pypdf`. Upload .txt/.md/.docx or paste the text.")
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("This PDF is password-protected.")
            pages = [(p.extract_text() or "") for p in reader.pages]
        except ValueError:
            raise
        except Exception:
            raise ValueError("Couldn't read this PDF. It may be corrupted.")
        text = clean_pdf_text("\n\n".join(pages))
        if len(text.split()) < 30:
            raise ValueError("No readable text found in this PDF (it may be a scanned image). Try a text-based PDF.")
        return text
    return data.decode("utf-8", "ignore")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/generate")
def generate():
    try:
        text = request.form.get("text", "").strip()
        f = request.files.get("file")
        if f and f.filename:
            text = read_upload(f).strip()
        if not text:
            return jsonify(error="Paste some text or upload a file first."), 400
        n = max(1, min(30, int(request.form.get("n", 10))))
        types = tuple(t for t in request.form.getlist("types") if t in {"mcq", "tf", "wh", "cloze"}) \
            or ("mcq", "tf", "wh", "cloze")
        seed = request.form.get("seed")
        gen = QuizGenerator(text, seed=int(seed) if seed else 42)
        qs = gen.generate(n, types)
        if not qs:
            return jsonify(error="Couldn't build questions from this text. Try longer, factual prose."), 422
        return jsonify(questions=[q.to_dict() for q in qs], sentences=len(gen.sentences))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except zipfile.BadZipFile:
        return jsonify(error="That .docx file looks corrupted."), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
