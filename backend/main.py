"""
main.py  —  Word Weaver FastAPI Backend
========================================
Routes
------
POST /suggest          → GPT-2 next-word suggestions
POST /grammar          → LanguageTool grammar check
POST /formality        → Formal / Informal classification
POST /spellcheck/text  → Full-text spell check (returns all misspelled words)
POST /spellcheck/word  → Single-word spell check (real-time)
GET  /health           → Health ping
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback

# ── Import our model modules ───────────────────────────────────────────────────
from gpt2_suggester  import get_suggestions
from grammar_checker import check_grammar
from formality_checker import check_formality
from spell_checker   import check_word, check_text
from word_generator import GPT2

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Word Weaver API",
    description="Backend for the Word Weaver research paper editor.",
    version="1.0.0",
)

# Allow all origins so the HTML file can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ─────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str
    n:    Optional[int] = 5          # for suggestions endpoint

class WordRequest(BaseModel):
    word: str

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Word Weaver API"}


# ─── 1. GPT-2 Next-Word Suggestions ───────────────────────────────────────────
gpt2 = GPT2()

@app.post("/generate")
def generate(req: TextRequest):
    try:
        generated = gpt2.predict_next(req.text, req.n or 5)
        return {"generated_text": generated}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="GPT-2 generation error.")



# ─── 2. Grammar Check ─────────────────────────────────────────────────────────
@app.post("/grammar")
def grammar(req: TextRequest):
    """
    Run LanguageTool grammar check on the full editor text.
    Returns a list of grammar issues with suggestions.
    """
    try:
        issues = check_grammar(req.text)
        return {
            "issue_count": len(issues),
            "issues":      issues,
        }
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Grammar check error.")


# ─── 3. Formality Check ───────────────────────────────────────────────────────
@app.post("/formality")
def formality(req: TextRequest):
    """
    Classify the editor text as FORMAL or INFORMAL.
    Returns label, confidence score, and breakdown details.
    """
    try:
        result = check_formality(req.text)
        return result
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Formality check error.")


# ─── 4a. Spell Check — full text ──────────────────────────────────────────────
@app.post("/spellcheck/text")
def spellcheck_text(req: TextRequest):
    """
    Check all words in the text and return misspelled ones with positions.
    Used when the user first toggles spell-check mode.
    """
    try:
        errors = check_text(req.text)
        return {
            "error_count": len(errors),
            "errors":      errors,
        }
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Spell check error.")


# ─── 4b. Spell Check — single word (real-time) ────────────────────────────────
@app.post("/spellcheck/word")
def spellcheck_word(req: WordRequest):
    """
    Check a single word typed by the user.
    Called on each word boundary (space / punctuation) in real-time.
    """
    try:
        result = check_word(req.word)
        return result
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Word spell check error.")
