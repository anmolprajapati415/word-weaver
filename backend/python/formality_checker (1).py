"""
formality_checker.py
---------------------
Uses a pretrained text-classification model to detect whether text is
FORMAL or INFORMAL.

Model used: s-nlp/roberta-base-formality-ranker
  - A RoBERTa model fine-tuned on the GYAFC formality corpus.
  - Output label 0 → INFORMAL, label 1 → FORMAL
  - Returns a confidence score 0-100.

If the model cannot be downloaded (offline), falls back to a lightweight
heuristic analyser so the app still works.
"""

from __future__ import annotations
import re

# ── Try to load the pretrained model ──────────────────────────────────────────
_pipeline = None

def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return True
    try:
        from transformers import pipeline as hf_pipeline
        print("[Formality] Loading roberta formality model …")
        _pipeline = hf_pipeline(
            "text-classification",
            model="s-nlp/roberta-base-formality-ranker",
            truncation=True,
            max_length=512,
        )
        print("[Formality] Model ready.")
        return True
    except Exception as e:
        print(f"[Formality] Could not load HuggingFace model ({e}). Using heuristic fallback.")
        return False

# ── Heuristic fallback ─────────────────────────────────────────────────────────
_INFORMAL_PATTERNS = [
    r"\b(gonna|wanna|gotta|kinda|sorta|dunno|ya|yep|nope|yeah|nah|lol|omg|btw|idk|imo|tbh|fyi|asap|bc|cuz|cos|cya)\b",
    r"\b(ain't|can't|won't|don't|isn't|aren't|wasn't|weren't|haven't|hadn't|doesn't|didn't|couldn't|wouldn't|shouldn't)\b",
    r"[!]{2,}",           # multiple exclamation marks
    r"[?]{2,}",           # multiple question marks
    r"\.{3,}",            # excessive ellipsis
    r"\b[A-Z]{4,}\b",     # ALL CAPS words (shouting)
]
_INFORMAL_RE = re.compile("|".join(_INFORMAL_PATTERNS), re.IGNORECASE)

_FORMAL_MARKERS = [
    r"\b(furthermore|moreover|consequently|therefore|thus|hence|accordingly|nevertheless|nonetheless|notwithstanding)\b",
    r"\b(regarding|concerning|pertaining|pursuant|herein|thereof|whereby|whereas|albeit)\b",
    r"\b(demonstrate|indicate|suggest|propose|conclude|analyse|analyze|investigate|examine|assess|evaluate)\b",
    r"\b(shall|may|would|could|ought)\b",
]
_FORMAL_RE = re.compile("|".join(_FORMAL_MARKERS), re.IGNORECASE)


def _heuristic_check(text: str) -> dict:
    words         = text.split()
    total_words   = max(len(words), 1)
    informal_hits = len(_INFORMAL_RE.findall(text))
    formal_hits   = len(_FORMAL_RE.findall(text))

    # Avg sentence length (longer → more formal)
    sentences     = re.split(r"[.!?]+", text)
    sentences     = [s.strip() for s in sentences if s.strip()]
    avg_sent_len  = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

    # Score: higher = more formal
    score = 50
    score -= informal_hits * 8
    score += formal_hits   * 6
    score += min((avg_sent_len - 10) * 1.5, 20)   # longer sentences bump up
    score  = max(0, min(100, score))

    label      = "FORMAL" if score >= 50 else "INFORMAL"
    confidence = score if label == "FORMAL" else (100 - score)

    return {
        "label":      label,
        "confidence": round(confidence, 1),
        "score":      round(score, 1),
        "method":     "heuristic",
        "details": {
            "informal_signals": informal_hits,
            "formal_signals":   formal_hits,
            "avg_sentence_len": round(avg_sent_len, 1),
        },
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def check_formality(text: str) -> dict:
    """
    Analyse `text` and return a dict with keys:
        label      : "FORMAL" | "INFORMAL"
        confidence : float 0-100
        score      : raw formality score 0-100
        method     : "model" | "heuristic"
        details    : extra breakdown info
    """
    if not text or not text.strip():
        return {"label": "UNKNOWN", "confidence": 0, "score": 0,
                "method": "none", "details": {}}

    model_ok = _load_pipeline()

    if model_ok and _pipeline is not None:
        try:
            # Model accepts up to 512 tokens; pass first 1 000 chars
            result    = _pipeline(text[:1000])[0]
            raw_label = result["label"].upper()       # "LABEL_0" or "LABEL_1"
            raw_score = float(result["score"])        # 0-1 probability

            # s-nlp model: LABEL_0 = informal, LABEL_1 = formal
            if "0" in raw_label:
                label = "INFORMAL"
                conf  = round((1 - raw_score) * 100, 1) if raw_score > 0.5 else round(raw_score * 100, 1)
            else:
                label = "FORMAL"
                conf  = round(raw_score * 100, 1)

            # Also run heuristic for extra detail
            heuristic = _heuristic_check(text)

            return {
                "label":      label,
                "confidence": conf,
                "score":      round(raw_score * 100, 1),
                "method":     "model",
                "details":    heuristic["details"],
            }
        except Exception as e:
            print(f"[Formality] Model inference error: {e}. Falling back to heuristic.")

    # Fallback
    return _heuristic_check(text)
