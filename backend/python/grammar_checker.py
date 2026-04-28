"""
grammar_checker.py
------------------
Wraps LanguageTool (via language_tool_python) to check grammar in a given text.
Used by main.py via check_grammar().
"""

import language_tool_python
from dataclasses import dataclass, asdict

# ── Load LanguageTool once ─────────────────────────────────────────────────────
print("[Grammar] Loading LanguageTool …")
_tool = language_tool_python.LanguageTool("en-US")
print("[Grammar] LanguageTool ready.")

# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class GrammarIssue:
    message:     str       # human-readable description
    context:     str       # the sentence / phrase containing the error
    bad_word:    str       # the flagged text
    offset:      int       # character offset in the original text
    length:      int       # length of the flagged span
    replacements: list     # suggested corrections (may be empty)
    rule_id:     str       # LanguageTool rule identifier
    category:    str       # e.g. "GRAMMAR", "TYPOS", "STYLE"

# ── Public API ─────────────────────────────────────────────────────────────────

def check_grammar(text: str) -> list[dict]:
    """
    Run LanguageTool on `text` and return a list of GrammarIssue dicts.
    Each dict has the fields defined in GrammarIssue above.
    """
    if not text or not text.strip():
        return []

    try:
        matches = _tool.check(text)
        issues  = []

        for m in matches:
            # Skip pure spelling issues (handled by spell checker separately)
            if m.ruleId.startswith("MORFOLOGIK_RULE"):
                continue

            issue = GrammarIssue(
                message      = m.message,
                context      = m.context,
                bad_word     = text[m.offset : m.offset + m.errorLength],
                offset       = m.offset,
                length       = m.errorLength,
                replacements = m.replacements[:3],   # cap at 3
                rule_id      = m.ruleId,
                category     = m.category,
            )
            issues.append(asdict(issue))

        return issues

    except Exception as e:
        print(f"[Grammar] Error: {e}")
        return []
