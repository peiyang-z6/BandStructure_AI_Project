"""Flag ambiguous scientific OCR without silently rewriting the source token."""

import re
import unicodedata
from collections import Counter


def review_document_text(text):
    """Flag broken PDF glyph mappings without rewriting mathematics or units."""
    suspicious = Counter(
        f"U+{ord(c):04X}"
        for c in text
        if (ord(c) < 32 and c not in "\n\r\t") or c == "\ufffd" or unicodedata.category(c) == "Co"
    )
    return {
        "status": "requires_visual_confirmation"
        if suspicious
        else "no_lexical_flags_not_certified",
        "suspicious_codepoints": dict(suspicious),
        "auto_corrected": False,
        "policy": "Nonempty extracted text is not verified scientific OCR. Compare symbols, signs, units and equations with original page vision.",
    }


def review_axis_tokens(regions):
    flagged = []
    for i, region in enumerate(regions):
        raw = region.get("text", "")
        flags = []
        suggestions = []
        if raw.strip() in {"O", "o"}:
            flags.append("letter_O_or_zero")
            suggestions = ["0", "O"]
        if "\ufffd" in raw or any(ord(c) < 32 for c in raw):
            flags.append("unreadable_glyph")
        if re.search(r"[A-Z][I|][A-Z]", raw):
            flags.append("possible_path_discontinuity")
            suggestions.append(raw.replace("I", "|"))
        if raw.strip().isdigit():
            flags.append("sign_must_be_checked_against_axis")
        if region.get("ocr_score", 1) < 0.95:
            flags.append("low_perception_score")
        if flags:
            flagged.append(
                {
                    "region_index": i,
                    "raw_text": raw,
                    "box": region.get("box"),
                    "flags": flags,
                    "proposed_tokens": suggestions,
                    "auto_applied": False,
                }
            )
    return {
        "status": "requires_visual_confirmation" if flagged else "no_lexical_flags_not_certified",
        "flagged_regions": flagged,
        "corrected_text": None,
        "policy": "Keep raw OCR; confirm signs, Gamma and X|W from the original crop. Suggestions never calibrate physics.",
    }
