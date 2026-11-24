"""Lightweight Arabic text normalization shared by all Masreyyat scripts.

Deliberately minimal and rule-based: this is a baseline, not a stemmer.
"""
import re

ARABIC_DIACRITICS = re.compile(r"[ً-ْٰ]")
TATWEEL = "ـ"

_ALEF_VARIANTS = "أإآٱ"
_YAA_VARIANTS = "ىئ"


def normalize(text: str) -> str:
    """Normalize Arabic text for bag-of-ngrams models.

    Steps: strip diacritics and tatweel, unify alef/yaa variants,
    normalize teh marbuta, collapse whitespace, lowercase Latin.
    """
    if not isinstance(text, str):
        return ""
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")
    for ch in _ALEF_VARIANTS:
        text = text.replace(ch, "ا")
    for ch in _YAA_VARIANTS:
        text = text.replace(ch, "ي")
    text = text.replace("ة", "ه")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
