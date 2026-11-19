"""One comparable key per legal document, whatever its spelling: law number, corpus file stem, vbpl / web file.

    law_key("50/2024/TT-NHNN") == stem_key("50_2024_TT-NHNN") == stem_key("50_2024_TT-NHNN__285123")
    law_key("230/QĐ-UBND") == stem_key("230_QĐ-UBND_1")
"""
import re
import unicodedata


def law_key(number: str) -> str:
    """NFC, no spaces, upper-case, '_' == '/', '/QĐ-TTg' == '/QĐ-TT'."""
    k = re.sub(r"\s+", "", unicodedata.normalize("NFC", number or "")).upper().replace("_", "/")
    return re.sub(r"-TTG$", "-TT", k)


def stem_key(stem: str) -> str:
    """Corpus file stem -> law_key. Drops "__<id>" (web / vbpl: same number, other years) and the course's
    "_1", "_2" copies of one number ("01-LĐ_TT_2"; numbers never end in "_<digits>" after a letter)."""
    stem = re.sub(r"__\d+$", "", stem or "")
    return law_key(re.sub(r"(?<=[^\d_])_\d+$", "", stem))
