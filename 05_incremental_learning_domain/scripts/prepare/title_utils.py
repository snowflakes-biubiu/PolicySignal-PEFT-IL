from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(title))).casefold()


def title_key(title: str) -> str:
    normalized = normalize_title(title)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()