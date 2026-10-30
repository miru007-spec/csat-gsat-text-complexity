"""
ctci_pipeline.py
================
Reference implementation of the six-variable Composite Text Complexity
Index (CTCI) described in Section 3.2 of:

  Lee, H. (2026+). "Text complexity in high-stakes EFL examinations:
  a comparative Coh-Metrix study of Korea's CSAT and Taiwan's GSAT."
  Submitted to System (Elsevier).

This is a from-scratch reconstruction written to match the formulas and
procedures stated in the manuscript's Methods section. It was written
independently of (and may differ in minor implementation details from)
whatever script originally produced the numbers reported in the paper;
it is provided as a documented, runnable reference implementation for
reproducibility and methodological transparency, not as a byte-for-byte
replica of the original run.

INPUT
-----
A CSV (or a pandas DataFrame) with one row per passage and columns:
    doc_id   : unique passage identifier, e.g. "CSAT_2024_21"
    exam     : "CSAT" or "GSAT" (or any group label)
    year     : administration year / academic year label
    text     : the full passage text (plain string)

The original CSAT and GSAT passage texts are NOT included in this
repository because they are copyrighted by the Korea Institute for
Curriculum and Evaluation (KICE) and Taiwan's College Entrance
Examination Center (CEEC) respectively. Qualifying researchers may
request the source materials directly from those institutions; this
pipeline can then be run on the resulting corpus.

OUTPUT
------
A DataFrame/CSV with one row per passage, the six raw variable values,
and the normalized CTCI score.

DEPENDENCIES
------------
    pip install pandas numpy

No third-party NLP library is required for this core module (regex-based
tokenization and a rule-based syllable counter are used, exactly as
described in Section 3.2, "in place of a third-party readability
library, so as to preserve exact continuity with the prior study's
corpus statistics"). See supplementary_analysis.py for the spaCy-based
supplementary analyses from Section 3.5.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Tokenization and syllable counting
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def tokenize_words(text: str) -> list[str]:
    """Regex-based word tokenization (alphabetic tokens, contractions kept)."""
    return _WORD_RE.findall(text)


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter.

    This is not a full sentence boundary disambiguation model; for a
    corpus with unusual abbreviation patterns, review the split output
    before trusting downstream sentence-level statistics (e.g. the
    sentence-level bootstrap in supplementary_analysis.py).
    """
    text = text.strip()
    if not text:
        return []
    raw = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


_VOWELS = "aeiouy"


def count_syllables(word: str) -> int:
    """Rule-based syllable counter (vowel-group heuristic with common
    English orthographic adjustments). Matches the approach described
    in Section 3.2: "a rule-based syllable counter ... in place of a
    third-party readability library."
    """
    w = word.lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0

    # Count vowel groups as syllable nuclei.
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups)

    # Silent trailing 'e' (but not 'le' preceded by a consonant, e.g. "table").
    if w.endswith("e") and not w.endswith("le"):
        count -= 1
    elif w.endswith("le") and len(w) > 2 and w[-3] not in _VOWELS:
        count += 0  # already counted as a group typically; no adjustment

    # Words ending in silent 'ed' after a non-t/d consonant don't add a syllable
    # (e.g. "walked" = 1 syllable, "wanted" = 2). The vowel-group count above
    # already handles most cases; this guards the common over-count.
    if w.endswith("ed") and len(w) > 2 and w[-3] not in _VOWELS and w[-3] not in "td":
        count -= 1

    return max(count, 1)


def flesch_kincaid_grade(text: str) -> float:
    """FK = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    (Kincaid, Fishburne, Rogers, & Chissom, 1975), as specified in Section 3.2.
    """
    words = tokenize_words(text)
    sentences = split_sentences(text)
    if not words or not sentences:
        return float("nan")
    syllables = sum(count_syllables(w) for w in words)
    return 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59


# ---------------------------------------------------------------------------
# 2. Moving-Average Type-Token Ratio (MATTR)
# ---------------------------------------------------------------------------

def mattr(text: str, window: int = 49) -> float:
    """Moving-Average Type-Token Ratio over a sliding window of `window`
    tokens (Covington & McFall, 2010). If the passage is shorter than the
    window, the single whole-passage TTR is returned.
    """
    words = [w.lower() for w in tokenize_words(text)]
    n = len(words)
    if n == 0:
        return float("nan")
    if n <= window:
        return len(set(words)) / n

    ratios = []
    for i in range(0, n - window + 1):
        seg = words[i : i + window]
        ratios.append(len(set(seg)) / window)
    return float(np.mean(ratios))


# ---------------------------------------------------------------------------
# 3. Connective density
# ---------------------------------------------------------------------------

# Coh-Metrix-style connective taxonomy (Graesser et al., 2004): causal,
# contrastive, additive, and temporal connectives. Multi-word connectives
# are matched as phrases so they are not undercounted (e.g. "for this
# reason", "in contrast", "as a result").
CONNECTIVES = {
    "causal": [
        "because", "since", "so that", "therefore", "thus", "hence",
        "as a result", "consequently", "for this reason", "due to",
        "owing to", "so",
    ],
    "contrastive": [
        "however", "although", "though", "but", "yet", "nevertheless",
        "nonetheless", "in contrast", "on the other hand", "whereas",
        "while", "despite", "in spite of", "conversely",
    ],
    "additive": [
        "and", "also", "moreover", "furthermore", "in addition",
        "additionally", "besides", "as well as",
    ],
    "temporal": [
        "then", "after", "before", "when", "while", "until", "once",
        "subsequently", "meanwhile", "afterward", "afterwards",
    ],
}

_ALL_CONNECTIVES = sorted(
    {c for group in CONNECTIVES.values() for c in group}, key=len, reverse=True
)
# Build one regex alternation, longest phrases first so multi-word
# connectives are matched before their single-word substrings.
_CONNECTIVE_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _ALL_CONNECTIVES) + r")\b",
    re.IGNORECASE,
)


def connective_density(text: str) -> float:
    """Proportion of tokens realizing a causal, contrastive, additive, or
    temporal connective (Section 3.2). Phrase matches count as one
    connective occurrence (not one per word in the phrase) but are
    normalized against the total token count, consistent with treating
    connective density as "proportion of tokens realizing ... connectives".
    """
    words = tokenize_words(text)
    if not words:
        return float("nan")
    matches = _CONNECTIVE_RE.findall(text)
    return len(matches) / len(words)


# ---------------------------------------------------------------------------
# 4. Imageability / concreteness
# ---------------------------------------------------------------------------

def load_mrc_lexicon(path: str) -> dict[str, float]:
    """Load an MRC Psycholinguistic Database-derived concreteness lexicon.

    Expected CSV format: two columns, `word` and `concreteness`
    (100-700 scale; Coltheart, 1981; Paivio, Yuille, & Madigan, 1968).

    The MRC Psycholinguistic Database itself is not redistributed here;
    obtain a copy (or an equivalent curated concreteness norms file, e.g.
    Brysbaert et al.'s extended concreteness norms rescaled to 100-700)
    and point this function at it.
    """
    df = pd.read_csv(path)
    df["word"] = df["word"].str.lower()
    return dict(zip(df["word"], df["concreteness"]))


def imageability(text: str, lexicon: dict[str, float]) -> float:
    """Mean concreteness/imageability rating of content words matched
    against the supplied lexicon (Section 3.2). Words not found in the
    lexicon are excluded from the mean (not imputed), matching the
    manuscript's description of a "curated subset" with limited coverage
    (see Section 5.4, point three).
    """
    words = [w.lower() for w in tokenize_words(text)]
    vals = [lexicon[w] for w in words if w in lexicon]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# 5. Nominalization index
# ---------------------------------------------------------------------------

_NOMINAL_SUFFIXES = ("tion", "ness", "ment", "ity", "ance", "ence")


def nominalization_index(text: str) -> float:
    """Proportion of tokens bearing a nominalizing suffix (-tion, -ness,
    -ment, -ity, -ance, -ence), following Biber & Gray's (2011) account
    of nominal style (Section 3.2).
    """
    words = tokenize_words(text)
    if not words:
        return float("nan")
    hits = sum(1 for w in words if w.lower().endswith(_NOMINAL_SUFFIXES))
    return hits / len(words)


# ---------------------------------------------------------------------------
# 6. Discourse coherence (Jaccard content-word overlap)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "as", "by", "with", "from", "into",
    "than", "then", "so", "if", "not", "no", "do", "does", "did", "has",
    "have", "had", "will", "would", "can", "could", "may", "might", "shall",
    "should", "must", "he", "she", "they", "we", "you", "i", "his", "her",
    "their", "our", "your", "my", "which", "who", "whom", "what", "when",
    "where", "why", "how", "there", "here", "also",
}


def discourse_coherence(text: str) -> float:
    """Mean Jaccard similarity of content-word overlap between adjacent
    sentences (Section 3.2), a computationally tractable approximation of
    Coh-Metrix's referential-cohesion indices (cf. Lapata & Barzilay, 2005).
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return float("nan")

    content_sets = []
    for s in sentences:
        words = {w.lower() for w in tokenize_words(s)}
        content_sets.append(words - _STOPWORDS)

    sims = []
    for a, b in zip(content_sets, content_sets[1:]):
        if not a and not b:
            continue
        union = a | b
        if not union:
            continue
        sims.append(len(a & b) / len(union))

    if not sims:
        return float("nan")
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# 7. CTCI assembly
# ---------------------------------------------------------------------------

@dataclass
class CTCIResult:
    doc_id: str
    fk_grade: float
    mattr: float
    connective_density: float
    imageability: float
    nominalization_index: float
    discourse_coherence: float
    ctci: float = field(default=float("nan"))


def compute_raw_variables(
    df: pd.DataFrame, mrc_lexicon: dict[str, float], text_col: str = "text", id_col: str = "doc_id"
) -> pd.DataFrame:
    """Compute the six raw (un-normalized) variables for every passage in `df`."""
    rows = []
    for _, r in df.iterrows():
        text = r[text_col]
        rows.append(
            {
                id_col: r[id_col],
                "fk_grade": flesch_kincaid_grade(text),
                "mattr": mattr(text, window=49),
                "connective_density": connective_density(text),
                "imageability": imageability(text, mrc_lexicon),
                "nominalization_index": nominalization_index(text),
                "discourse_coherence": discourse_coherence(text),
            }
        )
    return pd.DataFrame(rows)


# Direction correction: for these four indices, LOWER raw values indicate
# GREATER difficulty, so they are reverse-coded before averaging (Section 3.2).
_REVERSE_CODE = {"connective_density", "imageability", "mattr", "discourse_coherence"}
_VARIABLES = [
    "fk_grade",
    "mattr",
    "connective_density",
    "imageability",
    "nominalization_index",
    "discourse_coherence",
]


def compute_ctci(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize each of the six variables across the full set of
    passages passed in (`raw_df`), direction-correct, and average with
    equal weighting to produce CTCI (range 0-1), following Section 3.2.

    IMPORTANT: CTCI is normalized WITHIN whatever set of passages is
    passed to this function. Passing a different passage set (e.g. a
    different-sized corpus, or CSAT-only vs. CSAT+GSAT pooled) will
    change the normalization reference and is not numerically comparable
    across runs — this is exactly the CTCI cross-validation caveat
    discussed in Section 4.5 and Section 5.4 of the manuscript. Always
    compute CTCI over the full pooled sample you intend to compare.
    """
    out = raw_df.copy()
    normalized_cols = []
    for var in _VARIABLES:
        col = raw_df[var]
        vmin, vmax = col.min(), col.max()
        if vmax == vmin:
            norm = pd.Series(0.5, index=col.index)  # degenerate case: no variance
        else:
            norm = (col - vmin) / (vmax - vmin)
        if var in _REVERSE_CODE:
            norm = 1 - norm
        norm_col = f"{var}_norm"
        out[norm_col] = norm
        normalized_cols.append(norm_col)

    out["ctci"] = out[normalized_cols].mean(axis=1)
    return out


def run_ctci_pipeline(
    df: pd.DataFrame, mrc_lexicon_path: str, text_col: str = "text", id_col: str = "doc_id"
) -> pd.DataFrame:
    """Convenience wrapper: load the lexicon, compute raw variables, compute CTCI.

    `df` must contain at least [id_col, text_col]; any additional columns
    (e.g. exam, year) are preserved by being left-joined back in.
    """
    lexicon = load_mrc_lexicon(mrc_lexicon_path)
    raw = compute_raw_variables(df, lexicon, text_col=text_col, id_col=id_col)
    with_ctci = compute_ctci(raw)
    meta_cols = [c for c in df.columns if c not in (text_col,)]
    return df[meta_cols].merge(with_ctci, on=id_col, how="left")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python ctci_pipeline.py <passages.csv> <mrc_lexicon.csv> <output.csv>")
        sys.exit(1)

    passages_path, lexicon_path, output_path = sys.argv[1:4]
    passages = pd.read_csv(passages_path)
    result = run_ctci_pipeline(passages, lexicon_path)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")
