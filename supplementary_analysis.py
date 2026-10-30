"""
supplementary_analysis.py
==========================
Reference implementation of the five supplementary analyses described in
Section 3.5 of the manuscript (see ctci_pipeline.py for the core
six-variable CTCI). As with ctci_pipeline.py, this is an independent
reconstruction from the Methods section description, provided for
reproducibility and transparency, not a byte-for-byte replica of
whatever script originally produced the paper's reported numbers.

Covers:
  1. Compound-noun density (spaCy dependency parsing)
  2. N-gram keyness (log-likelihood, Rayson & Garside, 2000)
  3. Syntactic complexity: mean dependency distance (MDD) and
     subordination ratio
  4. Lexical frequency profiling (LFP) using Zipf frequency bands
  5. Sentence-level nonparametric bootstrap for FK Grade

Asyndetic-coordination coding (semicolon classification) is NOT
automated here: the manuscript (Section 3.5, "Asyndetic coordination")
describes this as a hand-classification task, and Section 5.4 notes it
was performed by a single rater. `SYNDETIC_MARKERS` below is provided
only as a convenience checklist for that manual coding, not as an
automatic classifier.

DEPENDENCIES
------------
    pip install spacy wordfreq numpy pandas scipy
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    zipf_frequency = None


_NLP = None


def get_nlp():
    """Lazily load the spaCy pipeline (en_core_web_sm), matching
    Section 3.5: "spaCy 3.8 (en_core_web_sm) for dependency parsing."
    """
    global _NLP
    if _NLP is None:
        if spacy is None:
            raise ImportError("spaCy is required: pip install spacy && python -m spacy download en_core_web_sm")
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ---------------------------------------------------------------------------
# 1. Compound-noun density
# ---------------------------------------------------------------------------

def compound_noun_density(text: str) -> float:
    """Tokens attached to a head noun via the `compound` dependency
    relation (e.g. "attention economy", "traffic light"), expressed per
    100 tokens (Section 3.5).
    """
    doc = get_nlp()(text)
    n_tokens = sum(1 for t in doc if not t.is_space and not t.is_punct)
    if n_tokens == 0:
        return float("nan")
    n_compound = sum(1 for t in doc if t.dep_ == "compound")
    return (n_compound / n_tokens) * 100


# ---------------------------------------------------------------------------
# 2. N-gram keyness (log-likelihood)
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def log_likelihood_keyness(
    target_tokens: list[str], reference_tokens: list[str], n: int = 1
) -> pd.DataFrame:
    """Log-likelihood (G2) keyness statistic comparing n-gram frequencies
    in a target corpus against a reference corpus (Rayson & Garside, 2000),
    as used for the word-level and n-gram keyness analysis in Sections 3.5
    and 4.3. No multiple-comparison correction is applied here, matching
    the manuscript's explicit statement that this analysis is reported as
    preliminary and descriptive (Section 3.5, Section 5.4).

    Returns a DataFrame with columns: ngram, freq_target, freq_reference,
    LL, sign (+1 if over-represented in target, -1 if under-represented).
    """
    target_grams = _ngrams(target_tokens, n) if n > 1 else target_tokens
    ref_grams = _ngrams(reference_tokens, n) if n > 1 else reference_tokens

    target_counts = Counter(target_grams)
    ref_counts = Counter(ref_grams)

    total_target = sum(target_counts.values())
    total_ref = sum(ref_counts.values())

    vocab = set(target_counts) | set(ref_counts)
    rows = []
    for gram in vocab:
        a = target_counts.get(gram, 0)  # freq in target
        b = ref_counts.get(gram, 0)  # freq in reference
        if a == 0 and b == 0:
            continue
        c = total_target
        d = total_ref
        e1 = c * (a + b) / (c + d)
        e2 = d * (a + b) / (c + d)
        ll = 0.0
        if a > 0:
            ll += a * math.log(a / e1)
        if b > 0:
            ll += b * math.log(b / e2)
        ll *= 2
        sign = 1 if (a / c if c else 0) > (b / d if d else 0) else -1
        rows.append(
            {"ngram": gram, "freq_target": a, "freq_reference": b, "LL": ll, "sign": sign}
        )

    df = pd.DataFrame(rows).sort_values("LL", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Syntactic complexity: MDD and subordination ratio
# ---------------------------------------------------------------------------

_SUBORDINATE_DEPS = {"advcl", "ccomp", "xcomp", "acl", "relcl", "csubj"}


def syntactic_complexity(text: str) -> dict:
    """Mean dependency distance (MDD; Liu, 2008) — mean absolute linear
    distance in tokens between each word and its syntactic head — and the
    subordination ratio (proportion of subordinate clausal dependents:
    advcl, ccomp, xcomp, acl, relcl, csubj, relative to root clauses),
    as described in Section 3.5.
    """
    doc = get_nlp()(text)
    distances = []
    n_subordinate = 0
    n_root = 0

    for sent in doc.sents:
        for tok in sent:
            if tok.dep_ == "ROOT":
                n_root += 1
                continue
            if tok.head != tok:
                distances.append(abs(tok.i - tok.head.i))
            if tok.dep_ in _SUBORDINATE_DEPS:
                n_subordinate += 1

    mdd = float(np.mean(distances)) if distances else float("nan")
    subordination_ratio = (n_subordinate / n_root) if n_root else float("nan")
    return {"mdd": mdd, "subordination_ratio": subordination_ratio}


# ---------------------------------------------------------------------------
# 4. Lexical frequency profiling (LFP)
# ---------------------------------------------------------------------------

_AWL_SUBSET_NOTE = (
    "Provide your own curated or full Academic Word List (Coxhead, 2000) "
    "headword set to `awl_headwords`. The manuscript explicitly flags its "
    "own AWL figure as based on a curated ~150-headword subset, not the "
    "complete validated 570-family list (Section 3.5, Section 5.4) — using "
    "the complete list here will NOT reproduce the paper's reported AWL "
    "percentages, and is preferable for any new analysis."
)


def lexical_frequency_profile(text: str, awl_headwords: Iterable[str] | None = None) -> dict:
    """Classifies content words into Zipf-frequency bands as a contemporary
    proxy for the K1/K2 frequency-band distinction (Laufer & Nation, 1995;
    Nation, 2001), using the wordfreq corpus-derived frequency database
    (Speer, 2022), per Section 3.5:
        K1-equivalent:    Zipf >= 5.0
        K2-equivalent:    4.0 <= Zipf < 5.0
        off-list:         Zipf < 4.0
    Optionally also reports the proportion of tokens found in a supplied
    Academic Word List headword set. See `_AWL_SUBSET_NOTE` above.
    """
    if zipf_frequency is None:
        raise ImportError("wordfreq is required: pip install wordfreq")

    words = [w.lower() for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)]
    if not words:
        return {"k1": float("nan"), "k2": float("nan"), "off_list": float("nan"), "awl": float("nan")}

    bands = {"k1": 0, "k2": 0, "off_list": 0}
    for w in words:
        z = zipf_frequency(w, "en")
        if z >= 5.0:
            bands["k1"] += 1
        elif z >= 4.0:
            bands["k2"] += 1
        else:
            bands["off_list"] += 1

    n = len(words)
    result = {k: v / n for k, v in bands.items()}

    if awl_headwords is not None:
        awl_set = {w.lower() for w in awl_headwords}
        awl_hits = sum(1 for w in words if w in awl_set)
        result["awl"] = awl_hits / n
    else:
        result["awl"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# 5. Sentence-level nonparametric bootstrap for FK Grade
# ---------------------------------------------------------------------------

def sentence_level_bootstrap_fk(
    group_sentences: dict[str, list[str]],
    n_resamples: int = 5000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Nonparametric bootstrap (Efron & Tibshirani, 1993) resampling
    SENTENCES (not passages) with replacement within each group, and
    recomputing an aggregate FK Grade from each resampled set, as
    described in Section 3.5 / 4.4.

    `group_sentences` maps a group label (e.g. "CSAT_2025") to a list of
    that group's individual sentences (strings).

    CAUTION (see Section 4.4 / Section 5.4 of the manuscript, and the
    reviewer-response edit applied there): sentences within the same
    passage are not independent draws — they share author, topic, and
    register. This bootstrap's precision should be read as describing
    within-sample sentence-level variability, not as a population-level
    sampling interval with the same precision a passage-level cluster
    bootstrap would yield. Consider also running a passage-level cluster
    bootstrap (resampling whole passages, then pooling their sentences)
    as a sensitivity check before relying on this interval alone.
    """
    from ctci_pipeline import flesch_kincaid_grade  # local import to avoid hard dependency at module load

    rng = np.random.default_rng(random_state)
    rows = []

    for group, sentences in group_sentences.items():
        sentences = [s for s in sentences if s.strip()]
        n = len(sentences)
        if n == 0:
            continue

        means = []
        for _ in range(n_resamples):
            sample = rng.choice(sentences, size=n, replace=True)
            joined = " ".join(sample)
            fk = flesch_kincaid_grade(joined)
            if not math.isnan(fk):
                means.append(fk)

        if not means:
            continue

        means = np.array(means)
        rows.append(
            {
                "group": group,
                "n_sentences": n,
                "mean_fk": float(np.mean(means)),
                "ci_lower": float(np.percentile(means, 2.5)),
                "ci_upper": float(np.percentile(means, 97.5)),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_group_difference(
    sentences_a: list[str],
    sentences_b: list[str],
    n_resamples: int = 5000,
    random_state: int | None = None,
) -> dict:
    """Bootstrap the FK Grade difference (A - B) between two groups'
    pooled sentences, returning the mean difference and 95% CI, as
    reported for the CSAT-GSAT comparison in Section 4.4.
    """
    from ctci_pipeline import flesch_kincaid_grade

    rng = np.random.default_rng(random_state)
    diffs = []
    na, nb = len(sentences_a), len(sentences_b)

    for _ in range(n_resamples):
        sa = rng.choice(sentences_a, size=na, replace=True)
        sb = rng.choice(sentences_b, size=nb, replace=True)
        fk_a = flesch_kincaid_grade(" ".join(sa))
        fk_b = flesch_kincaid_grade(" ".join(sb))
        if not (math.isnan(fk_a) or math.isnan(fk_b)):
            diffs.append(fk_a - fk_b)

    diffs = np.array(diffs)
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "n_resamples_used": len(diffs),
    }


# ---------------------------------------------------------------------------
# Asyndetic-coordination coding checklist (manual coding aid, not an
# automatic classifier — see module docstring).
# ---------------------------------------------------------------------------

SYNDETIC_MARKERS = {
    "coordinating_conjunctions": ["but", "and", "or", "yet", "so", "for"],
    "conjunctive_adverbs_subordinators": [
        "however", "therefore", "while", "although", "thus", "hence",
        "nevertheless", "moreover", "furthermore", "consequently",
    ],
}


def find_semicolon_contexts(text: str, window: int = 60) -> list[dict]:
    """Returns the text surrounding every semicolon in `text`, plus a flag
    for whether a syndetic marker immediately follows it, as a STARTING
    POINT for the manual coding described in Section 3.5 ("Asyndetic
    coordination") — the final classification should still be checked by
    a human coder, exactly as the manuscript's own analysis was (Section
    5.4 notes this coding was performed by a single rater, a limitation
    of the original analysis worth avoiding by having two raters check
    this list independently for any new corpus).
    """
    all_markers = (
        SYNDETIC_MARKERS["coordinating_conjunctions"]
        + SYNDETIC_MARKERS["conjunctive_adverbs_subordinators"]
    )
    marker_re = re.compile(r"^\s*(" + "|".join(all_markers) + r")\b", re.IGNORECASE)

    results = []
    for m in re.finditer(";", text):
        i = m.start()
        after = text[i + 1 : i + 1 + window]
        is_syndetic = bool(marker_re.match(after))
        results.append(
            {
                "context": text[max(0, i - window) : i + window],
                "likely_syndetic": is_syndetic,
            }
        )
    return results


if __name__ == "__main__":
    print("This module provides functions; import it rather than running directly.")
    print("See README.md for example usage.")
