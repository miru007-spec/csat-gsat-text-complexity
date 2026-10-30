"""
run_example.py
===============
End-to-end sanity check using synthetic (non-copyrighted) sample
passages and a tiny stub MRC-style lexicon. This does NOT reproduce
any numbers from the manuscript — it only demonstrates that the
pipeline runs correctly on arbitrary English text.

Run from the repository root:

    python example/run_example.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from ctci_pipeline import run_ctci_pipeline
from supplementary_analysis import (
    compound_noun_density,
    syntactic_complexity,
    lexical_frequency_profile,
)

SAMPLE_PASSAGES = pd.DataFrame(
    [
        {
            "doc_id": "SAMPLE_01",
            "group": "A",
            "text": (
                "The committee's decision reflects a broader tension between "
                "institutional accountability and individual autonomy. "
                "Although the proposal was framed as a procedural adjustment, "
                "its implications extend well beyond administrative "
                "convenience, raising a question about whether efficiency "
                "and fairness can be pursued simultaneously."
            ),
        },
        {
            "doc_id": "SAMPLE_02",
            "group": "B",
            "text": (
                "Last weekend I went hiking with my brother. We packed "
                "sandwiches and water bottles before we left. The trail was "
                "muddy because it had rained the night before, but the view "
                "from the top was worth the climb."
            ),
        },
    ]
)

LEXICON_PATH = os.path.join(os.path.dirname(__file__), "sample_lexicon.csv")


def main():
    result = run_ctci_pipeline(SAMPLE_PASSAGES, LEXICON_PATH)
    print("\n=== CTCI pipeline output ===")
    print(result[["doc_id", "group", "fk_grade", "mattr", "ctci"]])

    print("\n=== Supplementary analyses (SAMPLE_01) ===")
    text = SAMPLE_PASSAGES.loc[0, "text"]
    print("compound-noun density:", round(compound_noun_density(text), 2))
    print("syntactic complexity:", syntactic_complexity(text))
    print("lexical frequency profile:", lexical_frequency_profile(text))

    print("\nSanity check complete — no numbers above correspond to the manuscript.")


if __name__ == "__main__":
    main()
