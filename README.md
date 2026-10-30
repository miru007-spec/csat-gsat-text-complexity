# CSAT–GSAT Text Complexity Pipeline

Reference implementation of the Composite Text Complexity Index (CTCI)
and supplementary analyses described in:

> Lee, H. (2026+). *Text complexity in high-stakes EFL examinations: a
> comparative Coh-Metrix study of Korea's CSAT and Taiwan's GSAT.*
> Submitted to *System* (Elsevier).

## What this is

This repository provides a documented, runnable Python implementation
of the six-variable CTCI (Section 3.2) and the five supplementary
analyses (Section 3.5) described in the manuscript's Methods section:
Flesch-Kincaid Grade Level, MATTR, connective density,
imageability/concreteness, nominalization index, discourse coherence,
compound-noun density, N-gram keyness, syntactic complexity (mean
dependency distance and subordination ratio), lexical frequency
profiling, and sentence-level bootstrap resampling.

**This code was written independently from the manuscript's Methods
description, after the original analysis was run**, to provide a
transparent, reproducible reference implementation for this data
availability statement. It may differ in minor implementation details
from whatever script originally produced the numbers reported in the
paper (for example, in edge-case handling of sentence splitting or
syllable counting); it is not claimed to be a byte-for-byte replica of
that original run.

## What this is NOT

- **It does not include the original CSAT or GSAT passage texts.**
  These are copyrighted by the Korea Institute for Curriculum and
  Evaluation (KICE) and Taiwan's College Entrance Examination Center
  (CEEC) respectively. Qualifying researchers may request the source
  materials directly from those institutions.
- **It does not include the MRC Psycholinguistic Database** used for
  the imageability/concreteness variable, which has its own license
  terms. `ctci_pipeline.load_mrc_lexicon()` expects a two-column CSV
  (`word`, `concreteness`, 100–700 scale) that you supply.
- **It does not include a full Academic Word List (AWL) checker.** The
  manuscript's own AWL figure is explicitly based on a curated
  ~150-headword subset of Coxhead (2000), not the complete validated
  570-family list (see Section 3.5 and Section 5.4 of the manuscript);
  `lexical_frequency_profile()` accepts any headword set you supply.
- **It does not automatically classify semicolons as syndetic or
  asyndetic.** The manuscript's asyndetic-coordination analysis
  (Section 3.5, Section 5.4) was performed by hand by a single rater;
  `find_semicolon_contexts()` here is a coding aid (it flags likely
  syndetic markers) to speed up manual review, not an automatic
  classifier that reproduces the paper's reported counts on its own.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Core CTCI pipeline

Input: a CSV with columns `doc_id`, `text` (plus any metadata columns
you want preserved, e.g. `exam`, `year`).

```bash
python ctci_pipeline.py passages.csv mrc_lexicon.csv output.csv
```

Or from Python:

```python
import pandas as pd
from ctci_pipeline import run_ctci_pipeline

passages = pd.read_csv("passages.csv")
result = run_ctci_pipeline(passages, "mrc_lexicon.csv")
result.to_csv("output.csv", index=False)
```

`result` contains the six raw variables, their min-max normalized
values, and the final `ctci` column.

**Important:** CTCI is normalized *within whatever passage set you pass
in*. Computing CTCI over a different-sized or differently composed
corpus will not produce numerically comparable scores — this is the
same normalization-reference-set caveat discussed in Section 4.5 and
Section 5.4 of the manuscript. Always run the full pooled comparison
set through `compute_ctci()` together.

### Supplementary analyses

```python
from supplementary_analysis import (
    compound_noun_density,
    syntactic_complexity,
    lexical_frequency_profile,
    log_likelihood_keyness,
    sentence_level_bootstrap_fk,
    bootstrap_group_difference,
    find_semicolon_contexts,
)

text = "..."
print(compound_noun_density(text))
print(syntactic_complexity(text))
print(lexical_frequency_profile(text, awl_headwords=my_awl_list))
```

See the docstring of each function for the exact formula/procedure it
implements and the corresponding section of the manuscript.

## A note on the sentence-level bootstrap

`sentence_level_bootstrap_fk()` and `bootstrap_group_difference()`
resample *sentences*, not *passages*, as the unit — matching Section
3.5/4.4 of the manuscript. As the manuscript's own revised Section 4.4
now notes, sentences within the same passage are not fully independent
of one another (they share author, topic, and register), so this
interval should be read as describing variability within the sampled
sentences rather than a population-level sampling interval with the
same precision a passage-level cluster bootstrap would yield. A
passage-level cluster bootstrap (resample whole passages with
replacement, then pool their sentences) is a reasonable sensitivity
check to run alongside this one.

## License

MIT License (see `LICENSE`). This covers the code only; it does not
grant any rights to the CSAT/GSAT source passages, which remain the
property of KICE and CEEC respectively.

## Citation

If you use this code, please cite the manuscript above.
