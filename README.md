# Transcript Intelligence — Take-Home Solution

## What's here

| Deliverable | Location |
|---|---|
| Slide deck (leadership-facing) | `slides/Transcript_Intelligence_Findings.pptx` — ready to present, with full speaker notes on every slide. `slides/slide_deck_outline.md` is the source-of-truth content if you want to edit it. |
| Notebook (technical reference) | `notebooks/Transcript_Intelligence_Analysis.ipynb` |
| Reusable pipeline code | `transcript_intelligence/` |
| Generated figures | `outputs/figures/` (embedded in both the notebook and the deck) |

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Transcript_Intelligence_Analysis.ipynb
```

First run downloads a small sentence-embedding model (~90MB, cached after) and the VADER lexicon
(~1MB). Both have offline fallbacks (see `categorize.py` / `sentiment.py`) so the notebook still
runs without internet, just with slightly weaker validation signal.

## Repo layout

```
transcript_intelligence/
  loader.py       # walks dataset/, builds meetings / sentences / events DataFrames,
                   # infers call_type (support/external/internal) since it's not labeled
  categorize.py    # theme classification (title-pattern rules + topic-tag tie-breakers),
                   # plus embedding-based validation of the taxonomy
  sentiment.py     # VADER cross-check of the dataset's own sentiment labels,
                   # call-type / theme / time-series aggregation
  insights.py      # bonus insights: account health scoring, talk-time balance,
                   # incident blast-radius analysis
notebooks/
  Transcript_Intelligence_Analysis.ipynb   # the full story, end to end, with reasoning
outputs/figures/    # every chart, saved as PNG (also embedded in the notebook)
slides/, video/      # the other two deliverables
```

## Key decisions at a glance

- **Call type isn't labeled in the dataset** — inferred from email-domain count + the company's
  own consistent title convention (`Support Case #...`, `Aegis / <Customer> - ...`). 100% coverage,
  fully rule-based, no manual labeling.
- **Categorization is hybrid, not pure LLM or pure clustering** — clustering alone gave weak
  separation (silhouette ~0.07); naive topic-keyword rules over-triggered on shared incident
  vocabulary (55/100 calls in one bucket). Final approach: title patterns first (highest
  precision), topic tags as a narrow tie-breaker, embeddings used only to validate the result
  (borderline-case flagging + agreement check against independent clustering).
- **Sentiment isn't re-derived from scratch** — the dataset already ships AI-extracted sentiment.
  We cross-checked it independently with VADER rather than replacing it: raw label agreement is
  low (37%) because of where the neutral/positive line sits, but call-level correlation is strong
  (r≈0.84) — evidence the provided scores are trustworthy, so effort went into trend analysis
  instead of reinventing sentiment scoring.
- **No synthetic data was generated.** The 100 provided transcripts already carry rich structured
  fields (topics, sentiment, key moments); the higher-value work was building a defensible,
  validated pipeline on top of real data.

See the notebook for the full reasoning behind each of these, including the dead ends.
