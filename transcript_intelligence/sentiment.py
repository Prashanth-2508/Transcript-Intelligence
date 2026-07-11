"""
Sentiment analysis across call types.

The dataset already ships with AI-derived sentiment at two levels:
  - per-sentence `sentimentType` (positive/neutral/negative) in transcript.json
  - per-call `overallSentiment` (very-negative..very-positive) and a
    continuous `sentimentScore` (1-5) in summary.json

Rather than throwing that away and re-running sentiment from scratch (which
would just be a noisier version of work already done upstream), we:
  1. Independently score every sentence with VADER (a lexicon-based
     sentiment model, not part of whatever pipeline produced this
     dataset) purely as a cross-check.
  2. Report how well the two agree. High agreement -> we can trust the
     provided labels and spend our effort on aggregation/trend analysis,
     which is where the actual business value is.
  3. Use the provided per-call `sentiment_score` as the primary metric for
     trend analysis, since it's more granular (continuous, 1-5) than a
     3-class re-derivation would be.
"""

from __future__ import annotations

import pandas as pd


def score_with_vader(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """Independent cross-check using VADER. Returns sentences_df with an
    added `vader_compound` and `vader_label` column."""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        sia = SentimentIntensityAnalyzer()
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[sentiment] VADER unavailable ({exc}); skipping cross-check.")
        df = sentences_df.copy()
        df["vader_compound"] = float("nan")
        df["vader_label"] = None
        return df

    df = sentences_df.copy()
    df["vader_compound"] = df["sentence"].fillna("").apply(
        lambda s: sia.polarity_scores(s)["compound"]
    )
    df["vader_label"] = pd.cut(
        df["vader_compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    )
    return df


def agreement_rate(sentences_df: pd.DataFrame) -> float:
    """Share of sentences where VADER's 3-class label matches the
    dataset's own `sentiment_type`."""
    valid = sentences_df.dropna(subset=["vader_label", "sentiment_type"])
    if len(valid) == 0:
        return float("nan")
    matches = (valid["vader_label"].astype(str) == valid["sentiment_type"].astype(str)).sum()
    return matches / len(valid)


def call_type_sentiment_summary(meetings_df: pd.DataFrame) -> pd.DataFrame:
    """Per-call-type aggregate sentiment stats for the trend narrative."""
    g = meetings_df.groupby("call_type", observed=True)
    out = g.agg(
        n_calls=("meeting_id", "count"),
        avg_sentiment_score=("sentiment_score", "mean"),
        pct_negative_or_mixed_negative=(
            "overall_sentiment",
            lambda s: s.isin(["negative", "very-negative", "mixed-negative"]).mean(),
        ),
        avg_churn_signals=("n_churn_signals", "mean"),
        avg_concerns=("n_concerns", "mean"),
        avg_technical_issues=("n_technical_issues", "mean"),
        avg_positive_pivots=("n_positive_pivots", "mean"),
    )
    return out.round(3)


def theme_sentiment_summary(meetings_df: pd.DataFrame) -> pd.DataFrame:
    """Per-theme aggregate sentiment -- which conversation topics run
    hottest/coldest, independent of who's on the call."""
    g = meetings_df.groupby("theme", observed=True)
    out = g.agg(
        n_calls=("meeting_id", "count"),
        avg_sentiment_score=("sentiment_score", "mean"),
        pct_negative_or_mixed_negative=(
            "overall_sentiment",
            lambda s: s.isin(["negative", "very-negative", "mixed-negative"]).mean(),
        ),
    ).round(3)
    return out.sort_values("avg_sentiment_score")


def sentiment_over_time(meetings_df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Rolling/weekly average sentiment per call type, for the trend line."""
    df = meetings_df.set_index("start_time")
    out = (
        df.groupby("call_type", observed=True)["sentiment_score"]
        .resample(freq)
        .mean()
        .reset_index()
    )
    return out
