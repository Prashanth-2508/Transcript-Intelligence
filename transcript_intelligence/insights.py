"""
Bonus insights beyond the two required tasks.

Implemented here (all derived from fields already in the dataset -- no
extra data needed):

1. talk_time_balance   -- who dominates customer-facing calls, Aegis or
                           the customer? A coaching / call-quality signal
                           for support & AM leaders.
2. account_health_scores -- a per-customer risk score blending sentiment,
                           topic tags, and key-moment signals, so a CS/sales
                           leader can see which accounts need attention
                           without reading every transcript.
3. incident_blast_radius -- quantifies how far a single reliability event
                           (the March Detect outage) rippled into support
                           and commercial conversations, and for how long.

Two further ideas are described (not implemented) in
notebooks/03_bonus_insights.ipynb: action-item accountability tracking,
and a "next-best-action" recommender per call.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Talk-time balance
# ---------------------------------------------------------------------------

def identify_aegis_speakers(sentences_df: pd.DataFrame, meetings_df: pd.DataFrame) -> set[str]:
    """The dataset never labels a speaker's company directly. We infer it:
    a name that (a) appears on more than one distinct customer's calls, or
    (b) appears on any internal-only call, must be an Aegis employee --
    only Aegis staff sit in on multiple accounts or purely internal
    meetings. Every other name is assumed to be that single customer's
    staff. Verified on this dataset: the two sets partition all 76 speaker
    names with no leftover ambiguity (see notebooks/03).
    """
    merged = sentences_df.merge(
        meetings_df[["meeting_id", "customer_domain", "call_type"]], on="meeting_id"
    )
    multi_domain = set(
        merged.groupby("speaker_name")["customer_domain"]
        .apply(lambda x: len(set(x.dropna())))
        .loc[lambda x: x > 1]
        .index
    )
    internal_call_speakers = set(
        merged.loc[merged["call_type"] == "internal", "speaker_name"].unique()
    )
    return multi_domain | internal_call_speakers


def talk_time_balance(
    sentences_df: pd.DataFrame,
    meetings_df: pd.DataFrame,
    aegis_speakers: set[str],
    imbalance_threshold: float = 0.60,
) -> pd.DataFrame:
    """Per customer-facing call: share of speaking time held by Aegis vs.
    the customer, using each sentence's (end_time - time) as a duration
    proxy. Flags calls where one side holds more than `imbalance_threshold`
    of talk time (default 60/40 -- on this sample, turns are fairly
    balanced, so a stricter 70/30 cutoff flags nothing; 60/40 is enough to
    separate a genuinely rep-led call, e.g. a demo, from a listening-mode
    call, e.g. a competitive evaluation). Tune per-org once real call
    data is available."""
    df = sentences_df.copy()
    df["talk_seconds"] = (df["end_time"] - df["time"]).clip(lower=0)
    df["side"] = df["speaker_name"].apply(
        lambda n: "aegis" if n in aegis_speakers else "customer"
    )

    customer_facing = meetings_df[meetings_df["call_type"] != "internal"][
        ["meeting_id", "title", "call_type", "customer_domain", "theme"]
    ]
    df = df.merge(customer_facing, on="meeting_id")  # drops internal calls, no "customer" side there

    pivot = (
        df.groupby(["meeting_id", "side"])["talk_seconds"]
        .sum()
        .unstack(fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot["aegis_share"] = pivot.get("aegis", 0) / pivot["total"]

    out = customer_facing.set_index("meeting_id").join(pivot[["aegis_share", "total"]])
    out = out.rename(columns={"total": "talk_seconds_total"}).reset_index()
    out["flag_imbalanced"] = (out["aegis_share"] > imbalance_threshold) | (
        out["aegis_share"] < 1 - imbalance_threshold
    )
    return out.sort_values("aegis_share", ascending=False)


# ---------------------------------------------------------------------------
# 2. Account health / churn-risk score
# ---------------------------------------------------------------------------

def account_health_scores(meetings_df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer account, aggregating every call about that
    account into a single 0-100 health score (100 = healthiest).

    Score = weighted blend of:
      - average sentiment_score across all their calls        (50%)
      - churn_risk topic tag frequency                          (25%)
      - churn_signal / concern key-moment frequency              (25%)

    This is intentionally simple and auditable -- a sales/CS leader can
    trace exactly why an account scored low, rather than trusting a black
    box. Weights are a starting point, easy to retune once someone with
    ground-truth renewal outcomes can validate against them.
    """
    df = meetings_df[meetings_df["customer_domain"].notna()].copy()
    df["has_churn_topic"] = df["topics"].apply(
        lambda ts: any("churn" in t.lower() for t in ts)
    )

    g = df.groupby("customer_domain")
    acc = g.agg(
        n_calls=("meeting_id", "count"),
        avg_sentiment_score=("sentiment_score", "mean"),
        pct_calls_with_churn_topic=("has_churn_topic", "mean"),
        avg_churn_signals=("n_churn_signals", "mean"),
        avg_concerns=("n_concerns", "mean"),
        last_call=("start_time", "max"),
        themes_seen=("theme", lambda s: sorted(set(s))),
    ).reset_index()

    # normalize each component to 0-1, higher = healthier
    sent_norm = (acc["avg_sentiment_score"] - 1) / (5 - 1)  # sentiment_score is on a 1-5 scale
    churn_topic_norm = 1 - acc["pct_calls_with_churn_topic"]
    signal_norm = 1 - (
        (acc["avg_churn_signals"] + acc["avg_concerns"])
        / (acc["avg_churn_signals"] + acc["avg_concerns"]).max()
    ).fillna(0)

    acc["health_score"] = (
        50 * sent_norm.clip(0, 1) + 25 * churn_topic_norm + 25 * signal_norm
    ).round(1)

    return acc.sort_values("health_score").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Incident blast radius
# ---------------------------------------------------------------------------

def incident_blast_radius(
    meetings_df: pd.DataFrame,
    incident_title_pattern: str = "Detect Outage|INCIDENT:",
    window_days: int = 21,
) -> dict:
    """Quantifies how far one reliability incident rippled outward:
    how many distinct customer accounts show up in support/external calls
    in the window right after it, and how sentiment by call type moved
    week over week through and after the window.
    """
    incident_calls = meetings_df[
        meetings_df["title"].str.contains(incident_title_pattern, regex=True)
    ]
    if incident_calls.empty:
        return {}

    start = incident_calls["start_time"].min()
    end = start + pd.Timedelta(days=window_days)

    window_df = meetings_df[
        (meetings_df["start_time"] >= start) & (meetings_df["start_time"] <= end)
    ]
    affected_customers = sorted(window_df["customer_domain"].dropna().unique())

    weekly = (
        meetings_df.set_index("start_time")
        .groupby("call_type", observed=True)["sentiment_score"]
        .resample("W")
        .mean()
        .reset_index()
    )

    baseline = meetings_df[meetings_df["start_time"] < start].groupby(
        "call_type", observed=True
    )["sentiment_score"].mean()
    trough = window_df.groupby("call_type", observed=True)["sentiment_score"].min()

    return {
        "incident_window_start": start,
        "incident_window_end": end,
        "incident_calls": incident_calls[["title", "start_time"]],
        "n_affected_customers": len(affected_customers),
        "affected_customers": affected_customers,
        "baseline_sentiment_by_type": baseline.round(2),
        "trough_sentiment_by_type": trough.round(2),
        "weekly_sentiment": weekly,
    }
