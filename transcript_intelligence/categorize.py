"""
Topic / theme categorization.

Why hybrid, not pure LLM or pure clustering (numbers reproduced in
notebooks/02_categorization.ipynb):

1. Pure unsupervised clustering (sentence-transformer embeddings of
   title+summary+topics, KMeans, k swept 5-12) tops out at a silhouette
   score of ~0.066 -- weak separation. These are all professional B2B
   calls between the same vendor and its customers, so surface vocabulary
   overlaps heavily across categories (product names, standard call
   phrasing). Clustering alone produced fuzzy, hard-to-defend buckets.

2. A naive keyword rule over the `topics` field alone over-triggers: tags
   like "incident post-mortem", "reliability concern", or "technical
   incident" show up on renewal calls, audit-prep calls, and SOC-2 reviews
   -- because a live outage is a running storyline that bleeds into
   almost every conversation that quarter, not because those calls are
   *about* the outage. An early version of this rulebook put 55/100 calls
   into "Incident & Outage" for exactly this reason.

3. The fix: this company already names its calls with a clear, consistent
   convention ("Support Case #1234 - ...", "Aegis / <Customer> - ...",
   "Detect Outage - ..."). That title convention is a stronger, more
   precise signal than free-text topic tags, so it's tier 1. Topic tags
   are only used as tier-2 tie-breakers on the two title patterns that are
   genuinely ambiguous (numbered support cases, and customer account
   calls) -- and even then we require specific compound phrases (e.g.
   "platform outage", not bare "incident") to avoid the same over-trigger.

4. Embedding similarity (`validate_themes`) is kept as a lightweight
   sanity layer: it flags any call whose theme centroid similarity is
   *lower* than the closest alternate theme, i.e. calls a human should
   double check.

This keeps categorization deterministic, explainable to a support/sales
leader ("why is this call tagged Compliance?"), and free of API costs --
while still using embeddings to validate the taxonomy rather than taking
the rules on faith.
"""

from __future__ import annotations

import re

import pandas as pd

FALLBACK_THEME = "Other / General Account Discussion"

_BILLING_KW = ["billing", "invoice", "overage", "charges", "license upgrade"]
# Deliberately narrow: only phrases tied to the *systemic* Detect pipeline
# event. Generic "service outage" / "api outage" tags also show up on
# isolated, single-customer auth or integration bugs (see
# notebooks/02_categorization.ipynb "borderline calls" section) -- those
# belong in Technical Support, not the flagship incident bucket.
_OUTAGE_KW = [
    "platform outage", "detect pipeline failure", "complete loss of threat visibility",
    "dashboard down", "event ingestion pipeline", "outage recovery",
    "sla breach", "sla credits",
]
_FEATURE_KW = [
    "custom report request", "restore request", "feature request",
    "custom compliance template", "custom report",
]

_AEGIS_RENEWAL_KW = ["renewal", "contract", "account review", "account recovery", "pricing"]
_AEGIS_ONBOARD_KW = ["onboarding", "deployment"]
_AEGIS_COMPETITIVE_KW = ["competitive", "vendor comparison"]
_AEGIS_COMPLIANCE_KW = ["compliance", "audit", "hipaa", "iso 27001", "pci dss", "soc 2"]
_AEGIS_FEEDBACK_KW = ["demo", "feedback", "preview"]
_AEGIS_ROADMAP_KW = ["roadmap"]
_AEGIS_INCIDENT_KW = ["urgent", "outage impact", "service reliability", "platform concerns"]


def _classify_title_tier(title_lc: str) -> str | None:
    """Tier 1: the company's own title convention. Highest precision."""
    if "detect outage" in title_lc:
        return "Incident & Outage Management"
    if title_lc.startswith("incident:"):
        return "Incident & Outage Management"
    if any(p in title_lc for p in ["war room", "escalation bridge", "root cause analysis",
                                    "post-incident review", "detect reliability"]):
        return "Incident & Outage Management"
    if "soc 2" in title_lc:
        return "Compliance & Audit"
    if any(p in title_lc for p in ["identity team", "product sync", "sprint planning",
                                    "sprint retro", "standup", "all hands",
                                    "quarterly planning", "scalability concerns"]):
        return "Product Roadmap & Planning"
    # "Comply v2 - ..." (startswith, not `in`) is the internal product
    # team's own naming convention. A support ticket that merely mentions
    # the product name ("... Comply v2 Report Formatting Issue") must NOT
    # match here -- that's a support case, not an internal planning call.
    if title_lc.startswith("comply v2"):
        return "Product Roadmap & Planning"
    if "win/loss" in title_lc or title_lc.startswith("competitive"):
        return "Competitive Intelligence"
    return None


def _classify_support_case(haystack: str) -> str:
    """Tier 2a: numbered support tickets. Requires exact compound phrases
    (not bare 'incident'/'outage') to avoid over-triggering on shared
    vocabulary."""
    if any(kw in haystack for kw in _BILLING_KW):
        return "Billing & Account Issues"
    if any(kw in haystack for kw in _OUTAGE_KW):
        return "Incident & Outage Management"
    if any(kw in haystack for kw in _FEATURE_KW):
        return "Product Feedback & Feature Requests"
    return "Technical Support & Bug Resolution"


def _classify_aegis_account_call(title_lc: str, haystack: str) -> str:
    """Tier 2b: 'Aegis / <Customer> - ...' relationship calls."""
    if any(kw in title_lc for kw in _AEGIS_INCIDENT_KW):
        return "Incident & Outage Management"
    if any(kw in title_lc for kw in _AEGIS_RENEWAL_KW):
        return "Commercial: Renewal & Retention"
    if any(kw in title_lc for kw in _AEGIS_ONBOARD_KW):
        return "Onboarding & Deployment"
    if any(kw in title_lc for kw in _AEGIS_COMPETITIVE_KW):
        return "Competitive Intelligence"
    if any(kw in title_lc for kw in _AEGIS_COMPLIANCE_KW):
        return "Compliance & Audit"
    if any(kw in title_lc for kw in _AEGIS_FEEDBACK_KW):
        return "Product Feedback & Feature Requests"
    if any(kw in title_lc for kw in _AEGIS_ROADMAP_KW):
        return "Product Roadmap & Planning"
    # fall through to topic tags before giving up
    if any(kw in haystack for kw in _OUTAGE_KW):
        return "Incident & Outage Management"
    return "Commercial: Renewal & Retention"  # default: it's an AM-led relationship call


def classify_theme(title: str, topics: list[str]) -> str:
    title_lc = title.lower()
    haystack = title_lc + " " + " ".join(topics).lower()

    tier1 = _classify_title_tier(title_lc)
    if tier1:
        return tier1

    if re.match(r"^support case #\d+", title_lc):
        return _classify_support_case(haystack)

    if title.startswith("Aegis /"):
        return _classify_aegis_account_call(title_lc, haystack)

    if title_lc.upper().startswith("URGENT:") or title_lc.upper().startswith("ESCALATION:"):
        return "Incident & Outage Management"

    return FALLBACK_THEME


def add_theme_column(meetings_df: pd.DataFrame) -> pd.DataFrame:
    df = meetings_df.copy()
    df["theme"] = [
        classify_theme(title, topics)
        for title, topics in zip(df["title"], df["topics"])
    ]
    return df


def embed_meetings(meetings_df: pd.DataFrame):
    """Sentence-transformer embeddings of title+summary+topics per meeting.

    Used only for validation (theme coherence, outlier flagging), not as
    the primary categorization signal. Falls back to a TF-IDF + SVD
    representation if the model can't be downloaded (e.g. no internet in
    a locked-down environment), so the notebook still runs end to end.
    """
    docs = (
        meetings_df["title"]
        + ". "
        + meetings_df["summary"].fillna("")
        + ". Topics: "
        + meetings_df["topics"].apply(lambda ts: ", ".join(ts))
    ).tolist()

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(docs, show_progress_bar=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[categorize] sentence-transformers unavailable ({exc}); "
              f"falling back to TF-IDF+SVD embeddings.")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
        X = tfidf.fit_transform(docs)
        svd = TruncatedSVD(n_components=min(50, X.shape[1] - 1), random_state=42)
        return svd.fit_transform(X)


def validate_themes(meetings_df: pd.DataFrame, embeddings) -> pd.DataFrame:
    """For each meeting, cosine similarity to its own theme's centroid vs.
    the best-fitting *other* theme's centroid. A negative margin flags a
    call worth a human second look.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    df = meetings_df.copy()
    themes = df["theme"].unique()
    centroids = {
        theme: embeddings[df["theme"].values == theme].mean(axis=0)
        for theme in themes
    }
    centroid_matrix = np.vstack([centroids[t] for t in themes])
    sims = cosine_similarity(embeddings, centroid_matrix)  # (n_meetings, n_themes)

    own_idx = np.array([list(themes).index(t) for t in df["theme"]])
    own_sim = sims[np.arange(len(df)), own_idx]

    sims_masked = sims.copy()
    sims_masked[np.arange(len(df)), own_idx] = -1
    best_other_sim = sims_masked.max(axis=1)
    best_other_theme = np.array(themes)[sims_masked.argmax(axis=1)]

    df["theme_fit_margin"] = own_sim - best_other_sim
    df["closest_alt_theme"] = best_other_theme
    return df
