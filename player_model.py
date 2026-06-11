"""
Player Performance & Clustering Module
- Loads compact wc_players.csv (built by build_player_data.py)
- KMeans clustering by playing style
- Stats aggregation for visualisation
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Features used for style clustering
CLUSTER_FEATURES = [
    "goals_per90", "assists_per90", "cards_per90",
    "mins_per_app", "height_in_cm", "age",
]

STYLE_LABELS = {
    0: "Prolific Finisher",
    1: "Creative Playmaker",
    2: "Defensive Anchor",
    3: "Box-to-Box Engine",
    4: "Squad Rotation",
    5: "Veteran Leader",
}

_PCACHE = {}

def load_players(path):
    if path in _PCACHE:
        return _PCACHE[path]
    df = pd.read_csv(path)
    # Fill NA
    for c in CLUSTER_FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    df["market_value_in_eur"] = df["market_value_in_eur"].fillna(0)
    df["international_caps"] = df["international_caps"].fillna(0)
    df["international_goals"] = df["international_goals"].fillna(0)
    _PCACHE[path] = df
    return df


def run_clustering(df, n_clusters=6, position_filter=None):
    """KMeans clustering on playing style. Returns df with 'cluster' + 'style' columns."""
    work = df.copy()
    if position_filter and position_filter != "All":
        work = work[work["position"] == position_filter].copy()

    if len(work) < n_clusters:
        work["cluster"] = 0
        work["style"] = "Insufficient data"
        return work, None

    X = work[CLUSTER_FEATURES].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    work["cluster"] = km.fit_predict(Xs)

    # Auto-label clusters by their characteristics
    cluster_profiles = {}
    for c in range(n_clusters):
        sub = work[work["cluster"] == c]
        cluster_profiles[c] = {
            "goals_per90": sub["goals_per90"].mean(),
            "assists_per90": sub["assists_per90"].mean(),
            "cards_per90": sub["cards_per90"].mean(),
            "age": sub["age"].mean(),
            "size": len(sub),
        }

    # Assign descriptive labels based on profile
    labels = _assign_labels(cluster_profiles)
    work["style"] = work["cluster"].map(labels)

    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=CLUSTER_FEATURES
    )
    centroids["style"] = [labels[i] for i in range(n_clusters)]
    centroids["size"] = [cluster_profiles[i]["size"] for i in range(n_clusters)]

    return work, centroids


def _assign_labels(profiles):
    """Heuristic labelling of clusters based on stat profiles."""
    labels = {}
    used = set()
    # Sort clusters by goals to find finishers
    by_goals = sorted(profiles.items(), key=lambda x: -x[1]["goals_per90"])
    by_assists = sorted(profiles.items(), key=lambda x: -x[1]["assists_per90"])
    by_cards = sorted(profiles.items(), key=lambda x: -x[1]["cards_per90"])
    by_age = sorted(profiles.items(), key=lambda x: -x[1]["age"])

    candidates = [
        (by_goals[0][0], "Prolific Finisher"),
        (by_assists[0][0], "Creative Playmaker"),
        (by_cards[0][0], "Defensive Anchor"),
        (by_age[0][0], "Veteran Leader"),
    ]
    for cid, label in candidates:
        if cid not in used:
            labels[cid] = label
            used.add(cid)

    # Remaining clusters
    generic = ["Box-to-Box Engine", "Squad Rotation", "Wide Threat", "Utility Player", "Emerging Talent"]
    gi = 0
    for cid in profiles:
        if cid not in labels:
            labels[cid] = generic[gi % len(generic)]
            gi += 1
    return labels


def nation_summary(df):
    """Aggregate stats per WC nation."""
    summary = df.groupby("wc_nation").agg(
        squad_size=("player_id", "count"),
        avg_age=("age", "mean"),
        total_value=("market_value_in_eur", "sum"),
        avg_value=("market_value_in_eur", "mean"),
        total_goals=("goals", "sum"),
        total_assists=("assists", "sum"),
        avg_height=("height_in_cm", "mean"),
    ).reset_index()
    summary["total_value_m"] = (summary["total_value"] / 1_000_000).round(1)
    summary["avg_value_m"] = (summary["avg_value"] / 1_000_000).round(2)
    summary["avg_age"] = summary["avg_age"].round(1)
    summary["avg_height"] = summary["avg_height"].round(0)
    return summary.sort_values("total_value_m", ascending=False)


def top_players(df, metric="market_value_in_eur", n=20, position=None, nation=None):
    work = df.copy()
    if position and position != "All":
        work = work[work["position"] == position]
    if nation and nation != "All":
        work = work[work["wc_nation"] == nation]
    return work.nlargest(n, metric)
