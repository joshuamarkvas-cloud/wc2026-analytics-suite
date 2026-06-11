"""
Preprocessing: builds a compact wc_players.csv from the large Transfermarkt files.
Run once: python3 build_player_data.py
Produces wc_players.csv (~few MB) so the app loads fast without the 1.8M-row appearances file.
"""
import pandas as pd
import numpy as np
import os

UPLOAD_DIR = os.environ.get("TM_DIR", ".")

# WC 2026 citizenship → canonical team mapping
CITIZENSHIP_MAP = {
    "Mexico":"Mexico","South Africa":"South Africa","Korea, South":"South Korea","South Korea":"South Korea",
    "Czech Republic":"Czech Republic","Czechia":"Czech Republic","Canada":"Canada",
    "Bosnia-Herzegovina":"Bosnia and Herzegovina","Qatar":"Qatar","Switzerland":"Switzerland",
    "Brazil":"Brazil","Morocco":"Morocco","Haiti":"Haiti","Scotland":"Scotland",
    "United States":"United States","USA":"United States","Paraguay":"Paraguay","Australia":"Australia",
    "Turkey":"Turkey","Türkiye":"Turkey","Germany":"Germany","Curacao":"Curacao","Curaçao":"Curacao",
    "Cote d'Ivoire":"Cote d Ivoire","Ivory Coast":"Cote d Ivoire","Côte d'Ivoire":"Cote d Ivoire",
    "Ecuador":"Ecuador","Netherlands":"Netherlands","Japan":"Japan","Sweden":"Sweden","Tunisia":"Tunisia",
    "Belgium":"Belgium","Egypt":"Egypt","Iran":"Iran","New Zealand":"New Zealand","Spain":"Spain",
    "Cape Verde":"Cape Verde","Cabo Verde":"Cape Verde","Saudi Arabia":"Saudi Arabia","Uruguay":"Uruguay",
    "France":"France","Senegal":"Senegal","Iraq":"Iraq","Norway":"Norway","Argentina":"Argentina",
    "Algeria":"Algeria","Austria":"Austria","Jordan":"Jordan","Portugal":"Portugal",
    "DR Congo":"DR Congo","Congo":"DR Congo","Congo DR":"DR Congo","Uzbekistan":"Uzbekistan",
    "Colombia":"Colombia","England":"England","Croatia":"Croatia","Ghana":"Ghana","Panama":"Panama",
}

def build(upload_dir=UPLOAD_DIR, out_path="wc_players.csv"):
    players = pd.read_csv(os.path.join(upload_dir, "players.csv"))
    apps    = pd.read_csv(os.path.join(upload_dir, "appearances.csv"))

    # Map to WC nation
    players["wc_nation"] = players["country_of_citizenship"].map(CITIZENSHIP_MAP)
    players = players[players["wc_nation"].notna()].copy()

    # Keep players active recently (last_season >= 2023) for relevance
    players = players[players["last_season"] >= 2023].copy()

    # Aggregate recent appearance stats (2023+)
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    recent = apps[apps["date"] >= "2023-01-01"]
    agg = recent.groupby("player_id").agg(
        apps=("game_id", "count"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        minutes=("minutes_played", "sum"),
        yellows=("yellow_cards", "sum"),
        reds=("red_cards", "sum"),
    ).reset_index()

    df = players.merge(agg, on="player_id", how="inner")

    # Only players with meaningful minutes
    df = df[df["minutes"] >= 450].copy()  # ~5 full matches

    # Per-90 metrics
    df["mins_per_app"]   = df["minutes"] / df["apps"].replace(0, 1)
    df["goals_per90"]    = df["goals"]   / (df["minutes"] / 90)
    df["assists_per90"]  = df["assists"] / (df["minutes"] / 90)
    df["ga_per90"]       = df["goals_per90"] + df["assists_per90"]
    df["cards_per90"]    = (df["yellows"] + df["reds"]) / (df["minutes"] / 90)

    # Age
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = ((pd.Timestamp("2026-06-11") - df["date_of_birth"]).dt.days / 365.25).round(1)

    # Select compact columns
    keep = [
        "player_id","name","wc_nation","position","sub_position","foot",
        "height_in_cm","age","market_value_in_eur","highest_market_value_in_eur",
        "international_caps","international_goals","current_club_name",
        "apps","goals","assists","minutes","yellows","reds",
        "goals_per90","assists_per90","ga_per90","cards_per90","mins_per_app",
    ]
    out = df[keep].copy()
    out = out[out["position"] != "Missing"]
    out.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path}: {len(out):,} players across {out['wc_nation'].nunique()} WC nations")
    return out

if __name__ == "__main__":
    build()
