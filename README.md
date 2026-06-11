# ⚽ FIFA World Cup 2026 Analytics Suite

Full match-prediction + player-performance dashboard for the 48-team 2026 World Cup.

## Tech Stack
Python · Pandas · Scikit-learn (RandomForest, LogisticRegression, KMeans) · Streamlit · Plotly · Seaborn

## Sections (left sidebar menu)
- **Home** — overview dashboard
- **Match & Score Predictor** — ML win/draw/loss + Poisson scoreline
- **Group Simulator** — all 72 group matches → Round of 32
- **Monte Carlo** — full tournament title odds
- **Bracket** — interactive knockout bracket
- **Form & H2H** — recent form + head-to-head history
- **Elo Rankings** — real eloratings.net ratings
- **Upsets & Dark Horses** — upset detector + underrated teams
- **WC Top Scorers** — all-time WC goal leaders
- **Player Dashboard** — Transfermarkt stats, squad values, performance
- **Player Style Clustering** — KMeans clusters players by playing style
- **All 48 Teams** — full team database

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Required data files (place in same folder as app.py)
From Kaggle martj42/international-football-results-from-1872-to-2017:
- results.csv, goalscorers.csv, shootouts.csv

For the player dashboard, build the compact player file once:
```bash
# Put the Transfermarkt CSVs (players.csv, appearances.csv) in this folder, then:
python3 build_player_data.py
```
This creates `wc_players.csv` (~1.2MB) used by the app. You only do this once.

### 3. Run
```bash
streamlit run app.py
```

## Files
- `app.py` — Streamlit UI with sidebar navigation
- `data_model.py` — match prediction models + tournament simulation
- `player_model.py` — player stats + KMeans clustering
- `build_player_data.py` — one-time preprocessing for Transfermarkt data
- `wc_players.csv` — compact player dataset (generated)
- `results.csv`, `goalscorers.csv`, `shootouts.csv` — match history
