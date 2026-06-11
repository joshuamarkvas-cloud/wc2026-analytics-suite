"""
⚽ FIFA World Cup 2026 — Full Analytics Suite
Match prediction + Player performance dashboard with KMeans clustering
Left sidebar navigation menu
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="WC 2026 Analytics", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

# ───────────────────────── CSS ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;600;700&display=swap');
:root{--gold:#F5C518;--green:#166534;--dark:#060A10;--panel:#0E1621;--panel2:#131E2E;--border:#1E2D3D;--text:#E8EFF7;--muted:#64748B;--blue:#3B82F6;--red:#EF4444;}
.main,.stApp{background:var(--dark)!important;color:var(--text);}
.block-container{padding-top:1.2rem;max-width:1450px;}
.hero{background:linear-gradient(120deg,#0A1A0A,#0D2B14 40%,#091525);border:1px solid #1B3A1B;border-radius:14px;padding:1.5rem 2.2rem;margin-bottom:1.3rem;position:relative;overflow:hidden;}
.hero::after{content:"🏆";position:absolute;right:2.2rem;top:50%;transform:translateY(-50%);font-size:5rem;opacity:.08;}
.hero h1{font-family:'Bebas Neue',sans-serif;font-size:2.5rem;letter-spacing:4px;color:var(--gold);margin:0 0 .2rem;}
.hero p{color:#6FCF97;font-size:.92rem;margin:0;font-weight:300;}
.badge{display:inline-block;background:#0D2B14;border:1px solid #2D6B3A;color:#6FCF97;border-radius:20px;font-size:.73rem;padding:.2rem .8rem;margin-top:.5rem;margin-right:4px;}
.kpi-row{display:flex;gap:10px;margin-bottom:1rem;}
.kpi{flex:1;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:.9rem 1rem;text-align:center;}
.kpi .val{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:var(--gold);line-height:1;}
.kpi .lbl{font-size:.67rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:.2rem;}
.team-card{background:var(--panel2);border:1px solid var(--border);border-radius:12px;padding:1.2rem;}
.team-card h4{font-family:'Bebas Neue',sans-serif;letter-spacing:1.5px;font-size:1.1rem;margin:0 0 .8rem;}
.trow{display:flex;justify-content:space-between;padding:.25rem 0;border-bottom:1px solid #182435;font-size:.81rem;}
.trow:last-child{border:none;}.lc{color:var(--muted);}.vc{font-weight:600;}
.elo-bar{height:7px;border-radius:4px;background:linear-gradient(90deg,var(--green),var(--gold));margin-top:3px;}
.result{border-radius:12px;padding:1.3rem 2rem;text-align:center;margin:1rem 0;}
.result h2{font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:3px;margin:0 0 .2rem;}
.result p{font-size:.85rem;color:var(--muted);margin:0;}
.result-home{background:linear-gradient(135deg,#052505,#0D3A0D);border:2px solid #22C55E;}
.result-away{background:linear-gradient(135deg,#050522,#0D0D3A);border:2px solid var(--blue);}
.result-draw{background:linear-gradient(135deg,#1A1200,#3A2800);border:2px solid var(--gold);}
.prob-wrap{margin:.4rem 0;}
.prob-label{font-size:.78rem;color:var(--muted);margin-bottom:3px;display:flex;justify-content:space-between;}
.bar-bg{background:#131E2E;border-radius:6px;height:20px;overflow:hidden;}
.bar{height:100%;border-radius:6px;}
.bar-h{background:linear-gradient(90deg,#166534,#22C55E);}
.bar-d{background:linear-gradient(90deg,#78350F,var(--gold));}
.bar-a{background:linear-gradient(90deg,#1E3A8A,var(--blue));}
.gt{width:100%;border-collapse:collapse;font-size:.81rem;}
.gt th{color:var(--muted);text-transform:uppercase;font-size:.68rem;letter-spacing:.8px;padding:.3rem .4rem;border-bottom:1px solid var(--border);text-align:center;}
.gt td{padding:.3rem .4rem;border-bottom:1px solid #0E1621;text-align:center;}
.gt tr:last-child td{border:none;}.gt .tn{text-align:left;font-weight:600;}
.qualify{background:#0D2B14;}.third{background:#1C1A00;}.elim{opacity:.55;}
section[data-testid="stSidebar"]{background:var(--panel)!important;border-right:1px solid var(--border)!important;}
.sec-hdr{font-family:'Bebas Neue',sans-serif;font-size:1.3rem;letter-spacing:2px;color:var(--gold);border-left:3px solid var(--gold);padding-left:.8rem;margin:1.1rem 0 .7rem;}
.info{background:#0E1621;border-left:3px solid var(--gold);border-radius:0 8px 8px 0;padding:.7rem 1rem;font-size:.81rem;color:var(--muted);margin:.7rem 0;}
.upset-badge{background:#3A0D0D;border:1px solid var(--red);color:#FCA5A5;border-radius:6px;padding:.15rem .5rem;font-size:.72rem;font-family:'Bebas Neue',sans-serif;letter-spacing:1px;}
.form-W{color:#22C55E;font-weight:700;}.form-D{color:var(--gold);font-weight:700;}.form-L{color:var(--red);font-weight:700;}
.bracket-match{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:.5rem .8rem;margin:.2rem 0;font-size:.82rem;}
.bracket-winner{border-color:var(--gold);color:var(--gold);font-weight:700;}
.nav-title{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:2px;color:var(--gold);margin:.3rem 0 .5rem;}
.player-card{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:1rem;margin-bottom:.6rem;}
hr{border-color:var(--border)!important;}
.stRadio [role="radiogroup"]{gap:2px;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH     = os.path.join(BASE_DIR, "results.csv")
GOALSCORERS_PATH = os.path.join(BASE_DIR, "goalscorers.csv")
SHOOTOUTS_PATH   = os.path.join(BASE_DIR, "shootouts.csv")
PLAYERS_PATH     = os.path.join(BASE_DIR, "wc_players.csv")

@st.cache_resource(show_spinner=False)
def load_model():
    from data_model import train_models
    return train_models(results_path=RESULTS_PATH, goalscorers_path=GOALSCORERS_PATH, shootouts_path=SHOOTOUTS_PATH)

@st.cache_data(show_spinner=False)
def load_player_df():
    if not os.path.exists(PLAYERS_PATH):
        return None
    from player_model import load_players
    return load_players(PLAYERS_PATH)

from data_model import (ALL_TEAMS, WC2026_GROUPS, TEAM_GROUP, TEAM_STATS, ELO_RATINGS,
    predict_match, predict_scoreline, simulate_group_stage, run_monte_carlo,
    get_team_form_history, get_h2h_history, get_top_wc_scorers, _h2h)

sorted_teams = sorted(ALL_TEAMS)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,30,46,0.5)",
    font=dict(color="#E8EFF7", family="DM Sans"),
    colorway=["#22C55E","#F5C518","#3B82F6","#EF4444","#A855F7","#06B6D4","#F97316","#EC4899"],
    margin=dict(l=10,r=10,t=40,b=10),
)

# ───────────────────────── SIDEBAR NAV ─────────────────────────
with st.sidebar:
    st.markdown('<div class="nav-title">⚽ WC 2026</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.75rem;color:#64748B;margin-bottom:.8rem;">Analytics Suite</div>', unsafe_allow_html=True)

    section = st.radio("Navigation", [
        "🏠 Home",
        "🎯 Match & Score Predictor",
        "📊 Group Simulator",
        "🏆 Monte Carlo",
        "🗺️ Bracket",
        "📉 Form & H2H",
        "🏅 Elo Rankings",
        "⚠️ Upsets & Dark Horses",
        "⚽ WC Top Scorers",
        "👥 Player Dashboard",
        "🎨 Player Style Clustering",
        "🗂️ All 48 Teams",
    ], label_visibility="collapsed")

    st.markdown("---")
    with st.spinner("Loading model…"):
        model_data = load_model()
    st.caption(f"**Model:** {model_data['model_name']}")
    st.caption(f"**Accuracy:** {model_data['test_acc']*100:.1f}%")
    st.caption(f"**Real matches:** {model_data['real_matches_used']:,}")
    st.markdown("---")
    st.markdown("📡 [eloratings.net](https://eloratings.net)")
    st.markdown("📊 [Match Data](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)")
    st.markdown("👤 [Player Data](https://www.kaggle.com/datasets/davidcariboo/player-scores)")

# Helpers
def elo_pct(t): return int((ELO_RATINGS.get(t,1600)-1400)/(2200-1400)*100)

def team_card(team, color, icon):
    s=TEAM_STATS[team]; g=TEAM_GROUP.get(team,"?"); elo=ELO_RATINGS.get(team,1600)
    titles="🏆"*s["wc_titles"] if s["wc_titles"]>0 else "—"
    return f"""<div class="team-card">
      <h4 style="color:{color};">{icon} {team} <span style="font-size:.7rem;background:#0D2B14;border:1px solid #2D6B3A;color:#6FCF97;border-radius:4px;padding:.1rem .4rem;">GROUP {g}</span></h4>
      <div class="trow"><span class="lc">⚡ Elo</span><span class="vc" style="color:var(--gold)">{elo}</span></div>
      <div style="margin:.15rem 0 .4rem;"><div class="elo-bar" style="width:{elo_pct(team)}%"></div></div>
      <div class="trow"><span class="lc">Attack</span><span class="vc">{s['attack']}</span></div>
      <div class="trow"><span class="lc">Defense</span><span class="vc">{s['defense']}</span></div>
      <div class="trow"><span class="lc">WC Titles</span><span class="vc">{titles}</span></div>
      <div class="trow"><span class="lc">WC Apps</span><span class="vc">{s['wc_apps']}</span></div>
      <div class="trow"><span class="lc">Avg Goals</span><span class="vc">{s['gs']} / {s['gc']}</span></div>
      <div class="trow"><span class="lc">Confederation</span><span class="vc">{s['confederation']}</span></div>
    </div>"""

FEAT_LABELS={"elo_diff":"Elo Difference","elo_home":"Home Elo","elo_away":"Away Elo","attack_diff":"Attack Edge","defense_diff":"Defense Edge","title_diff":"Title Pedigree","experience_diff":"WC Experience","goal_diff":"Avg Goals Edge","defense_adv":"Defensive Resilience","h2h_win_pct":"H2H Win %","h2h_goal_diff_avg":"H2H Goal Diff","h2h_matches":"H2H Matches","home_form_pts":"Home Form","away_form_pts":"Away Form","home_goals_per_game":"Home Goals/G","away_goals_per_game":"Away Goals/G","home_conceded_per_game":"Home Conceded/G","away_conceded_per_game":"Away Conceded/G","home_penalty_win_pct":"Home Pen Win %","away_penalty_win_pct":"Away Pen Win %","home_wc_goals_per_game":"Home WC Goals/G","away_wc_goals_per_game":"Away WC Goals/G"}

# ═══════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════
if section == "🏠 Home":
    st.markdown("""<div class="hero"><h1>⚽ FIFA WORLD CUP 2026 ANALYTICS SUITE</h1>
    <p>Match prediction · Tournament simulation · Player performance & style clustering</p>
    <div style="margin-top:.5rem;"><span class="badge">🇺🇸🇲🇽🇨🇦 North America</span><span class="badge">48 Teams · 12 Groups</span><span class="badge">5,800+ Players</span></div>
    </div>""", unsafe_allow_html=True)

    player_df = load_player_df()
    st.markdown('<div class="kpi-row">' + "".join(f'<div class="kpi"><div class="val">{v}</div><div class="lbl">{l}</div></div>' for v,l in [
        ("48","Teams"),("12","Groups"),(f"{model_data['real_matches_used']:,}","Historical Matches"),
        (f"{len(player_df):,}" if player_df is not None else "—","Players Tracked"),
        (f"{model_data['test_acc']*100:.0f}%","Model Accuracy")]) + "</div>",unsafe_allow_html=True)

    st.markdown("<div class='sec-hdr'>What's Inside</div>",unsafe_allow_html=True)
    cols=st.columns(3)
    features=[
        ("🎯 Match & Score Predictor","ML win/draw/loss probabilities + Poisson scoreline prediction for any matchup"),
        ("📊 Group Simulator","Simulate all 72 group matches, see who advances to the Round of 32"),
        ("🏆 Monte Carlo","Run the full tournament thousands of times for title-winning odds"),
        ("🗺️ Bracket","Build your own knockout bracket pick by pick"),
        ("📉 Form & H2H","Recent form strips and full head-to-head history"),
        ("🏅 Elo Rankings","Live Elo ratings for all 48 teams from eloratings.net"),
        ("⚠️ Upsets & Dark Horses","Spot likely upsets and underrated teams"),
        ("⚽ WC Top Scorers","All-time World Cup goal leaders from match data"),
        ("👥 Player Dashboard","Transfermarkt player stats, squad values, top performers"),
        ("🎨 Style Clustering","KMeans clusters players by playing style"),
    ]
    for i,(title,desc) in enumerate(features):
        with cols[i%3]:
            st.markdown(f'<div class="player-card"><div style="font-family:Bebas Neue,sans-serif;font-size:1.05rem;letter-spacing:1px;color:var(--gold);">{title}</div><div style="font-size:.8rem;color:var(--muted);margin-top:.3rem;">{desc}</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="info">👈 Use the menu on the left to explore each section. All data is embedded — no uploads needed.</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MATCH & SCORE PREDICTOR (combined)
# ═══════════════════════════════════════════════════════════════
elif section == "🎯 Match & Score Predictor":
    st.markdown("<div class='sec-hdr'>Match & Score Predictor</div>",unsafe_allow_html=True)
    c1,c2,c3 = st.columns([5,1,5])
    with c1: home_team = st.selectbox("🏠 Home Team", sorted_teams, index=sorted_teams.index("Brazil"))
    with c2: st.markdown("<div style='height:60px;display:flex;align-items:center;justify-content:center;font-family:Bebas Neue,sans-serif;font-size:2.5rem;color:#F5C518;'>VS</div>",unsafe_allow_html=True)
    with c3: away_team = st.selectbox("✈️ Away Team", sorted_teams, index=sorted_teams.index("Germany"))

    if home_team == away_team:
        st.warning("⚠️ Select two different teams."); st.stop()

    ch,cv,ca=st.columns([5,1,5])
    with ch: st.markdown(team_card(home_team,"#22C55E","🏠"),unsafe_allow_html=True)
    with cv: st.markdown("<div style='height:180px;display:flex;align-items:center;justify-content:center;font-family:Bebas Neue,sans-serif;font-size:2rem;color:#F5C518;'>VS</div>",unsafe_allow_html=True)
    with ca: st.markdown(team_card(away_team,"#3B82F6","✈️"),unsafe_allow_html=True)

    result=predict_match(model_data,home_team,away_team)
    pred=result["prediction"]; h_pct=result["home_win"]*100; d_pct=result["draw"]*100; a_pct=result["away_win"]*100
    elo_gap=result["home_elo"]-result["away_elo"]

    st.markdown("<div class='sec-hdr'>Prediction</div>",unsafe_allow_html=True)
    if pred=="Home Win": cls,icon,hl="result-home","🏠",f"{home_team} WIN"; sub=f"{h_pct:.0f}% · Elo gap {elo_gap:+d}"
    elif pred=="Away Win": cls,icon,hl="result-away","✈️",f"{away_team} WIN"; sub=f"{a_pct:.0f}% · Elo gap {elo_gap:+d}"
    else: cls,icon,hl="result-draw","🤝","DRAW"; sub=f"Draw {d_pct:.0f}% · Elo gap {elo_gap:+d}"
    st.markdown(f'<div class="result {cls}"><h2>{icon} {hl}</h2><p>{sub}</p></div>',unsafe_allow_html=True)

    fav=home_team if ELO_RATINGS.get(home_team,1600)>=ELO_RATINGS.get(away_team,1600) else away_team
    dog=away_team if fav==home_team else home_team
    dog_pct=a_pct if fav==home_team else h_pct
    if dog_pct>=35:
        st.markdown(f'<div style="margin:.5rem 0;">⚠️ <span class="upset-badge">UPSET ALERT</span> {dog} has a <strong>{dog_pct:.0f}%</strong> chance vs {fav}!</div>',unsafe_allow_html=True)

    # Plotly donut for outcome
    pcol1, pcol2 = st.columns([1,1])
    with pcol1:
        fig=go.Figure(data=[go.Pie(labels=[f"{home_team} Win","Draw",f"{away_team} Win"],values=[h_pct,d_pct,a_pct],hole=.55,marker=dict(colors=["#22C55E","#F5C518","#3B82F6"]))])
        fig.update_layout(title="Outcome Probability",**PLOTLY_LAYOUT,height=320,showlegend=True,legend=dict(orientation="h",y=-0.1))
        st.plotly_chart(fig,use_container_width=True)
    with pcol2:
        scores,h_lam,a_lam=predict_scoreline(model_data,home_team,away_team)
        st.markdown(f'<div class="kpi-row"><div class="kpi"><div class="val">{h_lam:.2f}</div><div class="lbl">{home_team} xG</div></div><div class="kpi"><div class="val">{a_lam:.2f}</div><div class="lbl">{away_team} xG</div></div></div>',unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;margin:.5rem 0;"><div style="font-size:.78rem;color:var(--muted);">Most Likely Score</div><div style="font-family:Bebas Neue,sans-serif;font-size:3rem;color:var(--gold);">{scores[0]["hg"]}–{scores[0]["ag"]}</div><div style="font-size:.78rem;color:var(--muted);">{scores[0]["prob"]*100:.1f}% probability</div></div>',unsafe_allow_html=True)

    st.markdown("<div class='sec-hdr'>Top Scorelines (Poisson Model)</div>",unsafe_allow_html=True)
    sc_cols=st.columns(5)
    for i,s in enumerate(scores[:10]):
        with sc_cols[i%5]:
            st.markdown(f'<div style="background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:.6rem;text-align:center;margin:.2rem 0;"><div style="font-family:Bebas Neue,sans-serif;font-size:1.6rem;color:var(--gold);">{s["hg"]}–{s["ag"]}</div><div style="font-size:.72rem;color:var(--muted);">{s["prob"]*100:.1f}%</div></div>',unsafe_allow_html=True)

    if model_data.get("df") is not None:
        wp,gd,cnt=_h2h(model_data["df"],home_team,away_team)
        if cnt>0:
            st.markdown(f'<div class="info">📚 <strong>H2H ({cnt} matches):</strong> {home_team} win rate <strong>{wp*100:.0f}%</strong> · Avg goal diff <strong>{gd:+.2f}</strong></div>',unsafe_allow_html=True)

    st.markdown("<div class='sec-hdr'>Feature Breakdown</div>",unsafe_allow_html=True)
    feat_df=pd.DataFrame([{"Feature":FEAT_LABELS.get(k,k),"Value":round(v,3),"Favours":f"🏠 {home_team}" if v>0.01 else (f"✈️ {away_team}" if v<-0.01 else "Neutral")} for k,v in result["features"].items()])
    st.dataframe(feat_df,use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════
# GROUP SIMULATOR
# ═══════════════════════════════════════════════════════════════
elif section == "📊 Group Simulator":
    st.markdown("<div class='sec-hdr'>2026 Group Stage Simulator</div>",unsafe_allow_html=True)
    god=max(WC2026_GROUPS.items(),key=lambda x:np.mean([ELO_RATINGS.get(t,1600) for t in x[1]]))
    god_avg=np.mean([ELO_RATINGS.get(t,1600) for t in god[1]])
    st.markdown(f'<div class="info">🔥 <strong>Group of Death: Group {god[0]}</strong> ({", ".join(god[1])}) — avg Elo {god_avg:.0f}</div>',unsafe_allow_html=True)

    if st.button("🔄 Simulate All 12 Groups",type="primary",use_container_width=True):
        with st.spinner("Simulating 72 matches…"):
            standings=simulate_group_stage(model_data)
        thirds=[]
        for g,data in standings.items():
            t,s=data["table"][2]; thirds.append((t,s,g))
        thirds.sort(key=lambda x:(-x[1]["Pts"],-x[1]["GD"],-x[1]["GF"]))
        adv_thirds=set(t for t,_,_ in thirds[:8])

        for row_start in range(0,12,3):
            cols=st.columns(3)
            for ci,(group,data) in enumerate(list(standings.items())[row_start:row_start+3]):
                with cols[ci]:
                    html=f"<div style='background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:.9rem;margin-bottom:.8rem;'>"
                    html+=f"<div style='font-family:Bebas Neue,sans-serif;font-size:1rem;letter-spacing:2px;color:var(--gold);margin-bottom:.5rem;'>GROUP {group}</div>"
                    html+="<table class='gt'><tr><th style='text-align:left;'>Team</th><th>Elo</th><th>Pts</th><th>GD</th></tr>"
                    for rank,(team,stats) in enumerate(data["table"]):
                        badge="✅" if rank<2 else ("3️⃣" if team in adv_thirds else "❌")
                        rc="qualify" if rank<2 else ("third" if team in adv_thirds else "elim")
                        gds=f"+{stats['GD']}" if stats['GD']>0 else str(stats['GD'])
                        html+=f"<tr class='{rc}'><td class='tn'>{badge} {team}</td><td>{ELO_RATINGS.get(team,1600)}</td><td><strong>{stats['Pts']}</strong></td><td>{gds}</td></tr>"
                    html+="</table><div style='margin-top:.5rem;font-size:.73rem;color:var(--muted);'>"
                    for m in data["matches"]: html+=f"<div>{m['home']} <strong>{m['hg']}–{m['ag']}</strong> {m['away']}</div>"
                    html+="</div></div>"
                    st.markdown(html,unsafe_allow_html=True)

        st.markdown("<div class='sec-hdr'>Best 3rd-Place Rankings</div>",unsafe_allow_html=True)
        tp_df=pd.DataFrame([{"Rank":i+1,"Team":t,"Group":g,"Elo":ELO_RATINGS.get(t,1600),"Pts":s["Pts"],"GD":s["GD"],"GF":s["GF"],"Advances":"✅" if t in adv_thirds else "❌"} for i,(t,s,g) in enumerate(thirds)])
        st.dataframe(tp_df,use_container_width=True,hide_index=True)

        st.markdown("<div class='sec-hdr'>Group Strength (avg Elo)</div>",unsafe_allow_html=True)
        grp_df=pd.DataFrame([{"Group":f"Group {g}","Avg Elo":round(np.mean([ELO_RATINGS.get(t,1600) for t in teams]),0)} for g,teams in WC2026_GROUPS.items()]).sort_values("Avg Elo",ascending=True)
        fig=px.bar(grp_df,x="Avg Elo",y="Group",orientation="h",color="Avg Elo",color_continuous_scale=["#166534","#F5C518"])
        fig.update_layout(**PLOTLY_LAYOUT,height=400)
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.markdown("<div style='text-align:center;padding:3rem;color:var(--muted);'><div style='font-size:3rem;'>⚽</div><div style='font-family:Bebas Neue,sans-serif;font-size:1.4rem;letter-spacing:2px;margin-top:.5rem;'>Click to simulate all 72 group stage matches</div></div>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════════════
elif section == "🏆 Monte Carlo":
    st.markdown("<div class='sec-hdr'>Tournament Winner — Monte Carlo</div>",unsafe_allow_html=True)
    st.markdown('<div class="info">Simulates the full tournament thousands of times for each team\'s title odds.</div>',unsafe_allow_html=True)
    n_sims=st.slider("Simulations",500,5000,2000,step=500)
    if st.button("🎲 Run Monte Carlo",type="primary",use_container_width=True):
        with st.spinner(f"Running {n_sims:,} simulations…"):
            mc=run_monte_carlo(model_data,n_sims)
        mc_df=pd.DataFrame([{"Team":t,"Group":f"Group {TEAM_GROUP[t]}","Elo":ELO_RATINGS.get(t,1600),"🏆 Win %":mc[t]["win_pct"],"🥈 Final %":mc[t]["final_pct"],"R32 %":mc[t]["r32_pct"]} for t in ALL_TEAMS])
        mc_df=mc_df.sort_values("🏆 Win %",ascending=False).reset_index(drop=True)
        mc_df.insert(0,"Rank",range(1,len(mc_df)+1))
        st.dataframe(mc_df,use_container_width=True,hide_index=True,height=450)
        top16=mc_df.head(16)
        fig=px.bar(top16,x="🏆 Win %",y="Team",orientation="h",color="🏆 Win %",color_continuous_scale=["#166534","#F5C518"])
        fig.update_layout(**PLOTLY_LAYOUT,height=500,yaxis=dict(autorange="reversed"),title="Top 16 — Title Probability")
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# BRACKET
# ═══════════════════════════════════════════════════════════════
elif section == "🗺️ Bracket":
    st.markdown("<div class='sec-hdr'>Interactive Tournament Bracket</div>",unsafe_allow_html=True)
    st.markdown('<div class="info">Pick winners through each round. Starts from R32 (top 2 per group + 8 best 3rd places, seeded by Elo).</div>',unsafe_allow_html=True)
    if "bracket_r32" not in st.session_state:
        r32=[]
        for g,teams in WC2026_GROUPS.items():
            by_elo=sorted(teams,key=lambda t:-ELO_RATINGS.get(t,1600))
            r32+= [by_elo[0],by_elo[1]]
        thirds=sorted([sorted(teams,key=lambda t:-ELO_RATINGS.get(t,1600))[2] for teams in WC2026_GROUPS.values()],key=lambda t:-ELO_RATINGS.get(t,1600))
        r32+=thirds[:8]
        st.session_state.bracket_r32=r32[:32]
        st.session_state.bracket_history={}
    rounds={"R32":"Round of 32","R16":"Round of 16","QF":"Quarter Finals","SF":"Semi Finals","F":"Final"}
    current=st.session_state.bracket_r32
    for rk,rn in rounds.items():
        st.markdown(f"<div class='sec-hdr'>{rn}</div>",unsafe_allow_html=True)
        if rk not in st.session_state.bracket_history:
            pairs=[(current[i],current[i+1]) for i in range(0,len(current)-1,2)]
            winners=[]
            cols=st.columns(min(4,max(1,len(pairs))))
            for pi,(ta,tb) in enumerate(pairs):
                with cols[pi%len(cols)]:
                    r=predict_match(model_data,ta,tb)
                    fav=ta if r["home_win"]>=r["away_win"] else tb
                    choice=st.selectbox(f"{ta} vs {tb}",[ta,tb],index=[ta,tb].index(fav),key=f"br_{rk}_{pi}")
                    conf=r["home_win"] if choice==ta else r["away_win"]
                    st.caption(f"Confidence: {conf*100:.0f}%")
                    winners.append(choice)
            if st.button("Advance →",key=f"adv_{rk}",type="primary"):
                st.session_state.bracket_history[rk]=winners; st.rerun()
            break
        else:
            winners=st.session_state.bracket_history[rk]
            pairs=[(current[i],current[i+1]) for i in range(0,len(current)-1,2)]
            html="<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
            for (ta,tb),w in zip(pairs,winners):
                html+=f"<div class='bracket-match'>{ta} v {tb} → <strong style='color:var(--gold)'>{w}</strong></div>"
            html+="</div>"; st.markdown(html,unsafe_allow_html=True)
            current=winners
            if rk=="F" and winners:
                st.markdown(f"<div style='text-align:center;font-family:Bebas Neue,sans-serif;font-size:2.5rem;color:var(--gold);margin:1rem;'>🏆 CHAMPION: {winners[0]} 🏆</div>",unsafe_allow_html=True)
    if st.button("🔄 Reset Bracket"):
        for k in list(st.session_state.keys()):
            if k.startswith("bracket"): del st.session_state[k]
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# FORM & H2H
# ═══════════════════════════════════════════════════════════════
elif section == "📉 Form & H2H":
    st.markdown("<div class='sec-hdr'>Form & Head-to-Head</div>",unsafe_allow_html=True)
    fc1,fc2,fc3=st.columns([5,1,5])
    with fc1: ft1=st.selectbox("Team 1",sorted_teams,index=sorted_teams.index("England"))
    with fc2: st.markdown("<div style='height:55px;display:flex;align-items:center;justify-content:center;font-family:Bebas Neue,sans-serif;font-size:1.5rem;color:#F5C518;'>VS</div>",unsafe_allow_html=True)
    with fc3: ft2=st.selectbox("Team 2",sorted_teams,index=sorted_teams.index("Argentina"))
    df=model_data.get("df")
    if ft1!=ft2 and df is not None:
        cf1,cf2=st.columns(2)
        for team,col in [(ft1,cf1),(ft2,cf2)]:
            with col:
                st.markdown(f"<div class='sec-hdr'>{team} — Last 15</div>",unsafe_allow_html=True)
                form=get_team_form_history(df,team,15)
                if form:
                    fs=" ".join(f'<span class="form-{r["result"]}">{r["result"]}</span>' for r in reversed(form))
                    st.markdown(f"<div style='font-size:1.1rem;letter-spacing:2px;margin-bottom:.5rem;'>{fs}</div>",unsafe_allow_html=True)
                    fdf=pd.DataFrame([{"Date":r["date"].strftime("%Y-%m-%d"),"Opp":r["opponent"] or "?","Score":f"{r['gf']}–{r['ga']}","R":r["result"]} for r in reversed(form)])
                    st.dataframe(fdf,use_container_width=True,hide_index=True,height=300)
        st.markdown("<div class='sec-hdr'>Head-to-Head History</div>",unsafe_allow_html=True)
        h2h=get_h2h_history(df,ft1,ft2)
        if h2h:
            wp,gd,cnt=_h2h(df,ft1,ft2)
            hw=sum(1 for r in h2h if r["result"]=="W"); hd=sum(1 for r in h2h if r["result"]=="D"); hl=sum(1 for r in h2h if r["result"]=="L")
            st.markdown(f'<div class="kpi-row"><div class="kpi"><div class="val">{cnt}</div><div class="lbl">Matches</div></div><div class="kpi"><div class="val">{hw}</div><div class="lbl">{ft1} W</div></div><div class="kpi"><div class="val">{hd}</div><div class="lbl">Draws</div></div><div class="kpi"><div class="val">{hl}</div><div class="lbl">{ft2} W</div></div></div>',unsafe_allow_html=True)
            # Pie of results
            fig=go.Figure(data=[go.Pie(labels=[f"{ft1} Wins","Draws",f"{ft2} Wins"],values=[hw,hd,hl],hole=.5,marker=dict(colors=["#22C55E","#F5C518","#3B82F6"]))])
            fig.update_layout(**PLOTLY_LAYOUT,height=300,title="H2H Record")
            st.plotly_chart(fig,use_container_width=True)
            h2h_df=pd.DataFrame([{"Date":r["date"].strftime("%Y-%m-%d"),ft1:r["home_score"],ft2:r["away_score"],"Result":r["result"],"Tournament":r["tournament"]} for r in reversed(h2h)])
            st.dataframe(h2h_df,use_container_width=True,hide_index=True,height=350)
        else:
            st.info(f"No H2H data between {ft1} and {ft2}.")

# ═══════════════════════════════════════════════════════════════
# ELO RANKINGS
# ═══════════════════════════════════════════════════════════════
elif section == "🏅 Elo Rankings":
    st.markdown("<div class='sec-hdr'>Elo Rankings — All 48 Teams</div>",unsafe_allow_html=True)
    st.markdown('<div class="info">Real Elo ratings from eloratings.net (June 2026).</div>',unsafe_allow_html=True)
    elo_df=pd.DataFrame([{"Team":t,"Group":f"Group {TEAM_GROUP[t]}","Elo":ELO_RATINGS.get(t,1600),"Confederation":TEAM_STATS[t]["confederation"]} for t in ALL_TEAMS]).sort_values("Elo",ascending=False).reset_index(drop=True)
    elo_df.insert(0,"Rank",range(1,len(elo_df)+1))
    c1,c2=st.columns([3,2])
    with c1:
        st.dataframe(elo_df,use_container_width=True,hide_index=True,height=600)
    with c2:
        fig=px.bar(elo_df.head(20).sort_values("Elo"),x="Elo",y="Team",orientation="h",color="Elo",color_continuous_scale=["#166534","#F5C518"])
        fig.update_layout(**PLOTLY_LAYOUT,height=600,title="Top 20")
        st.plotly_chart(fig,use_container_width=True)
    # Confederation avg
    st.markdown("<div class='sec-hdr'>Average Elo by Confederation</div>",unsafe_allow_html=True)
    conf_elo=elo_df.groupby("Confederation")["Elo"].mean().round(0).reset_index().sort_values("Elo",ascending=False)
    fig2=px.bar(conf_elo,x="Confederation",y="Elo",color="Elo",color_continuous_scale=["#166534","#F5C518"])
    fig2.update_layout(**PLOTLY_LAYOUT,height=350)
    st.plotly_chart(fig2,use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# UPSETS & DARK HORSES
# ═══════════════════════════════════════════════════════════════
elif section == "⚠️ Upsets & Dark Horses":
    st.markdown("<div class='sec-hdr'>⚠️ Upset Detector</div>",unsafe_allow_html=True)
    upsets=[]
    for g,teams in WC2026_GROUPS.items():
        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                h,a=teams[i],teams[j]; r=predict_match(model_data,h,a)
                fav=h if ELO_RATINGS.get(h,1600)>=ELO_RATINGS.get(a,1600) else a
                dog=a if fav==h else h
                dog_pct=(r["away_win"] if fav==h else r["home_win"])*100
                if dog_pct>=35:
                    upsets.append({"Group":f"Group {g}","Favourite":fav,"Underdog":dog,"Elo Gap":abs(ELO_RATINGS.get(h,1600)-ELO_RATINGS.get(a,1600)),"Upset %":round(dog_pct,1)})
    upsets.sort(key=lambda x:-x["Upset %"])
    if upsets: st.dataframe(pd.DataFrame(upsets),use_container_width=True,hide_index=True)
    else: st.info("No major upset candidates in group stage.")

    st.markdown("<div class='sec-hdr'>🌟 Dark Horses</div>",unsafe_allow_html=True)
    dh=[]
    for t in ALL_TEAMS:
        elo=ELO_RATINGS.get(t,1600)
        elo_rank=sorted(ALL_TEAMS,key=lambda x:-ELO_RATINGS.get(x,1600)).index(t)+1
        ped_rank=sorted(ALL_TEAMS,key=lambda x:-(TEAM_STATS[x]["wc_titles"]*5+TEAM_STATS[x]["wc_apps"])).index(t)+1
        gap=ped_rank-elo_rank
        if gap>=8 and elo>=1700:
            dh.append({"Team":t,"Group":f"Group {TEAM_GROUP[t]}","Elo":elo,"Elo Rank":elo_rank,"Pedigree Rank":ped_rank,"Underrated By":gap})
    dh.sort(key=lambda x:-x["Underrated By"])
    if dh: st.dataframe(pd.DataFrame(dh),use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════
# WC TOP SCORERS
# ═══════════════════════════════════════════════════════════════
elif section == "⚽ WC Top Scorers":
    st.markdown("<div class='sec-hdr'>World Cup All-Time Top Scorers</div>",unsafe_allow_html=True)
    df=model_data.get("df"); gs_df=model_data.get("gs_df")
    if df is not None and gs_df is not None:
        scorers=get_top_wc_scorers(df,gs_df,30)
        if scorers:
            sc_df=pd.DataFrame(scorers); sc_df.columns=["Player","Team","Goals","Penalties"]
            sc_df.insert(0,"Rank",range(1,len(sc_df)+1))
            c1,c2=st.columns([2,3])
            with c1: st.dataframe(sc_df,use_container_width=True,hide_index=True,height=600)
            with c2:
                fig=px.bar(sc_df.head(15).sort_values("Goals"),x="Goals",y="Player",orientation="h",color="Goals",color_continuous_scale=["#166534","#F5C518"])
                fig.update_layout(**PLOTLY_LAYOUT,height=600,title="Top 15 WC Scorers")
                st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("Add goalscorers.csv to the folder.")

# ═══════════════════════════════════════════════════════════════
# PLAYER DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif section == "👥 Player Dashboard":
    st.markdown("<div class='sec-hdr'>Player Performance Dashboard</div>",unsafe_allow_html=True)
    player_df=load_player_df()
    if player_df is None:
        st.error("⚠️ wc_players.csv not found. Run `python3 build_player_data.py` first (with the Transfermarkt CSVs in the folder).")
        st.stop()

    from player_model import nation_summary, top_players

    st.markdown('<div class="info">Player data from Transfermarkt (Kaggle). Stats aggregated from club appearances 2023–2026. Note: Qatar players not covered in dataset.</div>',unsafe_allow_html=True)

    # KPIs
    st.markdown('<div class="kpi-row">' + "".join(f'<div class="kpi"><div class="val">{v}</div><div class="lbl">{l}</div></div>' for v,l in [
        (f"{len(player_df):,}","Players"),(f"{player_df['wc_nation'].nunique()}","Nations"),
        (f"€{player_df['market_value_in_eur'].sum()/1e9:.1f}B","Total Value"),
        (f"{player_df['age'].mean():.1f}","Avg Age"),
        (f"{player_df['goals'].sum():,.0f}","Total Goals")]) + "</div>",unsafe_allow_html=True)

    # Filters
    fcol1,fcol2,fcol3=st.columns(3)
    with fcol1: pos_filter=st.selectbox("Position",["All","Attack","Midfield","Defender","Goalkeeper"])
    with fcol2: nation_filter=st.selectbox("Nation",["All"]+sorted(player_df["wc_nation"].unique()))
    with fcol3: metric=st.selectbox("Rank by",["market_value_in_eur","goals","assists","ga_per90","international_caps"],format_func=lambda x:{"market_value_in_eur":"Market Value","goals":"Goals","assists":"Assists","ga_per90":"Goals+Assists per 90","international_caps":"Intl Caps"}[x])

    # Top players table
    st.markdown("<div class='sec-hdr'>Top Players</div>",unsafe_allow_html=True)
    tp=top_players(player_df,metric,25,pos_filter,nation_filter)
    disp=tp[["name","wc_nation","position","sub_position","age","current_club_name","market_value_in_eur","goals","assists","international_caps"]].copy()
    disp["market_value_in_eur"]=(disp["market_value_in_eur"]/1e6).round(1)
    disp.columns=["Player","Nation","Position","Role","Age","Club","Value (€M)","Goals","Assists","Caps"]
    st.dataframe(disp,use_container_width=True,hide_index=True,height=450)

    # Squad value by nation
    st.markdown("<div class='sec-hdr'>Squad Value by Nation</div>",unsafe_allow_html=True)
    ns=nation_summary(player_df).head(25)
    fig=px.bar(ns.sort_values("total_value_m"),x="total_value_m",y="wc_nation",orientation="h",color="total_value_m",color_continuous_scale=["#166534","#F5C518"],labels={"total_value_m":"Total Value (€M)","wc_nation":""})
    fig.update_layout(**PLOTLY_LAYOUT,height=600)
    st.plotly_chart(fig,use_container_width=True)

    # Scatter: value vs goals
    st.markdown("<div class='sec-hdr'>Market Value vs Goal Output</div>",unsafe_allow_html=True)
    scatter_df=player_df[player_df["minutes"]>=900].copy()
    scatter_df["value_m"]=scatter_df["market_value_in_eur"]/1e6
    fig2=px.scatter(scatter_df,x="goals",y="value_m",color="position",size="minutes",hover_name="name",hover_data=["wc_nation","age"],labels={"goals":"Goals (2023-26)","value_m":"Market Value (€M)"},opacity=0.7)
    fig2.update_layout(**PLOTLY_LAYOUT,height=500)
    st.plotly_chart(fig2,use_container_width=True)

    # Age distribution
    st.markdown("<div class='sec-hdr'>Age Distribution by Position</div>",unsafe_allow_html=True)
    fig3=px.histogram(player_df,x="age",color="position",nbins=30,opacity=0.7,labels={"age":"Age"})
    fig3.update_layout(**PLOTLY_LAYOUT,height=400,barmode="overlay")
    st.plotly_chart(fig3,use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PLAYER STYLE CLUSTERING
# ═══════════════════════════════════════════════════════════════
elif section == "🎨 Player Style Clustering":
    st.markdown("<div class='sec-hdr'>Player Style Clustering — KMeans</div>",unsafe_allow_html=True)
    player_df=load_player_df()
    if player_df is None:
        st.error("⚠️ wc_players.csv not found. Run `python3 build_player_data.py` first.")
        st.stop()
    from player_model import run_clustering, CLUSTER_FEATURES

    st.markdown('<div class="info">KMeans clusters players into playing-style groups based on goals/90, assists/90, cards/90, minutes per appearance, height, and age. Choose a position and number of clusters.</div>',unsafe_allow_html=True)

    cc1,cc2=st.columns(2)
    with cc1: pos=st.selectbox("Position",["Attack","Midfield","Defender","Goalkeeper","All"])
    with cc2: k=st.slider("Number of clusters (styles)",3,6,5)

    with st.spinner("Running KMeans clustering…"):
        clustered,centroids=run_clustering(player_df,k,pos)

    if centroids is not None:
        st.markdown("<div class='sec-hdr'>Cluster Profiles</div>",unsafe_allow_html=True)
        cent_disp=centroids.copy()
        cent_disp["goals_per90"]=cent_disp["goals_per90"].round(2)
        cent_disp["assists_per90"]=cent_disp["assists_per90"].round(2)
        cent_disp["cards_per90"]=cent_disp["cards_per90"].round(2)
        cent_disp["age"]=cent_disp["age"].round(1)
        cent_disp["height_in_cm"]=cent_disp["height_in_cm"].round(0)
        cent_disp["mins_per_app"]=cent_disp["mins_per_app"].round(0)
        cent_disp=cent_disp[["style","size","goals_per90","assists_per90","cards_per90","mins_per_app","height_in_cm","age"]]
        cent_disp.columns=["Style","Players","Goals/90","Assists/90","Cards/90","Mins/App","Height","Avg Age"]
        st.dataframe(cent_disp,use_container_width=True,hide_index=True)

        # Scatter plot of clusters (goals vs assists)
        st.markdown("<div class='sec-hdr'>Style Map: Goals vs Assists per 90</div>",unsafe_allow_html=True)
        plot_df=clustered[clustered["minutes"]>=600].copy()
        fig=px.scatter(plot_df,x="goals_per90",y="assists_per90",color="style",size="market_value_in_eur",hover_name="name",hover_data=["wc_nation","position","age"],labels={"goals_per90":"Goals per 90","assists_per90":"Assists per 90"},opacity=0.75)
        fig.update_layout(**PLOTLY_LAYOUT,height=550,legend=dict(orientation="h",y=-0.15))
        st.plotly_chart(fig,use_container_width=True)

        # Players by cluster
        st.markdown("<div class='sec-hdr'>Explore Players by Style</div>",unsafe_allow_html=True)
        style_choice=st.selectbox("Select a style",sorted(clustered["style"].unique()))
        style_players=clustered[clustered["style"]==style_choice].nlargest(20,"market_value_in_eur")
        sp_disp=style_players[["name","wc_nation","position","age","goals_per90","assists_per90","market_value_in_eur"]].copy()
        sp_disp["market_value_in_eur"]=(sp_disp["market_value_in_eur"]/1e6).round(1)
        sp_disp["goals_per90"]=sp_disp["goals_per90"].round(2)
        sp_disp["assists_per90"]=sp_disp["assists_per90"].round(2)
        sp_disp.columns=["Player","Nation","Position","Age","Goals/90","Assists/90","Value (€M)"]
        st.dataframe(sp_disp,use_container_width=True,hide_index=True)

        # Cluster size pie
        st.markdown("<div class='sec-hdr'>Style Distribution</div>",unsafe_allow_html=True)
        size_df=clustered["style"].value_counts().reset_index()
        size_df.columns=["Style","Count"]
        fig2=px.pie(size_df,names="Style",values="Count",hole=0.45)
        fig2.update_layout(**PLOTLY_LAYOUT,height=400)
        st.plotly_chart(fig2,use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# ALL 48 TEAMS
# ═══════════════════════════════════════════════════════════════
elif section == "🗂️ All 48 Teams":
    st.markdown("<div class='sec-hdr'>All 48 Teams</div>",unsafe_allow_html=True)
    all_df=pd.DataFrame([{"Group":f"Group {TEAM_GROUP[t]}","Team":t,"Confederation":TEAM_STATS[t]["confederation"],"Elo":ELO_RATINGS.get(t,1600),"Attack":TEAM_STATS[t]["attack"],"Defense":TEAM_STATS[t]["defense"],"WC Titles":TEAM_STATS[t]["wc_titles"],"WC Apps":TEAM_STATS[t]["wc_apps"]} for t in sorted(ALL_TEAMS,key=lambda x:(TEAM_GROUP[x],x))])
    st.dataframe(all_df,use_container_width=True,hide_index=True,height=600)
    conf_df=all_df["Confederation"].value_counts().reset_index(); conf_df.columns=["Confederation","Teams"]
    fig=px.pie(conf_df,names="Confederation",values="Teams",hole=0.45)
    fig.update_layout(**PLOTLY_LAYOUT,height=400)
    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")
st.markdown('<div class="info"><strong>Sources:</strong> Elo (eloratings.net) · Match data (Kaggle martj42) · Player data (Kaggle davidcariboo/player-scores) · ML: scikit-learn · Viz: Plotly · Clustering: KMeans</div>',unsafe_allow_html=True)
