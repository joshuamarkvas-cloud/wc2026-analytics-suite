"""
FIFA World Cup 2026 Predictor - Full Enhanced Data & Model Module
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings("ignore")

WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Cote d Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
ALL_TEAMS = [t for teams in WC2026_GROUPS.values() for t in teams]
TEAM_GROUP = {t: g for g, teams in WC2026_GROUPS.items() for t in teams}

# Real Elo ratings (eloratings.net, June 2026)
ELO_RATINGS = {
    "Spain":2155,"Argentina":2113,"France":2062,"England":2020,
    "Brazil":1988,"Portugal":1984,"Colombia":1977,"Netherlands":1944,
    "Germany":1925,"Croatia":1933,"Ecuador":1933,"Norway":1922,
    "Switzerland":1897,"Uruguay":1890,"Turkey":1880,"Japan":1879,
    "Senegal":1869,"Belgium":1849,"Morocco":1840,"Mexico":1820,
    "United States":1810,"South Korea":1800,"Australia":1775,
    "Sweden":1770,"Austria":1760,"Scotland":1750,"Algeria":1740,
    "Cote d Ivoire":1735,"Iran":1720,"Tunisia":1700,"Egypt":1695,
    "Czech Republic":1690,"Canada":1685,"Ghana":1670,"Paraguay":1660,
    "Panama":1620,"Saudi Arabia":1615,"Bosnia and Herzegovina":1610,
    "South Africa":1600,"DR Congo":1590,"Iraq":1580,"Jordan":1560,
    "New Zealand":1540,"Uzbekistan":1530,"Cape Verde":1520,
    "Qatar":1490,"Haiti":1460,"Curacao":1420,
}

TEAM_STATS = {
    "Mexico":                {"attack":77,"defense":77,"wc_titles":0,"wc_apps":17,"gs":1.4,"gc":1.3,"confederation":"CONCACAF"},
    "South Africa":          {"attack":64,"defense":65,"wc_titles":0,"wc_apps":3, "gs":0.9,"gc":1.5,"confederation":"CAF"},
    "South Korea":           {"attack":73,"defense":74,"wc_titles":0,"wc_apps":11,"gs":1.2,"gc":1.3,"confederation":"AFC"},
    "Czech Republic":        {"attack":71,"defense":73,"wc_titles":0,"wc_apps":9, "gs":1.2,"gc":1.2,"confederation":"UEFA"},
    "Canada":                {"attack":71,"defense":71,"wc_titles":0,"wc_apps":2, "gs":1.1,"gc":1.4,"confederation":"CONCACAF"},
    "Bosnia and Herzegovina":{"attack":70,"defense":68,"wc_titles":0,"wc_apps":1, "gs":1.0,"gc":1.4,"confederation":"UEFA"},
    "Qatar":                 {"attack":60,"defense":62,"wc_titles":0,"wc_apps":2, "gs":0.7,"gc":1.8,"confederation":"AFC"},
    "Switzerland":           {"attack":75,"defense":79,"wc_titles":0,"wc_apps":12,"gs":1.3,"gc":1.1,"confederation":"UEFA"},
    "Brazil":                {"attack":87,"defense":85,"wc_titles":5,"wc_apps":22,"gs":2.3,"gc":1.0,"confederation":"CONMEBOL"},
    "Morocco":               {"attack":74,"defense":79,"wc_titles":0,"wc_apps":6, "gs":1.2,"gc":0.9,"confederation":"CAF"},
    "Haiti":                 {"attack":57,"defense":59,"wc_titles":0,"wc_apps":1, "gs":0.6,"gc":1.9,"confederation":"CONCACAF"},
    "Scotland":              {"attack":70,"defense":71,"wc_titles":0,"wc_apps":8, "gs":1.1,"gc":1.3,"confederation":"UEFA"},
    "United States":         {"attack":73,"defense":73,"wc_titles":0,"wc_apps":11,"gs":1.3,"gc":1.4,"confederation":"CONCACAF"},
    "Paraguay":              {"attack":67,"defense":67,"wc_titles":0,"wc_apps":9, "gs":1.0,"gc":1.4,"confederation":"CONMEBOL"},
    "Australia":             {"attack":70,"defense":72,"wc_titles":0,"wc_apps":6, "gs":1.1,"gc":1.5,"confederation":"AFC"},
    "Turkey":                {"attack":72,"defense":70,"wc_titles":0,"wc_apps":2, "gs":1.2,"gc":1.3,"confederation":"UEFA"},
    "Germany":               {"attack":84,"defense":86,"wc_titles":4,"wc_apps":20,"gs":2.2,"gc":1.1,"confederation":"UEFA"},
    "Curacao":               {"attack":53,"defense":55,"wc_titles":0,"wc_apps":1, "gs":0.6,"gc":2.0,"confederation":"CONCACAF"},
    "Cote d Ivoire":         {"attack":72,"defense":70,"wc_titles":0,"wc_apps":3, "gs":1.2,"gc":1.3,"confederation":"CAF"},
    "Ecuador":               {"attack":71,"defense":71,"wc_titles":0,"wc_apps":4, "gs":1.2,"gc":1.3,"confederation":"CONMEBOL"},
    "Netherlands":           {"attack":82,"defense":81,"wc_titles":0,"wc_apps":11,"gs":1.9,"gc":1.1,"confederation":"UEFA"},
    "Japan":                 {"attack":73,"defense":75,"wc_titles":0,"wc_apps":7, "gs":1.2,"gc":1.3,"confederation":"AFC"},
    "Sweden":                {"attack":74,"defense":74,"wc_titles":0,"wc_apps":12,"gs":1.4,"gc":1.2,"confederation":"UEFA"},
    "Tunisia":               {"attack":67,"defense":70,"wc_titles":0,"wc_apps":6, "gs":0.9,"gc":1.4,"confederation":"CAF"},
    "Belgium":               {"attack":83,"defense":79,"wc_titles":0,"wc_apps":14,"gs":1.7,"gc":1.1,"confederation":"UEFA"},
    "Egypt":                 {"attack":65,"defense":67,"wc_titles":0,"wc_apps":3, "gs":0.9,"gc":1.3,"confederation":"CAF"},
    "Iran":                  {"attack":68,"defense":72,"wc_titles":0,"wc_apps":6, "gs":1.0,"gc":1.3,"confederation":"AFC"},
    "New Zealand":           {"attack":61,"defense":63,"wc_titles":0,"wc_apps":2, "gs":0.7,"gc":1.8,"confederation":"OFC"},
    "Spain":                 {"attack":89,"defense":87,"wc_titles":1,"wc_apps":16,"gs":1.9,"gc":0.8,"confederation":"UEFA"},
    "Cape Verde":            {"attack":58,"defense":60,"wc_titles":0,"wc_apps":1, "gs":0.7,"gc":1.8,"confederation":"CAF"},
    "Saudi Arabia":          {"attack":67,"defense":67,"wc_titles":0,"wc_apps":6, "gs":1.0,"gc":1.6,"confederation":"AFC"},
    "Uruguay":               {"attack":79,"defense":79,"wc_titles":2,"wc_apps":14,"gs":1.6,"gc":1.1,"confederation":"CONMEBOL"},
    "France":                {"attack":87,"defense":85,"wc_titles":2,"wc_apps":16,"gs":2.0,"gc":1.0,"confederation":"UEFA"},
    "Senegal":               {"attack":75,"defense":75,"wc_titles":0,"wc_apps":4, "gs":1.3,"gc":1.2,"confederation":"CAF"},
    "Iraq":                  {"attack":60,"defense":61,"wc_titles":0,"wc_apps":1, "gs":0.7,"gc":1.7,"confederation":"AFC"},
    "Norway":                {"attack":76,"defense":73,"wc_titles":0,"wc_apps":2, "gs":1.4,"gc":1.2,"confederation":"UEFA"},
    "Argentina":             {"attack":88,"defense":83,"wc_titles":3,"wc_apps":18,"gs":2.1,"gc":1.1,"confederation":"CONMEBOL"},
    "Algeria":               {"attack":70,"defense":70,"wc_titles":0,"wc_apps":4, "gs":1.1,"gc":1.3,"confederation":"CAF"},
    "Austria":               {"attack":72,"defense":72,"wc_titles":0,"wc_apps":7, "gs":1.2,"gc":1.3,"confederation":"UEFA"},
    "Jordan":                {"attack":58,"defense":61,"wc_titles":0,"wc_apps":1, "gs":0.7,"gc":1.7,"confederation":"AFC"},
    "Portugal":              {"attack":87,"defense":80,"wc_titles":0,"wc_apps":9, "gs":1.8,"gc":1.0,"confederation":"UEFA"},
    "DR Congo":              {"attack":63,"defense":64,"wc_titles":0,"wc_apps":1, "gs":0.8,"gc":1.6,"confederation":"CAF"},
    "Uzbekistan":            {"attack":63,"defense":64,"wc_titles":0,"wc_apps":1, "gs":0.8,"gc":1.6,"confederation":"AFC"},
    "Colombia":              {"attack":77,"defense":74,"wc_titles":0,"wc_apps":6, "gs":1.5,"gc":1.2,"confederation":"CONMEBOL"},
    "England":               {"attack":83,"defense":82,"wc_titles":1,"wc_apps":16,"gs":1.8,"gc":1.1,"confederation":"UEFA"},
    "Croatia":               {"attack":79,"defense":80,"wc_titles":0,"wc_apps":7, "gs":1.5,"gc":1.0,"confederation":"UEFA"},
    "Ghana":                 {"attack":70,"defense":70,"wc_titles":0,"wc_apps":4, "gs":1.1,"gc":1.4,"confederation":"CAF"},
    "Panama":                {"attack":62,"defense":65,"wc_titles":0,"wc_apps":2, "gs":0.8,"gc":1.7,"confederation":"CONCACAF"},
}

NAME_ALIASES = {
    "South Korea":["Korea Republic","South Korea"],
    "United States":["USA","United States"],
    "Czech Republic":["Czech Republic","Czechoslovakia"],
    "Cote d Ivoire":["Ivory Coast","Cote d'Ivoire","Côte d'Ivoire"],
    "Turkey":["Turkey","Türkiye"],
    "Bosnia and Herzegovina":["Bosnia-Herzegovina","Bosnia and Herzegovina"],
    "DR Congo":["DR Congo","Congo DR","Zaire","Zaïre","Congo-Kinshasa"],
    "Cape Verde":["Cape Verde","Cabo Verde"],
    "Curacao":["Curacao","Curaçao","Netherlands Antilles"],
    "Ghana":["Ghana","Gold Coast"],
    "Egypt":["Egypt","United Arab Republic"],
}
ALIAS_TO_CANONICAL = {}
for canonical, aliases in NAME_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical
for t in TEAM_STATS:
    ALIAS_TO_CANONICAL[t.lower()] = t

def _canon(name):
    return ALIAS_TO_CANONICAL.get(str(name).strip().lower(), None)

FEATURE_COLS = [
    "elo_diff","elo_home","elo_away",
    "attack_diff","defense_diff","title_diff","experience_diff",
    "goal_diff","defense_adv",
    "h2h_win_pct","h2h_goal_diff_avg","h2h_matches",
    "home_form_pts","away_form_pts",
    "home_goals_per_game","away_goals_per_game",
    "home_conceded_per_game","away_conceded_per_game",
    "home_penalty_win_pct","away_penalty_win_pct",
    "home_wc_goals_per_game","away_wc_goals_per_game",
]

_CACHE = {}

def _load_all_data(results_path, goalscorers_path=None, shootouts_path=None):
    if results_path in _CACHE:
        return _CACHE[results_path]
    df = pd.read_csv(results_path)
    df["date"] = pd.to_datetime(df["date"])
    df["home_canon"] = df["home_team"].apply(_canon)
    df["away_canon"] = df["away_team"].apply(_canon)
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()

    gs_df = None
    if goalscorers_path and os.path.exists(goalscorers_path):
        gs_df = pd.read_csv(goalscorers_path)
        gs_df["date"] = pd.to_datetime(gs_df["date"])
        gs_df["team_canon"] = gs_df["team"].apply(_canon)

    so_df = None
    if shootouts_path and os.path.exists(shootouts_path):
        so_df = pd.read_csv(shootouts_path)
        so_df["date"] = pd.to_datetime(so_df["date"])
        so_df["winner_canon"] = so_df["winner"].apply(_canon)
        so_df["home_canon"]   = so_df["home_team"].apply(_canon)
        so_df["away_canon"]   = so_df["away_team"].apply(_canon)

    _CACHE[results_path] = (df, gs_df, so_df)
    return df, gs_df, so_df

def _safe(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else float(default)
    except:
        return float(default)

def _h2h(df, home, away):
    mask = (
        ((df["home_canon"]==home)&(df["away_canon"]==away)) |
        ((df["home_canon"]==away)&(df["away_canon"]==home))
    )
    sub = df[mask]
    if len(sub)==0: return 0.333, 0.0, 0
    wins=gf=ga=0
    for _,row in sub.iterrows():
        if row["home_canon"]==home: f,a=row["home_score"],row["away_score"]
        else: f,a=row["away_score"],row["home_score"]
        gf+=f; ga+=a
        wins += 1 if f>a else (0.5 if f==a else 0)
    n=len(sub)
    return _safe(wins/n,0.333), _safe((gf-ga)/n,0.0), n

def _team_form(df, team, n=10):
    mask=(df["home_canon"]==team)|(df["away_canon"]==team)
    sub=df[mask].sort_values("date").tail(n)
    if len(sub)==0: return 1.0
    pts=0
    for _,row in sub.iterrows():
        gf,ga=(row["home_score"],row["away_score"]) if row["home_canon"]==team else (row["away_score"],row["home_score"])
        pts += 3 if gf>ga else (1 if gf==ga else 0)
    return _safe(pts/len(sub))

def _team_goals(df, team, n=20):
    mask=(df["home_canon"]==team)|(df["away_canon"]==team)
    sub=df[mask].sort_values("date").tail(n)
    if len(sub)==0: return TEAM_STATS[team]["gs"], TEAM_STATS[team]["gc"]
    gf=ga=0
    for _,row in sub.iterrows():
        if row["home_canon"]==team: gf+=row["home_score"]; ga+=row["away_score"]
        else: gf+=row["away_score"]; ga+=row["home_score"]
    return _safe(gf/len(sub)), _safe(ga/len(sub))

def _wc_goals(df, team):
    mask=((df["home_canon"]==team)|(df["away_canon"]==team))&(df["tournament"]=="FIFA World Cup")
    sub=df[mask]
    if len(sub)==0: return _safe(TEAM_STATS[team]["gs"])
    gf=sum(row["home_score"] if row["home_canon"]==team else row["away_score"] for _,row in sub.iterrows())
    return _safe(gf/len(sub))

def _penalty_win_pct(so_df, team):
    if so_df is None: return 0.5
    mask=(so_df["home_canon"]==team)|(so_df["away_canon"]==team)
    sub=so_df[mask]
    if len(sub)==0: return 0.5
    return _safe((sub["winner_canon"]==team).sum()/len(sub), 0.5)

def build_features(home, away, df=None, gs_df=None, so_df=None):
    h,a = TEAM_STATS[home], TEAM_STATS[away]
    h_elo=float(ELO_RATINGS.get(home,1600))
    a_elo=float(ELO_RATINGS.get(away,1600))
    if df is not None:
        h2h_wp,h2h_gd,h2h_n = _h2h(df,home,away)
        h_form=_team_form(df,home); a_form=_team_form(df,away)
        h_gpg,h_cpg=_team_goals(df,home); a_gpg,a_cpg=_team_goals(df,away)
        h_wc=_wc_goals(df,home); a_wc=_wc_goals(df,away)
        h_pen=_penalty_win_pct(so_df,home); a_pen=_penalty_win_pct(so_df,away)
    else:
        h2h_wp,h2h_gd,h2h_n=0.333,0.0,0
        h_form=a_form=1.0
        h_gpg,h_cpg=h["gs"],h["gc"]; a_gpg,a_cpg=a["gs"],a["gc"]
        h_wc=h["gs"]; a_wc=a["gs"]; h_pen=a_pen=0.5
    feats={
        "elo_diff":h_elo-a_elo,"elo_home":h_elo,"elo_away":a_elo,
        "attack_diff":float(h["attack"]-a["attack"]),
        "defense_diff":float(h["defense"]-a["defense"]),
        "title_diff":float(h["wc_titles"]-a["wc_titles"]),
        "experience_diff":float(h["wc_apps"]-a["wc_apps"]),
        "goal_diff":float(h["gs"]-a["gs"]),"defense_adv":float(a["gc"]-h["gc"]),
        "h2h_win_pct":h2h_wp,"h2h_goal_diff_avg":h2h_gd,"h2h_matches":float(h2h_n),
        "home_form_pts":h_form,"away_form_pts":a_form,
        "home_goals_per_game":h_gpg,"away_goals_per_game":a_gpg,
        "home_conceded_per_game":h_cpg,"away_conceded_per_game":a_cpg,
        "home_penalty_win_pct":h_pen,"away_penalty_win_pct":a_pen,
        "home_wc_goals_per_game":h_wc,"away_wc_goals_per_game":a_wc,
    }
    return {k:(_safe(v) if not np.isfinite(float(v)) else float(v)) for k,v in feats.items()}

def _generate_training_data(df=None, gs_df=None, so_df=None, n=10000, seed=42):
    np.random.seed(seed)
    records=[]
    team_list=list(TEAM_STATS.keys())
    real_rows=[]
    if df is not None:
        wc_set=set(TEAM_STATS.keys())
        hist=df[df["home_canon"].isin(wc_set)&df["away_canon"].isin(wc_set)].copy()
        hist=hist[hist["date"]<"2026-06-11"]
        for _,row in hist.iterrows():
            home,away=row["home_canon"],row["away_canon"]
            if home not in TEAM_STATS or away not in TEAM_STATS: continue
            hg,ag=row["home_score"],row["away_score"]
            outcome="Home Win" if hg>ag else ("Away Win" if ag>hg else "Draw")
            feats=build_features(home,away,df,gs_df,so_df)
            real_rows.append({**feats,"outcome":outcome})
    for _ in range(n):
        home=np.random.choice(team_list)
        away=np.random.choice([t for t in team_list if t!=home])
        feats=build_features(home,away,df,gs_df,so_df)
        s=(feats["elo_diff"]*0.008+feats["attack_diff"]*0.2+feats["defense_diff"]*0.15+
           feats["title_diff"]*1.2+feats["experience_diff"]*0.25+feats["goal_diff"]*1.5+
           feats["defense_adv"]*1.5+(feats["h2h_win_pct"]-0.333)*12+
           feats["h2h_goal_diff_avg"]*1.5+(feats["home_form_pts"]-feats["away_form_pts"])*0.5+
           np.random.normal(0,3))
        if s>8: p=[0.65,0.15,0.20]
        elif s>3: p=[0.50,0.22,0.28]
        elif s>-3: p=[0.32,0.30,0.38]
        elif s>-8: p=[0.22,0.22,0.56]
        else: p=[0.15,0.15,0.70]
        outcome=np.random.choice(["Home Win","Draw","Away Win"],p=p)
        records.append({**feats,"outcome":outcome})
    all_rows=real_rows+records
    result_df=pd.DataFrame(all_rows)
    result_df[FEATURE_COLS]=result_df[FEATURE_COLS].fillna(0.0)
    return result_df, len(real_rows)

def train_models(results_path=None, goalscorers_path=None, shootouts_path=None):
    df,gs_df,so_df=None,None,None
    data_source="synthetic only"
    real_match_count=0
    if results_path and os.path.exists(results_path):
        try:
            df,gs_df,so_df=_load_all_data(results_path,goalscorers_path,shootouts_path)
            data_source="Kaggle historical + synthetic"
        except Exception as e:
            data_source=f"synthetic (error:{e})"
    train_df,real_match_count=_generate_training_data(df,gs_df,so_df)
    X=train_df[FEATURE_COLS].fillna(0.0)
    y=train_df["outcome"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    classifiers={
        "Random Forest":Pipeline([("rf",RandomForestClassifier(n_estimators=400,max_depth=12,min_samples_leaf=3,random_state=42))]),
        "Logistic Regression":Pipeline([("scaler",StandardScaler()),("lr",LogisticRegression(max_iter=2000,C=0.5,random_state=42))]),
    }
    results={}
    for name,clf in classifiers.items():
        clf.fit(X_train,y_train)
        cv=cross_val_score(clf,X_train,y_train,cv=5).mean()
        tacc=accuracy_score(y_test,clf.predict(X_test))
        results[name]={"model":clf,"cv":cv,"test_acc":tacc}
    best_name=max(results,key=lambda k:results[k]["cv"])
    best=results[best_name]
    fi=None
    rf_step=best["model"].named_steps.get("rf")
    if rf_step and hasattr(rf_step,"feature_importances_"):
        fi=dict(zip(FEATURE_COLS,rf_step.feature_importances_))
    return {
        "model":best["model"],"model_name":best_name,
        "cv_acc":best["cv"],"test_acc":best["test_acc"],
        "all_results":results,"feature_imp":fi,
        "classes":list(best["model"].classes_),
        "train_size":len(X_train),"test_size":len(X_test),
        "data_source":data_source,"real_matches_used":real_match_count,
        "df":df,"gs_df":gs_df,"so_df":so_df,
    }

def predict_match(model_data, home, away):
    df=model_data.get("df"); gs_df=model_data.get("gs_df"); so_df=model_data.get("so_df")
    feats=build_features(home,away,df,gs_df,so_df)
    X=pd.DataFrame([feats])[FEATURE_COLS].fillna(0.0)
    proba=model_data["model"].predict_proba(X)[0]
    classes=model_data["classes"]
    prob_dict=dict(zip(classes,proba))
    return {
        "prediction":classes[proba.argmax()],
        "home_win":prob_dict.get("Home Win",0),
        "draw":prob_dict.get("Draw",0),
        "away_win":prob_dict.get("Away Win",0),
        "home_stats":TEAM_STATS[home],"away_stats":TEAM_STATS[away],
        "home_elo":ELO_RATINGS.get(home,1600),"away_elo":ELO_RATINGS.get(away,1600),
        "features":feats,
    }

def predict_scoreline(model_data, home, away):
    """Poisson model scoreline prediction"""
    df=model_data.get("df")
    h,a=TEAM_STATS[home],TEAM_STATS[away]
    if df is not None:
        h_gpg,h_cpg=_team_goals(df,home)
        a_gpg,a_cpg=_team_goals(df,away)
    else:
        h_gpg,h_cpg=h["gs"],h["gc"]; a_gpg,a_cpg=a["gs"],a["gc"]
    elo_adj=(ELO_RATINGS.get(home,1600)-ELO_RATINGS.get(away,1600))/400
    h_lambda=max(0.3,(h_gpg*0.6+a_cpg*0.4)*(1+elo_adj*0.15))
    a_lambda=max(0.3,(a_gpg*0.6+h_cpg*0.4)*(1-elo_adj*0.15))
    scores=[]
    for hg in range(7):
        for ag in range(7):
            from math import exp, factorial
            p_h=exp(-h_lambda)*(h_lambda**hg)/factorial(hg)
            p_a=exp(-a_lambda)*(a_lambda**ag)/factorial(ag)
            scores.append({"hg":hg,"ag":ag,"prob":p_h*p_a})
    scores.sort(key=lambda x:-x["prob"])
    return scores[:10], h_lambda, a_lambda

def get_team_form_history(df, team, n=20):
    """Returns last n match results for form chart"""
    if df is None: return []
    mask=(df["home_canon"]==team)|(df["away_canon"]==team)
    sub=df[mask].sort_values("date").tail(n)
    rows=[]
    for _,row in sub.iterrows():
        is_home=row["home_canon"]==team
        gf=row["home_score"] if is_home else row["away_score"]
        ga=row["away_score"] if is_home else row["home_score"]
        opp=row["away_canon"] if is_home else row["home_canon"]
        result="W" if gf>ga else ("D" if gf==ga else "L")
        rows.append({"date":row["date"],"opponent":opp,"gf":int(gf),"ga":int(ga),"result":result,"tournament":row["tournament"]})
    return rows

def get_h2h_history(df, home, away):
    """Full H2H match history"""
    if df is None: return []
    mask=(((df["home_canon"]==home)&(df["away_canon"]==away))|
          ((df["home_canon"]==away)&(df["away_canon"]==home)))
    sub=df[mask].sort_values("date")
    rows=[]
    for _,row in sub.iterrows():
        if row["home_canon"]==home:
            hg,ag=row["home_score"],row["away_score"]
        else:
            hg,ag=row["away_score"],row["home_score"]
        result="W" if hg>ag else ("D" if hg==ag else "L")
        rows.append({"date":row["date"],"home_score":int(hg),"away_score":int(ag),"result":result,"tournament":row["tournament"]})
    return rows

def get_top_wc_scorers(df, gs_df, n=20):
    """Top WC scorers from real data"""
    if df is None or gs_df is None: return []
    wc=df[df["tournament"]=="FIFA World Cup"][["date","home_team","away_team"]].copy()
    merged=gs_df.merge(wc,on=["date","home_team","away_team"])
    merged=merged[merged["own_goal"]==False]
    top=merged.groupby(["scorer","team"]).agg(
        goals=("scorer","count"),
        penalties=("penalty","sum")
    ).reset_index().sort_values("goals",ascending=False).head(n)
    return top.to_dict("records")

def simulate_group_stage(model_data):
    standings={}
    for group,teams in WC2026_GROUPS.items():
        table={t:{"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"GD":0,"Pts":0} for t in teams}
        matches=[]
        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                home,away=teams[i],teams[j]
                r=predict_match(model_data,home,away)
                pred=r["prediction"]
                h_exp=max(0.5,r["home_stats"]["gs"]*0.6+r["away_stats"]["gc"]*0.4)
                a_exp=max(0.5,r["away_stats"]["gs"]*0.6+r["home_stats"]["gc"]*0.4)
                np.random.seed(abs(hash((home,away)))%(2**31))
                hg=max(0,round(np.random.poisson(h_exp)))
                ag=max(0,round(np.random.poisson(a_exp)))
                if pred=="Home Win" and hg<=ag: hg=ag+1
                elif pred=="Away Win" and ag<=hg: ag=hg+1
                elif pred=="Draw": hg=ag=min(hg,ag)
                table[home]["P"]+=1; table[away]["P"]+=1
                table[home]["GF"]+=hg; table[home]["GA"]+=ag; table[home]["GD"]+=hg-ag
                table[away]["GF"]+=ag; table[away]["GA"]+=hg; table[away]["GD"]+=ag-hg
                if hg>ag: table[home]["W"]+=1; table[home]["Pts"]+=3; table[away]["L"]+=1
                elif ag>hg: table[away]["W"]+=1; table[away]["Pts"]+=3; table[home]["L"]+=1
                else: table[home]["D"]+=1; table[home]["Pts"]+=1; table[away]["D"]+=1; table[away]["Pts"]+=1
                matches.append({"home":home,"away":away,"hg":hg,"ag":ag})
        sorted_table=sorted(table.items(),key=lambda x:(-x[1]["Pts"],-x[1]["GD"],-x[1]["GF"]))
        standings[group]={"table":sorted_table,"matches":matches}
    return standings

def run_monte_carlo(model_data, n_sims=5000):
    """Run full tournament Monte Carlo simulation"""
    win_counts={t:0 for t in ALL_TEAMS}
    r32_counts={t:0 for t in ALL_TEAMS}
    sf_counts={t:0 for t in ALL_TEAMS}
    final_counts={t:0 for t in ALL_TEAMS}

    for sim in range(n_sims):
        np.random.seed(sim)
        # Group stage
        standings={}
        for group,teams in WC2026_GROUPS.items():
            table={t:{"Pts":0,"GD":0,"GF":0} for t in teams}
            for i in range(len(teams)):
                for j in range(i+1,len(teams)):
                    home,away=teams[i],teams[j]
                    r=predict_match(model_data,home,away)
                    h_lambda=max(0.3,TEAM_STATS[home]["gs"]*0.6+TEAM_STATS[away]["gc"]*0.4)
                    a_lambda=max(0.3,TEAM_STATS[away]["gs"]*0.6+TEAM_STATS[home]["gc"]*0.4)
                    hg=np.random.poisson(h_lambda); ag=np.random.poisson(a_lambda)
                    table[home]["GF"]+=hg; table[home]["GD"]+=hg-ag
                    table[away]["GF"]+=ag; table[away]["GD"]+=ag-hg
                    if hg>ag: table[home]["Pts"]+=3
                    elif ag>hg: table[away]["Pts"]+=3
                    else: table[home]["Pts"]+=1; table[away]["Pts"]+=1
            sorted_t=sorted(table.items(),key=lambda x:(-x[1]["Pts"],-x[1]["GD"],-x[1]["GF"]))
            standings[group]=sorted_t

        # Collect qualifiers
        qualified=[]
        thirds=[]
        for g,sorted_t in standings.items():
            qualified.append(sorted_t[0][0])
            qualified.append(sorted_t[1][0])
            thirds.append((sorted_t[2][0],sorted_t[2][1]))
        thirds.sort(key=lambda x:(-x[1]["Pts"],-x[1]["GD"],-x[1]["GF"]))
        for t,_ in thirds[:8]:
            qualified.append(t)
            r32_counts[t]+=1
        for t in qualified[:24]:
            r32_counts[t]+=1

        # Knockout rounds
        def play_knockout(team_a, team_b):
            r=predict_match(model_data,team_a,team_b)
            probs=[r["home_win"],r["draw"],r["away_win"]]
            # In knockout, draw goes to AET/pens - slight home advantage in pens
            h_pen=_penalty_win_pct(model_data.get("so_df"),team_a)
            outcome=np.random.choice(["H","D","A"],p=probs)
            if outcome=="H": return team_a
            if outcome=="A": return team_b
            return team_a if np.random.random()<h_pen else team_b

        round_teams=qualified[:]
        np.random.shuffle(round_teams)
        for rnd_name in ["R32","R16","QF","SF","F"]:
            next_round=[]
            for k in range(0,len(round_teams),2):
                if k+1<len(round_teams):
                    winner=play_knockout(round_teams[k],round_teams[k+1])
                    next_round.append(winner)
                    if rnd_name=="SF": final_counts[winner]+=1
            round_teams=next_round
        if round_teams:
            win_counts[round_teams[0]]+=1

    results={}
    for t in ALL_TEAMS:
        results[t]={
            "r32_pct":round(r32_counts[t]/n_sims*100,1),
            "sf_pct":round(sf_counts[t]/n_sims*100,1),
            "final_pct":round(final_counts[t]/n_sims*100,1),
            "win_pct":round(win_counts[t]/n_sims*100,1),
        }
    return results
