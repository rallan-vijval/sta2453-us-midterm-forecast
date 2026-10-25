"""
Purpose : Simulate every dataset the pipeline will produce, so that scripts and
          tests can be written and checked before any real data exist.
Author  : Allwin Steve Paul Justin Anantharaj (Member 3)
Date    : 2026-10-22
Inputs  : none
Outputs : data/00-simulated_data/
            races.parquet               one row per 2026 race (36 states + 3 territories)
            candidates.parquet          one row per candidate
            governors_not_up.parquet    the 14 states with no race in 2026
            history.parquet             simulated 2014/2018/2022 governor results
            personas.parquet            150 (competitive) or 75 (safe) personas per state
            cells.parquet, frame.parquet  Method B groups and their population shares
            responses_method_a.parquet  one row per persona x model x repeat
            method_a_estimates.parquet  Method A vote shares per race x model x repeat
            method_b_estimates.parquet  Method B vote shares per race x model x repeat
            method_c_responses.parquet  "forecaster" answers per race x model x repeat
            polls.parquet               simulated polls in competitive states
            forecast.parquet            a simple simulated forecast per race
            simulations.parquet         1,000 simulated draws of governorships
Notes   : Variables depend on each other the way real ones do:
          - party identity depends on state lean and age;
          - ideology depends on party identity;
          - education depends on age, and income depends on education;
          - vote choice depends on party identity, incumbency and the model;
          - turnout depends on age and strength of partisanship;
          - the House vote mostly matches the governor vote (some ticket-splitting).
          About 2% of governor votes are missing and 0.5% of answers are malformed,
          on purpose, so the tests have something to catch.
Run     : uv run python scripts/00-simulate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Set-up
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "00-simulated_data"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(853)

# The 36 states that elect a governor on 3 November 2026.
STATES_2026 = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Iowa",
    "Kansas", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Nebraska", "Nevada", "New Hampshire", "New Mexico", "New York", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Vermont", "Wisconsin", "Wyoming",
]
TERRITORIES = ["Guam", "Northern Mariana Islands", "U.S. Virgin Islands"]

# The 14 states whose governor is not up in 2026.
NOT_UP = [
    "Delaware", "Indiana", "Kentucky", "Louisiana", "Mississippi", "Missouri",
    "Montana", "New Jersey", "North Carolina", "North Dakota", "Utah",
    "Virginia", "Washington", "West Virginia",
]

# Three model sets, one per team member, three placeholder models each.
MODELS = pd.DataFrame(
    {
        "member": ["member1"] * 3 + ["member2"] * 3 + ["member3"] * 3,
        "model_set": ["closed_us"] * 3 + ["open_chinese"] * 3 + ["open_western"] * 3,
        "model_slug": [f"sim/model-{i}" for i in range(1, 10)],
    }
)
# Each model leans slightly toward one party (in log-odds), as real models do.
MODELS["model_bias"] = rng.normal(0, 0.35, len(MODELS))
REPEATS = 3

PARTY7 = [
    "a strong Democrat", "a not very strong Democrat", "an independent who leans Democratic",
    "an independent", "an independent who leans Republican",
    "a not very strong Republican", "a strong Republican",
]
# Log-odds of voting for the Democrat, by party identity.
PARTY7_LOGIT = dict(zip(PARTY7, [3.2, 2.2, 1.6, 0.0, -1.6, -2.2, -3.2]))
# Mean position on the 1 (very liberal) to 5 (very conservative) scale.
PARTY7_IDEOLOGY = dict(zip(PARTY7, [1.7, 2.3, 2.4, 3.0, 3.6, 3.7, 4.3]))
IDEOLOGY = ["very liberal", "liberal", "moderate", "conservative", "very conservative"]
TURNOUT = ["Definitely will vote", "Probably will vote", "Unlikely to vote", "Will not vote"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# 1. Races
# ---------------------------------------------------------------------------
# Partisan lean in Cook PVI style: positive = more Democratic than the nation.
pvi = rng.normal(0, 9, len(STATES_2026)).round()
lean_party = np.where(pvi > 0, "D", np.where(pvi < 0, "R", "EVEN"))
# Most governors match their state's lean, but about one in seven do not.
p_dem_gov = np.clip(sigmoid(pvi / 3) * 0.86 + 0.07, 0, 1)
incumbent_party = np.where(rng.random(len(pvi)) < p_dem_gov, "D", "R")
open_seat = rng.random(len(pvi)) < 0.45
competitive = (np.abs(pvi) <= 5) | ((lean_party != "EVEN") & (lean_party != incumbent_party))

states = pd.DataFrame(
    {
        "race": STATES_2026,
        "type": "state",
        "pvi": pvi,
        "incumbent_party": incumbent_party,
        "open_seat": open_seat,
        "competitive": competitive,
    }
)
territories = pd.DataFrame(
    {
        "race": TERRITORIES,
        "type": "territory",
        "pvi": np.nan,
        "incumbent_party": ["D", "I", "D"],
        "open_seat": [True, False, True],
        "competitive": False,
    }
)
races = states.copy()  # 36 state races only; territories are out of scope 
# races = pd.concat([states, territories], ignore_index=True)

# A "true" 2026 Democratic two-party margin per race (percentage points), used
# only to make the simulated answers and polls hang together.
inc_sign = np.where(races["open_seat"], 0, np.where(races["incumbent_party"] == "D", 1, -1))
midterm_shift = 4.0  # the president's party (Republican in 2026) usually loses ground
races["true_margin"] = (
    1.8 * races["pvi"].fillna(10) + 6 * inc_sign + midterm_shift + rng.normal(0, 8, len(races))
)

# ---------------------------------------------------------------------------
# 2. Candidates (Democrat, Republican, sometimes an independent)
# ---------------------------------------------------------------------------
cand_rows = []
for _, r in races.iterrows():
    cand_rows.append({"race": r.race, "candidate": f"{r.race} Democrat", "party": "D"})
    cand_rows.append({"race": r.race, "candidate": f"{r.race} Republican", "party": "R"})
    if rng.random() < 0.25:
        cand_rows.append({"race": r.race, "candidate": f"{r.race} Independent", "party": "O"})
candidates = pd.DataFrame(cand_rows).merge(
    races[["race", "incumbent_party", "open_seat"]], on="race"
)
candidates["incumbent"] = (~candidates["open_seat"]) & (
    candidates["party"] == candidates["incumbent_party"]
)
candidates = candidates[["race", "candidate", "party", "incumbent"]]

# ---------------------------------------------------------------------------
# 3. The 14 governorships not up in 2026
# ---------------------------------------------------------------------------
governors_not_up = pd.DataFrame(
    {"state": NOT_UP, "party": rng.choice(["D", "R"], size=len(NOT_UP), p=[0.45, 0.55])}
)

# ---------------------------------------------------------------------------
# 4. History: 2014, 2018, 2022 results in the same 36 states (for Method C)
# ---------------------------------------------------------------------------
PRES_DEM = {2014: 1, 2018: -1, 2022: 1}  # +1 when the president is a Democrat
hist_rows = []
for year, pres in PRES_DEM.items():
    for s, p in zip(STATES_2026, pvi):
        p_year = p + rng.normal(0, 1.5)
        ran = rng.random() < 0.55
        inc_p = rng.choice(["D", "R"], p=[sigmoid(p_year / 3), 1 - sigmoid(p_year / 3)])
        sign = (1 if inc_p == "D" else -1) if ran else 0
        margin = 1.8 * p_year + 6 * sign - 4 * pres + rng.normal(0, 9)
        other = rng.uniform(0, 4)
        dem = (100 - other) * (50 + margin / 2) / 100
        hist_rows.append(
            {
                "year": year,
                "race": s,
                "pvi": round(p_year),
                "incumbent_party": inc_p,
                "incumbent_ran": ran,
                "dem_pct": round(dem, 1),
                "rep_pct": round(100 - other - dem, 1),
            }
        )
history = pd.DataFrame(hist_rows)

# ---------------------------------------------------------------------------
# 5. Personas
# ---------------------------------------------------------------------------
persona_rows = []
pid = 0
for _, r in states.iterrows():
    n = 150 if r.competitive else 75
    age = np.clip(np.round(18 + rng.beta(1.6, 1.9, n) * 72), 18, 95).astype(int)
    # Party group from a three-way logit: Democratic-leaning states and younger
    # people lean Democratic.
    u_d = r.pvi / 14 - 0.012 * (age - 50)
    u_r = -r.pvi / 14 + 0.012 * (age - 50)
    u_i = np.full(n, -0.5)
    expu = np.exp(np.vstack([u_d, u_i, u_r]))
    probs = expu / expu.sum(axis=0)
    group = np.array([rng.choice(["D", "I", "R"], p=probs[:, i]) for i in range(n)])
    party7 = []
    for g in group:
        if g == "D":
            party7.append(rng.choice(PARTY7[:3], p=[0.5, 0.25, 0.25]))
        elif g == "R":
            party7.append(rng.choice(PARTY7[4:], p=[0.25, 0.25, 0.5]))
        else:
            party7.append(PARTY7[3])
    party7 = np.array(party7)
    ideo_num = np.clip(
        np.round(np.array([PARTY7_IDEOLOGY[p] for p in party7]) + rng.normal(0, 0.7, n)), 1, 5
    ).astype(int)
    degree = rng.random(n) < sigmoid(-0.6 + 0.4 * (ideo_num < 3) - 0.01 * (age - 45))
    education = np.where(
        degree,
        rng.choice(["a four-year college degree", "a postgraduate degree"], n, p=[0.65, 0.35]),
        rng.choice(["a high school diploma", "some college", "a two-year degree"], n, p=[0.45, 0.35, 0.2]),
    )
    log_income = rng.normal(10.9 + 0.45 * degree, 0.6)
    income = pd.cut(
        np.exp(log_income),
        bins=[0, 30_000, 50_000, 80_000, 120_000, 200_000, np.inf],
        labels=["under $30,000", "$30,000-$49,999", "$50,000-$79,999",
                "$80,000-$119,999", "$120,000-$199,999", "$200,000 or more"],
    ).astype(str)
    gender = rng.choice(["man", "woman"], n)
    race_eth = rng.choice(["White", "Black", "Hispanic", "Asian", "Other"], n,
                          p=[0.64, 0.12, 0.14, 0.06, 0.04])
    survey_year = np.where(rng.random(n) < 0.85, 2025, 2024)
    weight = rng.lognormal(0, 0.5, n)
    weight = weight / weight.mean()
    p_harris = sigmoid(np.array([PARTY7_LOGIT[p] for p in party7]) * 1.1)
    vote_2024 = np.where(rng.random(n) < p_harris, "Harris", "Trump")
    for i in range(n):
        persona_rows.append(
            {
                "persona_id": f"P{pid:05d}",
                "race": r.race,
                "survey_year": int(survey_year[i]),
                "age": int(age[i]),
                "gender": gender[i],
                "race_ethnicity": race_eth[i],
                "education": education[i],
                "degree": bool(degree[i]),
                "income": income[i],
                "party_id": party7[i],
                "party_group": group[i],
                "ideology": IDEOLOGY[ideo_num[i] - 1],
                "weight": float(weight[i]),
                "validation": i < 10,
                "vote_2024": vote_2024[i],
            }
        )
        pid += 1
personas = pd.DataFrame(persona_rows)
personas["age_group"] = pd.cut(
    personas["age"], [17, 29, 44, 64, 120], labels=["18-29", "30-44", "45-64", "65+"]
).astype(str)
personas["persona_text"] = (
    "You are a " + personas["age"].astype(str) + "-year-old " + personas["gender"]
    + " who lives in " + personas["race"] + ". You describe your race/ethnicity as "
    + personas["race_ethnicity"] + ". Your highest level of education is "
    + personas["education"] + ", and your family income is " + personas["income"]
    + ". In politics, you consider yourself " + personas["party_id"]
    + " and your views are " + personas["ideology"] + ". You are registered to vote."
)

# ---------------------------------------------------------------------------
# 6. Method B groups (cells) and the weighting frame
# ---------------------------------------------------------------------------
cells = (
    pd.MultiIndex.from_product(
        [STATES_2026, ["D", "I", "R"], ["18-29", "30-44", "45-64", "65+"], [True, False]],
        names=["race", "party_group", "age_group", "degree"],
    )
    .to_frame(index=False)
)
cells["cell_id"] = [f"C{i:04d}" for i in range(len(cells))]
cells["cell_description"] = (
    "registered " + cells["party_group"].map({"D": "Democrats", "I": "Independents", "R": "Republicans"})
    + " aged " + cells["age_group"] + " "
    + np.where(cells["degree"], "with", "without") + " a four-year college degree"
)
frame = cells[["race", "cell_id", "party_group", "age_group", "degree"]].copy()
frame["share"] = np.concatenate(
    [rng.dirichlet(np.ones(24) * 3) for _ in STATES_2026]
)

# ---------------------------------------------------------------------------
# 7. Method A responses: persona x model x repeat
# ---------------------------------------------------------------------------
resp = personas[["persona_id", "race", "age", "party_id"]].merge(MODELS, how="cross")
resp = resp.loc[resp.index.repeat(REPEATS)].reset_index(drop=True)
resp["repeat"] = resp.groupby(["persona_id", "model_slug"]).cumcount() + 1
resp = resp.merge(races[["race", "incumbent_party", "open_seat"]], on="race")

inc = np.where(resp["open_seat"], 0, np.where(resp["incumbent_party"] == "D", 1, -1))
u = resp["party_id"].map(PARTY7_LOGIT) + 0.4 * inc + resp["model_bias"] + rng.logistic(0, 0.6, len(resp))
p_dem = sigmoid(u)
q = 0.04 + 0.08 * (resp["party_id"] == PARTY7[3])  # chance of "Other"/"Don't know"
draw = rng.random(len(resp))
choice = np.where(
    draw < q * 0.4, "Other",
    np.where(draw < q, "Don't know", np.where(rng.random(len(resp)) < p_dem, "D", "R")),
)
resp["governor_vote"] = np.where(
    choice == "D", resp["race"] + " Democrat",
    np.where(choice == "R", resp["race"] + " Republican", choice),
)
# House vote: usually the same party; about 8% split their ticket.
split = rng.random(len(resp)) < 0.08
house = np.where(choice == "D", "D", np.where(choice == "R", "R", "Don't know"))
house = np.where(split & (house == "D"), "R", np.where(split & (house == "R"), "D", house))
resp["house_vote"] = pd.Series(house).map(
    {"D": "The Democratic Party candidate", "R": "The Republican Party candidate", "Don't know": "Don't know"}
).values
# Turnout: older people and strong partisans say they will vote.
strong = resp["party_id"].str.contains("strong").astype(float)
p_def = np.clip(0.45 + 0.005 * (resp["age"] - 45) + 0.15 * strong, 0.05, 0.95)
t = rng.random(len(resp))
resp["turnout"] = np.where(
    t < p_def, TURNOUT[0],
    np.where(t < p_def + 0.25, TURNOUT[1], np.where(t < p_def + 0.33, TURNOUT[2], TURNOUT[3])),
)
resp["raw_text"] = (
    '{"turnout": "' + resp["turnout"] + '", "governor_vote": "' + resp["governor_vote"]
    + '", "house_vote": "' + resp["house_vote"] + '"}'
)
resp["parsed"] = True
# Deliberate problems: 2% missing governor votes, 0.5% malformed answers.
missing = rng.random(len(resp)) < 0.02
malformed = (~missing) & (rng.random(len(resp)) < 0.005)
resp.loc[missing, "governor_vote"] = np.nan
resp.loc[malformed, ["turnout", "governor_vote", "house_vote"]] = np.nan
resp.loc[malformed, "raw_text"] = "I'm sorry, but I can't answer as this person."
resp.loc[missing | malformed, "parsed"] = False
resp["method"] = "A"
resp["run_date"] = "2026-10-23"
resp["prompt_tokens"] = rng.integers(280, 360, len(resp))
resp["completion_tokens"] = rng.integers(25, 45, len(resp))
resp["cost_usd"] = (resp["prompt_tokens"] * 0.15 + resp["completion_tokens"] * 0.6) / 1e6
responses_a = resp[
    ["run_date", "member", "model_set", "model_slug", "method", "persona_id", "race",
     "repeat", "turnout", "governor_vote", "house_vote", "raw_text", "parsed",
     "prompt_tokens", "completion_tokens", "cost_usd"]
]

# ---------------------------------------------------------------------------
# 8. Method A estimates (what Member 1's script will produce)
# ---------------------------------------------------------------------------
TURNOUT_WEIGHT = dict(zip(TURNOUT, [1.0, 0.7, 0.2, 0.0]))
ok = responses_a[responses_a["parsed"]].copy()
ok["w"] = ok["turnout"].map(TURNOUT_WEIGHT)
ok["is_d"] = ok["governor_vote"].str.endswith(" Democrat")
ok["is_r"] = ok["governor_vote"].str.endswith(" Republican")
g = ok.groupby(["race", "member", "model_set", "model_slug", "repeat"])
method_a = g.apply(
    lambda d: pd.Series(
        {
            "dem_share": 100 * (d.w * d.is_d).sum() / d.w.sum(),
            "rep_share": 100 * (d.w * d.is_r).sum() / d.w.sum(),
            "n_personas": d.persona_id.nunique(),
        }
    ),
    include_groups=False,
).reset_index()
method_a["other_share"] = 100 - method_a["dem_share"] - method_a["rep_share"]
method_a.insert(0, "method", "A")

# ---------------------------------------------------------------------------
# 9. Method B estimates and Method C "forecaster" answers
# ---------------------------------------------------------------------------
grid = races[["race", "type", "true_margin"]].merge(MODELS, how="cross")
grid = grid.loc[grid.index.repeat(REPEATS)].reset_index(drop=True)
grid["repeat"] = grid.groupby(["race", "model_slug"]).cumcount() + 1

b = grid[grid["type"] == "state"].copy()
other = rng.uniform(2, 6, len(b))
two_party_d = 50 + 0.8 * b["true_margin"] / 2 + 6 * b["model_bias"] + rng.normal(0, 1.5, len(b))
b["dem_share"] = (100 - other) * two_party_d / 100
b["rep_share"] = 100 - other - b["dem_share"]
b["other_share"] = other
b["n_cells"] = 24
method_b = b[["race", "member", "model_set", "model_slug", "repeat",
              "dem_share", "rep_share", "n_cells", "other_share"]].copy()
method_b.insert(0, "method", "B")

c = grid.copy()
c["dem_two_party_share"] = np.clip(
    50 + c["true_margin"] / 2 + 5 * c["model_bias"] + rng.normal(0, 3, len(c)), 5, 95
).round(1)
c["uncertainty_plus_minus"] = rng.choice([3, 4, 5, 6, 8], len(c))
c["probability_dem_wins"] = np.round(
    sigmoid((c["dem_two_party_share"] - 50) / c["uncertainty_plus_minus"] * 1.7), 2
)
method_c = c[["race", "member", "model_set", "model_slug", "repeat",
              "dem_two_party_share", "uncertainty_plus_minus", "probability_dem_wins"]]

# ---------------------------------------------------------------------------
# 10. Polls (competitive states only, 0-6 polls each)
# ---------------------------------------------------------------------------
poll_rows = []
for _, r in races[races["competitive"]].iterrows():
    for k in range(rng.integers(0, 7)):
        n = int(rng.integers(400, 1500))
        und = rng.uniform(3, 10)
        d = (100 - und) * (50 + (r.true_margin + rng.normal(0, 4)) / 2) / 100
        poll_rows.append(
            {
                "race": r.race,
                "pollster": f"Pollster {chr(65 + int(rng.integers(0, 8)))}",
                "end_date": str(pd.Timestamp("2026-08-01") + pd.Timedelta(days=int(rng.integers(0, 80))))[:10],
                "sample_size": n,
                "population": rng.choice(["LV", "RV"]),
                "dem_pct": round(d, 1),
                "rep_pct": round(100 - und - d, 1),
            }
        )
polls = pd.DataFrame(poll_rows)

# ---------------------------------------------------------------------------
# 11. A simple simulated forecast and 1,000 draws of governorships
# ---------------------------------------------------------------------------
st = races[races["type"] == "state"].reset_index(drop=True)
n_draws = 1000
national = rng.normal(0, 3, n_draws)
state_err = rng.normal(0, 7, (n_draws, len(st)))
margins = st["true_margin"].to_numpy() + national[:, None] + state_err
dem_wins = margins > 0
fixed_d = int((governors_not_up["party"] == "D").sum())
simulations = pd.DataFrame(
    {
        "draw": np.arange(1, n_draws + 1),
        "dem_governors": dem_wins.sum(axis=1) + fixed_d,
    }
)
simulations["rep_governors"] = 50 - simulations["dem_governors"]
inc_d = (st["incumbent_party"] == "D").to_numpy()
simulations["flips"] = (dem_wins != inc_d).sum(axis=1)

share = 50 + margins / 2
forecast = pd.DataFrame(
    {
        "race": st["race"],
        "type": "state",
        "incumbent_party": st["incumbent_party"],
        "dem_share_mean": share.mean(axis=0),
        "dem_share_lo80": np.percentile(share, 10, axis=0),
        "dem_share_hi80": np.percentile(share, 90, axis=0),
        "dem_share_lo95": np.percentile(share, 2.5, axis=0),
        "dem_share_hi95": np.percentile(share, 97.5, axis=0),
        "p_dem_win": dem_wins.mean(axis=0),
    }
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
outputs = {
    "races": races,
    "candidates": candidates,
    "governors_not_up": governors_not_up,
    "history": history,
    "personas": personas,
    "cells": cells,
    "frame": frame,
    "responses_method_a": responses_a,
    "method_a_estimates": method_a,
    "method_b_estimates": method_b,
    "method_c_responses": method_c,
    "polls": polls,
    "forecast": forecast,
    "simulations": simulations,
}
for name, df in outputs.items():
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"{name:<22} {len(df):>8,} rows")
