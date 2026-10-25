"""
Purpose : Turn everything into one forecast for every 2026 governor race.
            1. Fundamentals (Method C, part 1): a regression of past governor
               margins on state lean (Cook PVI), incumbency and the midterm
               penalty, fitted on the 2014, 2018 and 2022 races.
            2. Silicon: Methods A and B from every model, calibrated against
               the races that have polls (poll = a + b x silicon).
            3. Polls: a recency- and size-weighted average where polls exist.
            4. The three are combined by inverse-variance weighting, then the
               election is simulated 10,000 times with a shared national swing.
Author  : Allwin Steve Paul Justin Anantharaj (Member 3)
Date    : 2026-10-22
Inputs  : real mode (default)
            data/01-raw_data/governor_races_2026.csv
            data/01-raw_data/governor_results_history.csv
            data/01-raw_data/governors_not_up_2026.csv
            data/01-raw_data/governor_polls_2026.csv          (optional)
            data/02-analysis_data/method_a_estimates.parquet
            data/02-analysis_data/method_b_estimates.parquet
            data/02-analysis_data/candidates.parquet          (names on the ballot)
          --simulated : the matching files in data/00-simulated_data/
Outputs : data/03-forecast/forecast.parquet     one row per race
          data/03-forecast/simulations.parquet  one row per simulated election
          data/03-forecast/method_grid.parquet  race x model set x method
          data/03-forecast/backtest.csv         cross-validated fundamentals accuracy
          data/03-forecast/run_info.json        settings, fitted error sizes, headline numbers
          data/03-forecast/race_summary.csv     readable table, closest races first
Notes   : Margins are Democratic minus Republican, in points of the two-party
          vote. Votes are counted by party: Alaska's three Republicans add up to
          one Republican share (a stand-in for its ranked-choice final round),
          and independents and minor parties (e.g. Rick Bennett in Maine) form a
          third "other" share, taken from the AI surveys. In each simulated
          election the largest of the three shares wins (plurality), so an
          independent can win. Georgia needs a majority: when no one passes 50%
          the race goes to a runoff, reported as p_runoff; the winner shown for
          Georgia is the first-round leader.
Run     : uv run python scripts/05-forecast.py [--simulated]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "03-forecast"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 853
N_SIMS = 10_000
NATIONAL_SD = 4.0        # assumed spread of the shared national swing (margin points)
POLL_FLOOR_SD = 5.0      # extra polling error that sample size does not capture
MIN_LOCAL_SD = 5.0       # floor on each race's own error: the three sources share some errors,
                         # so treating them as independent would otherwise be overconfident
MIN_SILICON_SD = 5.0     # floor on the silicon error estimated from polled races
POLL_HALF_LIFE = 30      # days: a poll a month older counts half as much
POLL_START = pd.Timestamp("2026-08-01")     # polls ending earlier are dropped (some go back to 2023)
FORECAST_DATE = pd.Timestamp("2026-11-03")  # poll age is measured to election day
MIN_RACES_FOR_SLOPE = 8   # polled races needed to fit a calibration slope as well as a shift
DEFAULT_SILICON_SD = 10.0  # silicon error when too few polled races to estimate it

OTHER_MIN_SD = 3.0       # floor on the error of the third-party share (points of the total vote)
OTHER_REL_SD = 0.2       # ... and at least 20% of the share itself (a 35% independent: +/- 7)
MAX_OTHER = 90.0         # cap on a simulated third-party share
RUNOFF_STATES = {"Georgia"}  # a candidate needs over 50%, otherwise there is a runoff
PRES_DEM = {2014: 1, 2018: -1, 2022: 1, 2026: -1}  # +1 when the president is a Democrat


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(simulated: bool) -> dict:
    if simulated:
        d = ROOT / "data" / "00-simulated_data"
        not_up = pd.read_parquet(d / "governors_not_up.parquet")
        return {
            "races": pd.read_parquet(d / "races.parquet"),
            "history": pd.read_parquet(d / "history.parquet"),
            "not_up": not_up,
            "polls": pd.read_parquet(d / "polls.parquet"),
            "a": pd.read_parquet(d / "method_a_estimates.parquet"),
            "b": pd.read_parquet(d / "method_b_estimates.parquet"),
            "c": pd.read_parquet(d / "method_c_responses.parquet"),
            "candidates": pd.read_parquet(d / "candidates.parquet").rename(columns={"party": "party_code"}),
        }
    raw, ana = ROOT / "data" / "01-raw_data", ROOT / "data" / "02-analysis_data"
    polls_path = raw / "governor_polls_2026.csv"
    c_files = sorted((ana / "responses").glob("*_C_*.parquet"))
    c = pd.concat([pd.read_parquet(f) for f in c_files], ignore_index=True) if c_files else pd.DataFrame()
    if not c.empty:
        c = c[c["parsed"].astype(bool)]
    return {
        "races": pd.read_csv(raw / "governor_races_2026.csv"),
        "history": pd.read_csv(raw / "governor_results_history.csv"),
        "not_up": pd.read_csv(raw / "governors_not_up_2026.csv"),
        "polls": pd.read_csv(polls_path) if polls_path.exists() else pd.DataFrame(),
        "a": pd.read_parquet(ana / "method_a_estimates.parquet"),
        "b": pd.read_parquet(ana / "method_b_estimates.parquet"),
        "c": c,
        "candidates": pd.read_parquet(ana / "candidates.parquet"),
    }


def two_party_margin(dem, rep):
    dem, rep = np.asarray(dem, float), np.asarray(rep, float)
    return 100 * (dem - rep) / (dem + rep)


# ---------------------------------------------------------------------------
# 1. Fundamentals
# ---------------------------------------------------------------------------
def design(df: pd.DataFrame, year_col_value=None) -> np.ndarray:
    """Columns: intercept, PVI, incumbent running (+1 D / -1 R / 0), president's party."""
    inc = np.where(
        df["incumbent_ran"].astype(bool),
        np.where(df["incumbent_party"] == "D", 1, np.where(df["incumbent_party"] == "R", -1, 0)),
        0,
    )
    pres = df["year"].map(PRES_DEM) if year_col_value is None else np.full(len(df), year_col_value)
    return np.column_stack([np.ones(len(df)), df["pvi"], inc, pres])


def fit_ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma = np.sqrt(resid @ resid / (len(y) - X.shape[1]))
    xtx_inv = np.linalg.pinv(X.T @ X)
    return beta, sigma, xtx_inv


def fundamentals(history: pd.DataFrame, races: pd.DataFrame):
    h = history.dropna(subset=["pvi"]).copy()
    h = h[(h["dem_pct"] > 0) & (h["rep_pct"] > 0)]  # drop uncontested races
    h["margin"] = two_party_margin(h["dem_pct"], h["rep_pct"])
    X, y = design(h), h["margin"].to_numpy()
    beta, sigma, xtx_inv = fit_ols(X, y)

    st = races[races["type"] == "state"].copy()
    st["year"] = 2026
    st["incumbent_ran"] = ~st["open_seat"].astype(bool)
    Xn = design(st, PRES_DEM[2026])
    mean = Xn @ beta
    # Prediction variance = residual variance + uncertainty in the coefficients.
    var = sigma**2 * (1 + np.einsum("ij,jk,ik->i", Xn, xtx_inv, Xn))
    coefs = pd.DataFrame({
        "term": ["intercept", "pvi", "incumbent_running", "president_democrat"],
        "estimate": beta,
        "std_error": sigma * np.sqrt(np.diag(xtx_inv)),
    })
    return st[["race"]].assign(fund_margin=mean, fund_var=var), coefs, sigma, h


def backtest(h: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Cross-validation grouped by state: each fold holds out about 6 states in every year."""
    rng = np.random.default_rng(SEED)
    states = h["race"].unique()
    fold = dict(zip(states, rng.permutation(np.arange(len(states)) % k)))
    preds = []
    for f in range(k):
        test = h["race"].map(fold) == f
        beta, sigma, _ = fit_ols(design(h[~test]), h.loc[~test, "margin"].to_numpy())
        mu = design(h[test]) @ beta
        preds.append(h[test].assign(pred=mu, p_dem=norm.cdf(mu / sigma)))
    p = pd.concat(preds)
    won = (p["margin"] > 0).astype(float)
    eps = 1e-6
    return pd.DataFrame([{
        "races": len(p),
        "rmse_margin": float(np.sqrt(((p["margin"] - p["pred"]) ** 2).mean())),
        "brier": float(((p["p_dem"] - won) ** 2).mean()),
        "log_score": float(np.mean(won * np.log(p["p_dem"] + eps) + (1 - won) * np.log(1 - p["p_dem"] + eps))),
        "winner_correct": float(((p["pred"] > 0) == (p["margin"] > 0)).mean()),
    }])


# ---------------------------------------------------------------------------
# 2. Silicon (Methods A and B)
# ---------------------------------------------------------------------------
def silicon(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """One estimate per race x method x model set (averaged over models and repeats)."""
    est = pd.concat([a.assign(method="A"), b.assign(method="B")], ignore_index=True)
    est = est.dropna(subset=["dem_share", "rep_share"])
    if "model_slug" in est:  # one estimate per model, so the simulation can draw each model
        est["model_set"] = est["model_slug"]
    est["margin"] = two_party_margin(est["dem_share"], est["rep_share"])
    # Weight each estimate by how many personas (A) or groups (B) it rests on, so a
    # small daily validation run counts less than a full run.
    n = est.get("n_personas", pd.Series(np.nan, index=est.index))
    n = n.fillna(est.get("n_cells", pd.Series(np.nan, index=est.index))).fillna(1)
    est["w"] = n
    est["wm"] = est["margin"] * est["w"]
    other = est.get("other_share", pd.Series(np.nan, index=est.index))
    est["wo"] = other.fillna(100 - est["dem_share"] - est["rep_share"]) * est["w"]
    g = est.groupby(["race", "method", "model_set"], as_index=False)[["wm", "wo", "w"]].sum()
    g["margin"] = g["wm"] / g["w"]
    g["other_share"] = g["wo"] / g["w"]
    return g[["race", "method", "model_set", "margin", "other_share"]]


# ---------------------------------------------------------------------------
# 3. Polls
# ---------------------------------------------------------------------------
def poll_average(polls: pd.DataFrame) -> pd.DataFrame:
    if polls.empty:
        return pd.DataFrame(columns=["race", "poll_margin", "poll_var", "n_polls", "poll_other", "n_polls_other"])
    p = polls.dropna(subset=["dem_pct", "rep_pct", "end_date"]).copy()
    p["end_date"] = pd.to_datetime(p["end_date"])
    p = p[p["end_date"] >= POLL_START]
    # A few polls report no sample size: give them the median of the polls kept.
    p["sample_size"] = p["sample_size"].fillna(p["sample_size"].median())
    two = p["dem_pct"] + p["rep_pct"]
    share = p["dem_pct"] / two
    n_eff = p["sample_size"] * two / 100
    p["margin"] = 100 * (2 * share - 1)
    p["var"] = 1e4 * 4 * share * (1 - share) / n_eff
    age = (FORECAST_DATE - p["end_date"]).dt.days.clip(lower=0)
    p["w"] = 0.5 ** (age / POLL_HALF_LIFE) / p["var"]
    # Third-party share of the decided vote, only from polls that name other candidates
    # or offer "other" (a poll listing just the two nominees says nothing about it).
    extra_cols = [c for c in ["minor_pct", "other_pct"] if c in p.columns]
    if extra_cols:
        extra = p[extra_cols].fillna(0).sum(axis=1)
        p["other"] = np.where(p[extra_cols].notna().any(axis=1), 100 * extra / (two + extra), np.nan)
    else:
        p["other"] = np.nan

    def summary(d):
        o = d.dropna(subset=["other"])
        return pd.Series({
            "poll_margin": np.average(d["margin"], weights=d["w"]),
            # variance of a weighted mean, plus error that sample size misses
            "poll_var": (d["w"] ** 2 * d["var"]).sum() / d["w"].sum() ** 2 + POLL_FLOOR_SD**2,
            "n_polls": len(d),
            "poll_other": np.average(o["other"], weights=o["w"]) if len(o) else np.nan,
            "n_polls_other": len(o),
        })

    return p.groupby("race").apply(summary, include_groups=False).reset_index()


# ---------------------------------------------------------------------------
# 4. Combine and simulate
# ---------------------------------------------------------------------------
def combine(means, variances):
    """Inverse-variance weighted average of independent estimates (NaNs ignored)."""
    m, v = np.asarray(means, float), np.asarray(variances, float)
    ok = ~np.isnan(m) & ~np.isnan(v)
    w = 1 / v[ok]
    return float((w * m[ok]).sum() / w.sum()), float(1 / w.sum())


def total_sd(var):
    """National part plus a race's own part, the latter never below MIN_LOCAL_SD."""
    local = np.sqrt(np.maximum(np.asarray(var, float) - NATIONAL_SD**2, MIN_LOCAL_SD**2))
    return np.sqrt(local**2 + NATIONAL_SD**2)


def main(simulated: bool):
    rng = np.random.default_rng(SEED)
    d = load(simulated)
    races = d["races"].copy()
    races["open_seat"] = races["open_seat"].astype(bool)

    # 1. Fundamentals
    fund, coefs, resid_sd, hist = fundamentals(d["history"], races)
    bt = backtest(hist)
    print("Fundamentals:", ", ".join(f"{t} {e:+.2f}" for t, e in zip(coefs.term, coefs.estimate)),
          f"| residual SD {resid_sd:.1f}")
    print("Backtest:", bt.round(3).to_dict("records")[0])

    # 2. Silicon, by race x method x model set
    sil = silicon(d["a"], d["b"])
    sil_race = sil.groupby("race").agg(sil_margin=("margin", "mean"), sil_between_sd=("margin", "std"),
                                       other_mean=("other_share", "mean"), other_between_sd=("other_share", "std"),
                                       ).reset_index()

    # 3. Polls, and the silicon error learned from polled races
    polls = poll_average(d["polls"])
    print(f"Polls: {int(polls['n_polls'].sum()) if len(polls) else 0} from {POLL_START:%d %b %Y} on, "
          f"in {len(polls)} races")
    # Calibrate the AI surveys against the polls: poll = a + b * silicon. A slope b
    # below 1 means the models exaggerate how partisan safe states are; with few
    # polled races only the average gap (b = 1) is estimated.
    cal = sil_race.merge(polls, on="race")
    if len(cal) >= MIN_RACES_FOR_SLOPE:
        cal_b, cal_a = np.polyfit(cal["sil_margin"], cal["poll_margin"], 1)
    elif len(cal) >= 3:
        cal_b, cal_a = 1.0, float((cal["poll_margin"] - cal["sil_margin"]).mean())
    else:
        cal_b, cal_a = 1.0, 0.0
    calibrate = lambda m: cal_a + cal_b * m  # noqa: E731
    if len(cal) >= 3:
        resid = cal["poll_margin"] - calibrate(cal["sil_margin"])
        sil_sd = float(np.sqrt(max((resid**2).mean() - cal["poll_var"].mean(), MIN_SILICON_SD**2)))
    else:
        sil_sd = DEFAULT_SILICON_SD
    sil_bias = float((cal["sil_margin"] - cal["poll_margin"]).mean()) if len(cal) else 0.0
    print(f"Silicon: average gap to polls {sil_bias:+.1f}; calibration poll = {cal_a:+.1f} + {cal_b:.2f} x silicon; "
          f"error SD {sil_sd:.1f} (from {len(cal)} polled races)")

    # Per-race combination, overall and for each method x model set.
    st = races[races["type"] == "state"].merge(fund, on="race").merge(sil_race, on="race", how="left")
    st = st.merge(polls, on="race", how="left")
    st["n_polls"] = st["n_polls"].fillna(0).astype(int)
    rows = []
    for _, r in st.iterrows():
        sil_var = (0 if np.isnan(r.sil_between_sd) else r.sil_between_sd**2) + sil_sd**2
        means = [r.fund_margin, calibrate(r.sil_margin), r.poll_margin]
        variances = [r.fund_var, sil_var, r.poll_var]
        m, v = combine(means, variances)
        # How much each source counts in this race (inverse-variance weights, sum to 1).
        w = np.array([0.0 if np.isnan(x) or np.isnan(y) else 1 / y for x, y in zip(means, variances)])
        w = w / w.sum()
        rows.append({"race": r.race, "margin_mean": m, "margin_sd": np.sqrt(v),
                     "weight_fundamentals": w[0], "weight_silicon": w[1], "weight_polls": w[2]})
    st = st.merge(pd.DataFrame(rows), on="race")

    combos = sil[["method", "model_set"]].drop_duplicates().reset_index(drop=True)
    grid = []
    for _, cb in combos.iterrows():
        s = sil[(sil.method == cb.method) & (sil.model_set == cb.model_set)][["race", "margin"]]
        g = st[["race", "fund_margin", "fund_var", "poll_margin", "poll_var"]].merge(s, on="race", how="left")
        for _, r in g.iterrows():
            m, v = combine([r.fund_margin, calibrate(r.margin), r.poll_margin],
                           [r.fund_var, sil_sd**2, r.poll_var])
            sd = float(total_sd(v))
            grid.append({"race": r.race, "method": cb.method, "model_set": cb.model_set,
                         "silicon_margin": r.margin, "combined_margin": m, "combined_sd": sd,
                         "p_dem_win": float(norm.cdf(m / sd))})
    grid = pd.DataFrame(grid)

    # Simulate: each draw uses one method x model set, a shared national swing,
    # and independent state errors.
    races_order = st["race"].tolist()
    gm = grid.pivot_table(index=["method", "model_set"], columns="race", values="combined_margin")[races_order]
    gs = grid.pivot_table(index=["method", "model_set"], columns="race", values="combined_sd")[races_order]
    pick = rng.integers(0, len(gm), N_SIMS)
    swing = rng.normal(0, NATIONAL_SD, N_SIMS)
    local_sd = np.sqrt(gs.to_numpy()[pick] ** 2 - NATIONAL_SD**2)
    margins = gm.to_numpy()[pick] + swing[:, None] + rng.normal(0, 1, local_sd.shape) * local_sd

    # Third-party share of the total vote, then plurality: the largest share wins.
    # The third-party share comes from polls that report it, otherwise from the AI surveys.
    st["other_source"] = np.where(st["poll_other"].notna(), "polls", "AI surveys")
    st["other_estimate"] = st["poll_other"].fillna(st["other_mean"]).fillna(0).clip(lower=0)
    spread = np.where(st["poll_other"].notna(), 0, st["other_between_sd"].fillna(0))
    st["other_sd"] = np.maximum.reduce([spread, OTHER_MIN_SD + 0 * spread, OTHER_REL_SD * st["other_estimate"]])
    other_mean = st["other_estimate"].to_numpy()
    other_sd = st["other_sd"].to_numpy()
    other = np.clip(other_mean[None, :] + rng.normal(0, 1, margins.shape) * other_sd[None, :], 0, MAX_OTHER)
    vote_d = (100 - other) * (50 + margins / 2) / 100
    vote_r = (100 - other) - vote_d
    votes = np.stack([vote_d, vote_r, other])            # party x draw x race
    winner = np.array(["D", "R", "O"])[votes.argmax(axis=0)]
    runoff = np.isin(st["race"].to_numpy(), list(RUNOFF_STATES))[None, :] & (votes.max(axis=0) <= 50)
    inc = st["incumbent_party"].to_numpy()
    flips = (winner != inc[None, :]).sum(axis=1)
    not_up = d["not_up"]
    fixed_d = int((not_up["party"] == "D").sum())
    fixed_r = int((not_up["party"] == "R").sum())
    fixed_o = len(not_up) - fixed_d - fixed_r
    sims = pd.DataFrame({
        "draw": np.arange(1, N_SIMS + 1),
        "method": gm.index.get_level_values(0)[pick],
        "model_set": gm.index.get_level_values(1)[pick],
        "national_swing": swing,
        "dem_governors": (winner == "D").sum(axis=1) + fixed_d,
        "rep_governors": (winner == "R").sum(axis=1) + fixed_r,
        "other_governors": (winner == "O").sum(axis=1) + fixed_o,
        "flips": flips,
    })
    assert (sims[["dem_governors", "rep_governors", "other_governors"]].sum(axis=1) == 50).all()

    share = 50 + margins / 2
    q = lambda p: np.percentile(share, p, axis=0)
    # Who is on the ballot, by party (only the candidates named in the prompts).
    cands = d["candidates"]
    if "include_in_prompt" in cands:
        cands = cands[cands["include_in_prompt"].astype(bool)]
    names = {code: cands[cands["party_code"] == code].groupby("race")["candidate"].agg(" / ".join)
             for code in ["D", "R", "O"]}
    by_method = sil.groupby(["race", "method"])["margin"].mean().unstack()
    by_method.columns = [f"silicon_margin_{m}" for m in by_method.columns]
    qm = lambda p: np.percentile(margins, p, axis=0)  # noqa: E731

    forecast = st[["race", "type", "incumbent_party", "open_seat", "competitive", "pvi",
                   "fund_margin", "sil_margin", "poll_margin", "n_polls", "n_polls_other",
                   "weight_fundamentals", "weight_silicon", "weight_polls",
                   "other_estimate", "other_source"]].copy()
    forecast.insert(1, "dem_candidate", forecast["race"].map(names["D"]))
    forecast.insert(2, "rep_candidate", forecast["race"].map(names["R"]))
    forecast.insert(3, "other_candidates", forecast["race"].map(names["O"]))
    forecast["silicon_margin_corrected"] = calibrate(forecast["sil_margin"])
    forecast = forecast.merge(by_method, left_on="race", right_index=True, how="left")
    forecast = forecast.assign(
        margin_mean=margins.mean(axis=0), margin_sd=margins.std(axis=0),
        margin_lo80=qm(10), margin_hi80=qm(90), margin_lo95=qm(2.5), margin_hi95=qm(97.5),
        dem_share_mean=share.mean(axis=0),
        dem_share_lo80=q(10), dem_share_hi80=q(90), dem_share_lo95=q(2.5), dem_share_hi95=q(97.5),
        dem_vote_mean=vote_d.mean(axis=0), rep_vote_mean=vote_r.mean(axis=0), other_vote_mean=other.mean(axis=0),
        p_dem_win=(winner == "D").mean(axis=0),
        p_rep_win=(winner == "R").mean(axis=0),
        p_other_win=(winner == "O").mean(axis=0),
        p_runoff=np.where(np.isin(st["race"], list(RUNOFF_STATES)), runoff.mean(axis=0), np.nan),
        p_flip=(winner != inc[None, :]).mean(axis=0),
    )

    probs = forecast[["p_dem_win", "p_rep_win", "p_other_win"]].to_numpy()
    forecast["favourite"] = np.array(["D", "R", "O"])[probs.argmax(axis=1)]
    forecast["p_favourite"] = probs.max(axis=1)
    forecast["rating"] = pd.cut(forecast["p_favourite"], [0, 0.6, 0.75, 0.9, 1.0001],
                                labels=["Toss-up", "Lean", "Likely", "Solid"], right=False).astype(str)
    forecast.loc[forecast["rating"] != "Toss-up", "rating"] += " " + forecast["favourite"]
    forecast["is_simulated"] = simulated
    sims["is_simulated"] = simulated
    grid["is_simulated"] = simulated

    forecast.to_parquet(OUT / "forecast.parquet", index=False)
    sims.to_parquet(OUT / "simulations.parquet", index=False)
    grid.to_parquet(OUT / "method_grid.parquet", index=False)
    bt.assign(is_simulated=simulated).to_csv(OUT / "backtest.csv", index=False)
    coefs.to_csv(OUT / "fundamentals_coefficients.csv", index=False)
    info = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulated_inputs": simulated, "n_simulations": N_SIMS, "seed": SEED,
        "national_sd": NATIONAL_SD, "poll_floor_sd": POLL_FLOOR_SD, "min_local_sd": MIN_LOCAL_SD,
        "fundamentals_residual_sd": resid_sd, "silicon_gap_to_polls": sil_bias, "calibration_intercept": float(cal_a),
        "calibration_slope": float(cal_b), "silicon_sd": sil_sd,
        "polled_races_used_for_calibration": int(len(cal)),
        "method_model_combinations": int(len(gm)),
        "polls_used": int(polls["n_polls"].sum()) if len(polls) else 0, "poll_start": str(POLL_START.date()),
        "dem_governors_median": float(sims["dem_governors"].median()),
        "dem_governors_80": [float(sims["dem_governors"].quantile(0.1)), float(sims["dem_governors"].quantile(0.9))],
        "p_dem_majority": float((sims["dem_governors"] >= 26).mean()),
        "p_rep_majority": float((sims["rep_governors"] >= 26).mean()),
        "flips_median": float(sims["flips"].median()),
    }
    # A readable table, closest races first.
    summary = forecast.assign(closeness=(forecast["p_favourite"] - 0.5).abs()).sort_values("closeness")
    summary[["race", "rating", "dem_candidate", "rep_candidate", "other_candidates", "dem_vote_mean",
             "rep_vote_mean", "other_vote_mean", "p_dem_win", "p_rep_win", "p_other_win", "p_runoff", "p_flip",
             "margin_mean", "margin_lo80", "margin_hi80", "fund_margin", "silicon_margin_corrected",
             "poll_margin", "n_polls", "weight_fundamentals", "weight_silicon", "weight_polls"]].round(3).to_csv(
        OUT / "race_summary.csv", index=False)
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Democratic governors: median {sims.dem_governors.median():.0f}, "
          f"80% range {sims.dem_governors.quantile(.1):.0f}-{sims.dem_governors.quantile(.9):.0f}; "
          f"majority: D {info['p_dem_majority']:.0%}, R {info['p_rep_majority']:.0%}")
    print("Ratings:", forecast["rating"].value_counts().to_dict())
    print("Saved to", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated", action="store_true", help="use data/00-simulated_data inputs")
    main(parser.parse_args().simulated)
