"""
Purpose : Turn everything into one forecast for every 2026 governor race.
            1. Fundamentals (Method C, part 1): a regression of past governor
               margins on state lean (Cook PVI), incumbency and the midterm
               penalty, fitted on the 2014, 2018 and 2022 races.
            2. Silicon: Methods A and B from every member's models, with a
               bias correction learned from races that have polls.
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
            data/02-analysis_data/responses/*_C_*.parquet     (territories; optional)
          --simulated : the matching files in data/00-simulated_data/
Outputs : data/03-forecast/forecast.parquet     one row per race
          data/03-forecast/simulations.parquet  one row per simulated election
          data/03-forecast/method_grid.parquet  race x model set x method
          data/03-forecast/backtest.csv         cross-validated fundamentals accuracy
          data/03-forecast/run_info.json        settings and fitted error sizes
Notes   : Margins are Democratic minus Republican, in points of the two-party
          vote. Third-party candidates are not modelled separately.
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
DEFAULT_SILICON_SD = 10.0  # silicon error when too few polled races to estimate it
TERRITORY_SD = 15.0      # territories have no personas, so wide uncertainty
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
    est["margin"] = two_party_margin(est["dem_share"], est["rep_share"])
    # Weight each estimate by how many personas (A) or groups (B) it rests on, so a
    # small daily validation run counts less than a full run.
    n = est.get("n_personas", pd.Series(np.nan, index=est.index))
    n = n.fillna(est.get("n_cells", pd.Series(np.nan, index=est.index))).fillna(1)
    est["w"] = n
    est["wm"] = est["margin"] * est["w"]
    g = est.groupby(["race", "method", "model_set"], as_index=False)[["wm", "w"]].sum()
    g["margin"] = g["wm"] / g["w"]
    return g[["race", "method", "model_set", "margin"]]


# ---------------------------------------------------------------------------
# 3. Polls
# ---------------------------------------------------------------------------
def poll_average(polls: pd.DataFrame) -> pd.DataFrame:
    if polls.empty:
        return pd.DataFrame(columns=["race", "poll_margin", "poll_var", "n_polls"])
    p = polls.dropna(subset=["dem_pct", "rep_pct", "sample_size"]).copy()
    p["end_date"] = pd.to_datetime(p["end_date"])
    two = p["dem_pct"] + p["rep_pct"]
    share = p["dem_pct"] / two
    n_eff = p["sample_size"] * two / 100
    p["margin"] = 100 * (2 * share - 1)
    p["var"] = 1e4 * 4 * share * (1 - share) / n_eff
    age = (p["end_date"].max() - p["end_date"]).dt.days
    p["w"] = 0.5 ** (age / POLL_HALF_LIFE) / p["var"]
    out = p.groupby("race").apply(
        lambda d: pd.Series({
            "poll_margin": np.average(d["margin"], weights=d["w"]),
            # variance of a weighted mean, plus error that sample size misses
            "poll_var": (d["w"] ** 2 * d["var"]).sum() / d["w"].sum() ** 2 + POLL_FLOOR_SD**2,
            "n_polls": len(d),
        }),
        include_groups=False,
    ).reset_index()
    return out


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
    sil_race = sil.groupby("race")["margin"].agg(sil_margin="mean", sil_between_sd="std").reset_index()

    # 3. Polls, and the silicon error learned from polled races
    polls = poll_average(d["polls"])
    cal = sil_race.merge(polls, on="race")
    if len(cal) >= 3:
        diff = cal["sil_margin"] - cal["poll_margin"]
        sil_bias = float(diff.mean())
        sil_sd = float(np.sqrt(max(((diff - sil_bias) ** 2).mean() - cal["poll_var"].mean(), MIN_SILICON_SD**2)))
    else:
        sil_bias, sil_sd = 0.0, DEFAULT_SILICON_SD
    print(f"Silicon: bias {sil_bias:+.1f}, error SD {sil_sd:.1f} (from {len(cal)} polled races)")

    # Per-race combination, overall and for each method x model set.
    st = races[races["type"] == "state"].merge(fund, on="race").merge(sil_race, on="race", how="left")
    st = st.merge(polls, on="race", how="left")
    st["n_polls"] = st["n_polls"].fillna(0).astype(int)
    rows = []
    for _, r in st.iterrows():
        sil_var = (0 if np.isnan(r.sil_between_sd) else r.sil_between_sd**2) + sil_sd**2
        m, v = combine([r.fund_margin, r.sil_margin - sil_bias, r.poll_margin],
                       [r.fund_var, sil_var, r.poll_var])
        rows.append({"race": r.race, "margin_mean": m, "margin_sd": np.sqrt(v)})
    st = st.merge(pd.DataFrame(rows), on="race")

    combos = sil[["method", "model_set"]].drop_duplicates().reset_index(drop=True)
    grid = []
    for _, cb in combos.iterrows():
        s = sil[(sil.method == cb.method) & (sil.model_set == cb.model_set)][["race", "margin"]]
        g = st[["race", "fund_margin", "fund_var", "poll_margin", "poll_var"]].merge(s, on="race", how="left")
        for _, r in g.iterrows():
            m, v = combine([r.fund_margin, r.margin - sil_bias, r.poll_margin],
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

    dem_win = margins > 0
    inc = st["incumbent_party"].to_numpy()
    winner = np.where(dem_win, "D", "R")
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
        "dem_governors": dem_win.sum(axis=1) + fixed_d,
        "rep_governors": (~dem_win).sum(axis=1) + fixed_r,
        "other_governors": fixed_o,
        "flips": flips,
    })
    assert (sims[["dem_governors", "rep_governors", "other_governors"]].sum(axis=1) == 50).all()

    share = 50 + margins / 2
    q = lambda p: np.percentile(share, p, axis=0)
    forecast = st[["race", "type", "incumbent_party", "open_seat", "competitive", "pvi",
                   "fund_margin", "sil_margin", "poll_margin", "n_polls"]].copy()
    forecast["silicon_margin_corrected"] = forecast["sil_margin"] - sil_bias
    forecast = forecast.assign(
        margin_mean=margins.mean(axis=0), margin_sd=margins.std(axis=0),
        dem_share_mean=share.mean(axis=0),
        dem_share_lo80=q(10), dem_share_hi80=q(90), dem_share_lo95=q(2.5), dem_share_hi95=q(97.5),
        p_dem_win=dem_win.mean(axis=0),
        p_flip=(winner != inc[None, :]).mean(axis=0),
    )

    # Territories: no personas, so the direct "forecaster" answers, wide error.
    terr = races[races["type"] == "territory"].copy()
    if len(terr):
        c = d["c"]
        direct = (
            c.assign(margin=2 * c["dem_two_party_share"] - 100).groupby("race")["margin"].mean()
            if not c.empty else pd.Series(dtype=float)
        )
        tm = terr["race"].map(direct)
        if "last_race_pct" in terr:
            fallback = np.where(terr["last_race_party"] == "D", 2 * terr["last_race_pct"] - 100,
                                100 - 2 * terr["last_race_pct"])
            tm = tm.fillna(pd.Series(fallback, index=terr.index))
        tm = tm.fillna(0).to_numpy()
        tshare = 50 + tm / 2
        forecast = pd.concat([forecast, pd.DataFrame({
            "race": terr["race"], "type": "territory", "incumbent_party": terr["incumbent_party"],
            "open_seat": terr["open_seat"], "competitive": False, "margin_mean": tm,
            "margin_sd": TERRITORY_SD, "dem_share_mean": tshare,
            "dem_share_lo80": tshare - 1.2816 * TERRITORY_SD / 2, "dem_share_hi80": tshare + 1.2816 * TERRITORY_SD / 2,
            "dem_share_lo95": tshare - 1.96 * TERRITORY_SD / 2, "dem_share_hi95": tshare + 1.96 * TERRITORY_SD / 2,
            "p_dem_win": norm.cdf(tm / TERRITORY_SD),
        })], ignore_index=True)

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
        "fundamentals_residual_sd": resid_sd, "silicon_bias": sil_bias, "silicon_sd": sil_sd,
        "polled_races_used_for_calibration": int(len(cal)),
        "method_model_combinations": int(len(gm)),
    }
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Democratic governors: median {sims.dem_governors.median():.0f}, "
          f"80% range {sims.dem_governors.quantile(.1):.0f}-{sims.dem_governors.quantile(.9):.0f}")
    print("Saved to", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated", action="store_true", help="use data/00-simulated_data inputs")
    main(parser.parse_args().simulated)
