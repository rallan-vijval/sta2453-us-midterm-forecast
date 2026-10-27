"""
Purpose : Summarise how far the synthetic answers can be trusted, and what they cost.
            1. Validation: for each of the six Method V checks, how often a persona's
               governor vote changes from its Method A answer when one thing in the
               prompt is changed, against how often it changes when simply asked again.
            2. Stability: how much Methods B and C, and the Method V answers, move
               from one day to the next when the prompts are identical.
            3. Costs and timing of every model run.
Author  : Shrey Sati (Member 2)
Date    : 2026-10-27
Inputs  : data/02-analysis_data/responses/*_{A,C,V}_*.parquet   (04-query_models.py)
          data/02-analysis_data/method_b_estimates.parquet       (05b-method_b.py)
          data/02-analysis_data/personas.parquet, candidates.parquet
          data/02-analysis_data/run_log.csv
Outputs : data/03-forecast/validation_summary.csv  one row per run date x model x check
          data/03-forecast/stability_summary.csv   one row per method x model
          data/03-forecast/cost_summary.csv        one row per run date x method x model
Notes   : - Each Method V answer is paired with the same persona's Method A answer
            from repeat 1, which used the same candidate order (order_reverse
            reverses that order). Unparsed answers on either side are left out.
          - The baseline ("asked again") is the share of Method A answers that
            change between repeat 1 and repeat 2 for the same validation personas.
            Repeat 2 also shuffles the candidate order, so the baseline includes
            any order effect; extra_changed_pct is the change beyond it.
          - Each check also reports one statistic, with and without the change:
              attention       named the state they live in correctly
              party_drop      party-line vote (Democrats for the Democrat,
              label_drop      Republicans for the Republican; independents
              governor_only   are left out)
              incumbent_drop  vote for the incumbent, in races where one runs
              order_reverse   chose the candidate listed first
          - Percentages carry a binomial standard error, 100 * sqrt(p(1 - p) / n).
          - Stability: within_day_sd is the spread across repeats asked the same day;
            between_day_sd is the spread of the daily means. The daily runs ask
            Method B once per group, so their daily means also carry some of the
            within-day spread. Method V asks the identical prompt each day, so its
            change against the first day is a pure measure of the models' randomness.
Run     : uv run python scripts/06-validation_checks.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANA, FC, load_candidates, load_responses, norm

# The statistic reported for each check: (label, column with the change,
# column without it). The columns are built in pair_answers().
CHECK_STAT = {
    "attention": ("named their state correctly", "state_ok", None),
    "party_drop": ("party-line vote", "party_line_v", "party_line_a"),
    "incumbent_drop": ("vote for the incumbent", "incumbent_v", "incumbent_a"),
    "label_drop": ("party-line vote", "party_line_v", "party_line_a"),
    "order_reverse": ("chose the first-listed candidate", "first_v", "first_a"),
    "governor_only": ("party-line vote", "party_line_v", "party_line_a"),
}


def pct(x: pd.Series) -> float:
    """Percentage of True values, ignoring missing ones."""
    x = x.dropna().astype(float)
    return 100 * x.mean() if len(x) else np.nan


def pct_se(x: pd.Series) -> float:
    """Binomial standard error of pct(x), in percentage points."""
    x = x.dropna().astype(float)
    if not len(x):
        return np.nan
    p = x.mean()
    return 100 * np.sqrt(p * (1 - p) / len(x))


def first_listed(order_json: str) -> str:
    return json.loads(order_json)[0]


# ---------------------------------------------------------------------------
# 1. Validation checks
# ---------------------------------------------------------------------------
def pair_answers(a: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """One row per Method V answer, beside the same persona's Method A answer
    (repeat 1, same model), with the flags each check's statistic needs."""
    personas = pd.read_parquet(ANA / "personas.parquet")[["persona_id", "party_group"]]
    cands = load_candidates()
    cands = cands[cands["include_in_prompt"]]
    party = cands.set_index(["race", "candidate"])["party_code"].to_dict()
    incumbent = cands[cands["incumbent"]].set_index("race")["candidate"].to_dict()

    base = a[(a["repeat"] == 1) & a["parsed"]]
    base = base[base["run_date"] == base["run_date"].min()]  # the full run
    base = base[["persona_id", "model_slug", "governor_vote", "candidate_order"]]
    pairs = v[v["parsed"]].merge(
        base, on=["persona_id", "model_slug"], suffixes=("_v", "_a")
    )
    pairs = pairs.merge(personas, on="persona_id", how="left")

    pairs["changed"] = pairs["governor_vote_v"] != pairs["governor_vote_a"]
    for side in ["v", "a"]:
        vote = pairs[f"governor_vote_{side}"]
        vote_party = pd.Series(
            [party.get((r, c)) for r, c in zip(pairs["race"], vote)], index=pairs.index
        )
        partisan = pairs["party_group"].isin(["D", "R"])
        pairs[f"dk_{side}"] = vote == "Don't know"
        pairs[f"party_line_{side}"] = (vote_party == pairs["party_group"]).where(
            partisan
        )
        inc = pairs["race"].map(incumbent)
        pairs[f"incumbent_{side}"] = (vote == inc).where(inc.notna())
        first = pairs[f"candidate_order_{side}"].map(first_listed)
        pairs[f"first_{side}"] = vote == first
    pairs["state_ok"] = (
        pairs["stated_state"].map(norm) == pairs["race"].map(norm)
    ).where(pairs["check"] == "attention")
    return pairs


def retest_rate(a: pd.DataFrame, persona_ids) -> pd.DataFrame:
    """Per model: how often a validation persona's Method A vote differs between
    repeat 1 and repeat 2 of the full run (simply asking again)."""
    a = a[a["parsed"] & a["persona_id"].isin(persona_ids)]
    a = a[a["run_date"] == a["run_date"].min()]
    wide = a.pivot_table(
        index=["persona_id", "model_slug"],
        columns="repeat",
        values="governor_vote",
        aggfunc="first",
    ).dropna(subset=[1, 2])
    changed = (wide[1] != wide[2]).rename("changed").reset_index()
    return (
        changed.groupby("model_slug")["changed"]
        .agg(retest_changed_pct=pct, retest_n="size")
        .reset_index()
    )


def validation_summary(a: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    pairs = pair_answers(a, v)
    rows = []
    for (run_date, model, check), g in pairs.groupby(
        ["run_date", "model_slug", "check"]
    ):
        label, with_col, without_col = CHECK_STAT[check]
        rows.append(
            {
                "run_date": run_date,
                "model_slug": model,
                "check": check,
                "n_pairs": len(g),
                "changed_pct": pct(g["changed"]),
                "changed_pct_se": pct_se(g["changed"]),
                "dont_know_pct": pct(g["dk_v"]),
                "dont_know_pct_without_change": pct(g["dk_a"]),
                "statistic": label,
                "statistic_pct": pct(g[with_col]),
                "statistic_pct_se": pct_se(g[with_col]),
                "statistic_pct_without_change": pct(g[without_col])
                if without_col
                else np.nan,
            }
        )
    out = pd.DataFrame(rows).merge(
        retest_rate(a, v["persona_id"].unique()), on="model_slug", how="left"
    )
    out["extra_changed_pct"] = out["changed_pct"] - out["retest_changed_pct"]
    return out


# ---------------------------------------------------------------------------
# 2. Day-to-day stability
# ---------------------------------------------------------------------------
def spread(df: pd.DataFrame, value: str, method: str, quantity: str) -> pd.DataFrame:
    """Per model: the standard deviation across repeats on the same day, and across
    days of the daily means, each averaged over races (points)."""
    within = (
        df.groupby(["model_slug", "race", "run_date"])[value]
        .std()
        .groupby("model_slug")
        .mean()
        .rename("within_day_sd")
    )
    daily = df.groupby(["model_slug", "race", "run_date"])[value].mean()
    between = (
        daily.groupby(["model_slug", "race"])
        .std()
        .groupby("model_slug")
        .mean()
        .rename("between_day_sd")
    )
    n_days = df.groupby("model_slug")["run_date"].nunique().rename("n_days")
    out = pd.concat([n_days, within, between], axis=1).reset_index()
    return out.assign(method=method, quantity=quantity)


def stability_summary(c: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    parts = []
    b = pd.read_parquet(ANA / "method_b_estimates.parquet")
    b["margin"] = (
        100 * (b["dem_share"] - b["rep_share"]) / (b["dem_share"] + b["rep_share"])
    )
    parts.append(spread(b, "margin", "B", "Democratic two-party margin (points)"))
    if len(c):
        c = c[c["parsed"]]
        parts.append(
            spread(c, "dem_two_party_share", "C", "Democratic two-party share (points)")
        )
    # Method V: the identical prompt on a later day against the first day.
    v = v[v["parsed"]]
    first_day = v["run_date"].min()
    first = v[v["run_date"] == first_day][["key", "governor_vote"]]
    later = v[v["run_date"] > first_day].merge(first, on="key", suffixes=("", "_first"))
    if len(later):
        later["changed"] = later["governor_vote"] != later["governor_vote_first"]
        vv = (
            later.groupby("model_slug")
            .agg(
                n_days=("run_date", lambda d: d.nunique() + 1),
                changed_vs_first_day_pct=("changed", pct),
                n_pairs=("changed", "size"),
            )
            .reset_index()
            .assign(method="V", quantity="vote changed vs the first day (%)")
        )
        parts.append(vv)
    cols = ["method", "model_slug", "quantity", "n_days", "within_day_sd",
            "between_day_sd", "changed_vs_first_day_pct", "n_pairs"]  # fmt: skip
    return pd.concat(parts, ignore_index=True).reindex(columns=cols)


# ---------------------------------------------------------------------------
# 3. Costs and timing
# ---------------------------------------------------------------------------
def cost_summary() -> pd.DataFrame:
    log = pd.read_csv(ANA / "run_log.csv")
    log = log[~log["simulated_inputs"].astype(bool)]
    log["parsed_calls"] = log["calls"] * log["parse_rate"]
    log["call_seconds"] = log["calls"] * log["mean_seconds"]
    # Wall-clock time of each run (all models run at once, so count each run once).
    runs = log.drop_duplicates(["run_date", "method", "started_utc"])
    runs = runs.assign(
        minutes=(
            pd.to_datetime(runs["finished_utc"]) - pd.to_datetime(runs["started_utc"])
        ).dt.total_seconds()
        / 60
    )
    wall = runs.groupby(["run_date", "method"])["minutes"].sum().rename("wall_minutes")
    out = (
        log.groupby(["run_date", "method", "model_slug"])
        .agg(
            calls=("calls", "sum"),
            parsed_calls=("parsed_calls", "sum"),
            prompt_tokens=("prompt_tokens", "sum"),
            completion_tokens=("completion_tokens", "sum"),
            reasoning_tokens=("reasoning_tokens", "sum"),
            cost_usd=("cost_usd", "sum"),
            call_seconds=("call_seconds", "sum"),
        )
        .reset_index()
        .merge(wall.reset_index(), on=["run_date", "method"], how="left")
    )
    out["parse_rate"] = out["parsed_calls"] / out["calls"]
    out["mean_seconds_per_call"] = out["call_seconds"] / out["calls"]
    out["cost_per_1000_calls_usd"] = 1000 * out["cost_usd"] / out["calls"]
    return out.drop(columns=["parsed_calls", "call_seconds"])


def main():
    a, v, c = load_responses("A"), load_responses("V"), load_responses("C")
    if a.empty or v.empty:
        raise SystemExit("06 needs Method A and Method V responses: run 04 first.")
    for d in (a, v, c):
        if len(d):
            d["parsed"] = d["parsed"].astype(bool)

    val = validation_summary(a, v)
    stab = stability_summary(c, v)
    cost = cost_summary()
    FC.mkdir(parents=True, exist_ok=True)
    val.to_csv(FC / "validation_summary.csv", index=False)
    stab.to_csv(FC / "stability_summary.csv", index=False)
    cost.to_csv(FC / "cost_summary.csv", index=False)

    first = val[val["run_date"] == val["run_date"].min()]
    print("Validation, first run: % of votes that changed (asked again: retest)")
    print(
        first.pivot_table(index="check", columns="model_slug", values="changed_pct")
        .round(1)
        .to_string()
    )
    print(
        "\nRetest:",
        first.groupby("model_slug")["retest_changed_pct"].first().round(1).to_dict(),
    )
    print("\nStability:\n" + stab.round(2).to_string(index=False))
    by_day = cost.groupby("run_date")[["calls", "cost_usd"]].sum()
    print("\nCost by run date (US$):\n" + by_day.round(3).to_string())
    print(f"Total: {cost['calls'].sum():,} calls, US${cost['cost_usd'].sum():.2f}")
    print(
        f"Saved validation_summary.csv, stability_summary.csv, cost_summary.csv to {FC}"
    )


if __name__ == "__main__":
    main()