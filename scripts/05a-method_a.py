"""
Purpose : Method A. Turn individual persona answers into a vote share for each
          race, separately for every model, repeat and run date.
Author  : Vijval Rallan (Member 1)
Date    : 2026-10-22
Inputs  : data/02-analysis_data/responses/*_A_*.parquet   (every member, every day)
          data/02-analysis_data/personas.parquet, frame.parquet, candidates.parquet
          --simulated : the matching files in data/00-simulated_data/
Outputs : data/02-analysis_data/method_a_estimates.parquet
          data/02-analysis_data/ticket_splitting.csv
          (with --simulated: *_05a versions in data/00-simulated_data/)
Notes   : 1. Likely-voter weight from the turnout answer: definitely 1.0,
             probably 0.7, unlikely 0.2, will not 0.
          2. Within each group (party x age x degree) the weighted shares
             for the Democrat, Republican and others are computed; "Don't
             know" answers count toward turnout but not toward any candidate.
          3. Groups are weighted by their share of the state's registered
             voters (frame.parquet) times their turnout rate: post-stratification.
          4. A group with no personas borrows the rate of its party group in
             that state, then the state's overall rate.
Run     : uv run python scripts/05a-method_a.py [--simulated]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANA, HOUSE_PARTY, SIM, TURNOUT_WEIGHT, load_candidates, load_responses  # noqa: E402

KEYS = ["race", "member", "model_set", "model_slug", "repeat", "run_date"]
CELL = ["party_group", "age_group", "degree"]
STATS = ["n", "tw", "d", "r", "o"]


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return df.groupby(by, observed=True).agg(
        n=("w", "size"), tw=("w", "sum"), d=("wd", "sum"), r=("wr", "sum"), o=("wo", "sum")
    ).reset_index()


def main(simulated: bool):
    src = SIM if simulated else ANA
    resp = load_responses("A", simulated)
    if resp.empty:
        raise SystemExit(
            "No Method A responses found in data/02-analysis_data/responses/.")
    personas = pd.read_parquet(src / "personas.parquet")[["persona_id"] + CELL]
    frame = pd.read_parquet(src / "frame.parquet")[["race"] + CELL + ["share"]]
    cands = load_candidates(simulated)[["race", "candidate", "party_code"]]

    r = resp[resp["parsed"].astype(bool)].copy()
    if "run_date" not in r:
        r["run_date"] = "unknown"
    r = r.merge(personas, on="persona_id", how="left")
    r = r.merge(cands.rename(columns={"candidate": "governor_vote"}), on=[
                "race", "governor_vote"], how="left")
    r["party"] = r["party_code"].fillna(
        r["governor_vote"].map({"Other": "O", "Don't know": "DK"}))
    r["w"] = r["turnout"].map(TURNOUT_WEIGHT).fillna(0)
    r["wd"] = r["w"] * (r["party"] == "D")
    r["wr"] = r["w"] * (r["party"] == "R")
    r["wo"] = r["w"] * (r["party"] == "O")

    # Rates at three levels: group, party group, state.
    cell = summarise(r, KEYS + CELL)
    pg = summarise(r, KEYS + ["party_group"]
                   ).rename(columns={s: f"{s}_pg" for s in STATS})
    st = summarise(r, KEYS).rename(columns={s: f"{s}_st" for s in STATS})

    # Every group in the frame, for every race x model x repeat x day.
    grid = st[KEYS].merge(frame, on="race")
    grid = grid.merge(cell, on=KEYS + CELL, how="left").merge(pg,
                                                              on=KEYS + ["party_group"], how="left")
    grid = grid.merge(st, on=KEYS, how="left")
    for s in STATS:  # fall back: group -> party group -> state
        grid[s] = grid[s].fillna(grid[f"{s}_pg"]).fillna(grid[f"{s}_st"])

    valid = (grid["d"] + grid["r"] + grid["o"]) > 0
    turnout_rate = np.where(grid["n"] > 0, grid["tw"] / grid["n"], 0)
    votes = grid["share"] * turnout_rate
    denom = (grid["d"] + grid["r"] + grid["o"]).where(valid, 1)
    grid["D"] = votes * grid["d"] / denom * valid
    grid["R"] = votes * grid["r"] / denom * valid
    grid["O"] = votes * grid["o"] / denom * valid

    est = grid.groupby(KEYS, as_index=False)[["D", "R", "O"]].sum()
    total = est[["D", "R", "O"]].sum(axis=1).replace(0, np.nan)
    est["dem_share"] = 100 * est["D"] / total
    est["rep_share"] = 100 * est["R"] / total
    est["other_share"] = 100 * est["O"] / total
    est = est.merge(cell.groupby(KEYS)["n"].sum().rename(
        "n_personas").reset_index(), on=KEYS)
    est = est.drop(columns=["D", "R", "O"]).dropna(subset=["dem_share"])
    est.insert(0, "method", "A")

    # Ticket-splitting: Democrat for governor but Republican for the House, or the reverse.
    t = r[r["party"].isin(["D", "R"])].copy()
    t["house_party"] = t["house_vote"].map(HOUSE_PARTY)
    t = t[t["house_party"].isin(["D", "R"])]
    split = (t.assign(split=t["party"] != t["house_party"])
             .groupby(["member", "model_set", "model_slug"], as_index=False)
             .agg(answers=("split", "size"), split_rate=("split", "mean")))

    suffix = "_05a" if simulated else ""
    est.to_parquet(src / f"method_a_estimates{suffix}.parquet", index=False)
    split.to_csv(src / f"ticket_splitting{suffix}.csv", index=False)
    print(f"Method A: {len(est):,} race x model x repeat estimates from {len(r):,} parsed answers; "
          f"median Democratic share {est['dem_share'].median():.1f}%")
    print(split.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated", action="store_true")
    main(parser.parse_args().simulated)
