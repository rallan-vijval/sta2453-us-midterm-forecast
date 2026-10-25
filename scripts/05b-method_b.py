"""
Purpose : Method B. Turn each demographic group's percentage split into a vote
          share per race, separately for every model, repeat and run date.
Author  : Shrey (Member 2)
Date    : 2026-10-22
Inputs  : data/02-analysis_data/responses/*_B_*.parquet   (every member, every day)
          data/02-analysis_data/cells.parquet, frame.parquet, candidates.parquet
Outputs : data/02-analysis_data/method_b_estimates.parquet
Notes   : Asking for a group's distribution, rather than one persona's single
          answer, counters mode collapse: a model asked to be one person
          tends to give the group's most typical answer every time.
          1. Each accepted distribution (sum 90 to 110) is rescaled to 100,
             and "Don't know" is set aside so shares are of decided voters.
          2. Each group counts in proportion to its frame share times its
             reported turnout percentage.
          3. Shares are summed by party: the Democratic nominee, the
             Republican nominee, and everyone else.
Run     : uv run python scripts/05b-method_b.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANA, load_candidates, load_responses  # noqa: E402

KEYS = ["race", "member", "model_set", "model_slug", "repeat", "run_date"]


def main():
    resp = load_responses("B")
    if resp.empty:
        raise SystemExit("No Method B responses found in data/02-analysis_data/responses/.")
    frame = pd.read_parquet(ANA / "frame.parquet")[["race", "cell_id", "share"]]
    party = load_candidates().set_index(["race", "candidate"])["party_code"].to_dict()

    r = resp[resp["parsed"].astype(bool)].copy()
    rows = []
    for row in r.itertuples():
        dist = json.loads(row.distribution_json)
        decided = {k: v for k, v in dist.items() if k != "Don't know"}
        total = sum(decided.values())
        if total <= 0:
            continue
        d = sum(v for k, v in decided.items() if party.get((row.race, k)) == "D") / total
        rp = sum(v for k, v in decided.items() if party.get((row.race, k)) == "R") / total
        rows.append({"race": row.race, "member": row.member, "model_set": row.model_set,
                     "model_slug": row.model_slug, "repeat": row.repeat, "run_date": row.run_date,
                     "cell_id": row.cell_id, "d": d, "r": rp, "turnout": row.turnout_percent / 100})
    g = pd.DataFrame(rows).merge(frame, on=["race", "cell_id"], how="inner")
    g["w"] = g["share"] * g["turnout"]
    g["wd"], g["wr"] = g["w"] * g["d"], g["w"] * g["r"]
    est = g.groupby(KEYS, as_index=False).agg(w=("w", "sum"), wd=("wd", "sum"), wr=("wr", "sum"),
                                              n_cells=("cell_id", "nunique"))
    est = est[est["w"] > 0]
    est["dem_share"] = 100 * est["wd"] / est["w"]
    est["rep_share"] = 100 * est["wr"] / est["w"]
    est["other_share"] = 100 - est["dem_share"] - est["rep_share"]
    est = est.drop(columns=["w", "wd", "wr"])
    est.insert(0, "method", "B")
    est.to_parquet(ANA / "method_b_estimates.parquet", index=False)
    print(f"Method B: {len(est):,} race x model x repeat estimates from {len(r):,} parsed answers; "
          f"median groups per estimate {est['n_cells'].median():.0f} of 24")


if __name__ == "__main__":
    main()
