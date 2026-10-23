"""
Purpose : Download and parse Wikipedia's gubernatorial election pages:
            - 2026: every race, its incumbent, Cook PVI, expert ratings and candidates;
            - 2014, 2018, 2022: past results in the same 36 states (trains Method C);
            - 2023, 2024, 2025: winners in the 14 states not voting in 2026.
Author  : Allwin Steve Paul Justin Anantharaj (Member 3)
Date    : 2026-10-22
Inputs  : https://en.wikipedia.org/wiki/<year>_United_States_gubernatorial_elections
Outputs : data/01-raw_data/wikipedia/gubernatorial_<year>.html   page snapshots
          data/01-raw_data/governor_races_2026.csv     one row per 2026 race
          data/01-raw_data/governor_candidates_2026.csv one row per 2026 candidate
          data/01-raw_data/governor_results_history.csv one row per state x past year
          data/01-raw_data/governors_not_up_2026.csv    the 14 seats not up in 2026
          data/01-raw_data/territory_results.csv        past territory results
          data/01-raw_data/download_log.csv             what was downloaded and when
Notes   : Pages change, so each page is saved once as HTML and re-parsed from the
          snapshot on later runs. Use --refresh to download fresh copies.
          PVI is stored Democratic-positive: "D+3" -> 3, "R+15" -> -15, "EVEN" -> 0.
Run     : uv run python scripts/01c-download_governor_data.py [--refresh]
"""

import argparse
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "01-raw_data"
SNAP = RAW / "wikipedia"
SNAP.mkdir(parents=True, exist_ok=True)

URL = "https://en.wikipedia.org/wiki/{year}_United_States_gubernatorial_elections"
# Wikipedia asks automated clients to identify themselves.
HEADERS = {
    "User-Agent": "STA2453-silicon-forecast/1.0 (University of Toronto course project; "
    "replace-with-your-email@mail.utoronto.ca)"
}

YEAR_NOW = 2026
YEARS_HISTORY = [2014, 2018, 2022]
YEARS_NOT_UP = [2023, 2024, 2025]

STATES_2026 = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Iowa",
    "Kansas", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Nebraska", "Nevada", "New Hampshire", "New Mexico", "New York", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Vermont", "Wisconsin", "Wyoming",
]
TERRITORY_NAMES = {
    "guam": "Guam",
    "northern mariana islands": "Northern Mariana Islands",
    "u.s. virgin islands": "U.S. Virgin Islands",
    "united states virgin islands": "U.S. Virgin Islands",
}

# Candidate text looks like "Kay Ivey (Republican) 66.9%" (past years) or
# "Doug Jones (Democratic)" (2026). Entries may be separated by ";", line
# breaks, or nothing at all, so the patterns read them one after another.
CAND_WITH_PCT = re.compile(r"([^()%;\n▌]+?)\s*\(([^()]+)\)\s*([\d.]+)\s*%")
CAND_NO_PCT = re.compile(r"([^()%;\n▌]+?)\s*\(([^()]+)\)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean(x) -> str:
    """Remove footnote markers, party stripes and odd spaces from a cell."""
    if pd.isna(x):
        return ""
    x = str(x)
    x = re.sub(r"\[[^\]]*\]", "", x)  # [a], [12], [citation needed]
    x = x.replace("▌", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", x).strip()


def party_code(party: str) -> str:
    """Democratic (incl. Minnesota's DFL) -> D, Republican -> R, anything else -> O."""
    p = clean(party).lower()
    if p.startswith("democratic") or p == "dfl":
        return "D"
    if p.startswith("republican"):
        return "R"
    return "O"


def parse_pvi(x) -> float:
    s = clean(x).upper().replace(" ", "")
    if s == "EVEN":
        return 0.0
    m = re.fullmatch(r"([DR])\+(\d+(?:\.\d+)?)", s)
    if not m:
        return float("nan")
    return float(m.group(2)) * (1 if m.group(1) == "D" else -1)


def parse_last_race(x):
    """'66.91% R' -> (66.91, 'R'); 'DFL' counts as D."""
    m = re.search(r"([\d.]+)\s*%\s*([A-Za-z]+)", clean(x))
    if not m:
        return float("nan"), ""
    return float(m.group(1)), party_code({"D": "Democratic", "R": "Republican"}.get(m.group(2), m.group(2)))


def parse_candidates(cell, with_pct: bool) -> list[dict]:
    text = clean(cell)
    pattern = CAND_WITH_PCT if with_pct else CAND_NO_PCT
    out = []
    for m in pattern.finditer(text):
        name = re.sub(r"^.*:\s*", "", m.group(1))  # drops "Runoff:", "First round:"
        name = re.sub(r"^(Others|Other)\b", "", name).strip(" ;,:")
        if not name:
            continue
        row = {"candidate": name, "party": m.group(2).strip(), "party_code": party_code(m.group(2))}
        if with_pct:
            row["pct"] = float(m.group(3))
        out.append(row)
    return out


def dedupe(cands: list[dict]) -> list[dict]:
    """A runoff or ranked-choice count lists a candidate twice; keep the larger (final) share."""
    best = {}
    for c in cands:
        if c["candidate"] not in best or c["pct"] > best[c["candidate"]]["pct"]:
            best[c["candidate"]] = c
    return list(best.values())


def last_name(name: str) -> str:
    tokens = re.sub(r"[.,]", " ", name).split()
    tokens = [t for t in tokens if t.lower() not in {"jr", "sr", "ii", "iii", "iv"}]
    return tokens[-1].lower() if tokens else ""


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the lowest header level and clean names, e.g. ('Incumbent','Governor') -> 'governor'."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    #df.columns = [clean(c).lower() for c in df.columns]
    df.columns = [clean(c).lower() for c in df.columns]
    if "candidates" not in df.columns:                                                                   # new
        # 2023 and 2025 call the candidate column "General election"                                      # new
        df.columns = ["candidates" if c in {"general election", "candidate"} else c for c in df.columns]  # new
    # Make duplicate names unique ("party", "party_1", ...).
    # Make duplicate names unique ("party", "party_1", ...).
    seen, cols = {}, []
    for c in df.columns:
        seen[c] = seen.get(c, -1) + 1
        cols.append(c if seen[c] == 0 else f"{c}_{seen[c]}")
    df.columns = cols
    return df


def find_table(tables, first_col: set, must_have: set, min_rows: int):
    for t in tables:
        t = flatten(t)
        cols = set(t.columns)
        if t.columns[0] in first_col and must_have <= cols and len(t) >= min_rows:
            return t
    raise ValueError(f"No table with first column in {first_col} and columns {must_have}")


def fetch(year: int, refresh: bool, log: list) -> str:
    path = SNAP / f"gubernatorial_{year}.html"
    if path.exists() and not refresh:
        log.append({"year": year, "url": URL.format(year=year), "file": path.name,
                    "source": "snapshot", "retrieved_utc": "", "bytes": path.stat().st_size})
        return path.read_text(encoding="utf-8")
    resp = requests.get(URL.format(year=year), headers=HEADERS, timeout=60)
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    log.append({"year": year, "url": URL.format(year=year), "file": path.name,
                "source": "downloaded",
                "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "bytes": len(resp.content)})
    return resp.text


def state_tables(html: str):
    """Return (summary table of states, ratings table or None, territory table or None)."""
    tables = pd.read_html(StringIO(html))
    summary = find_table(tables, {"state", "states"}, {"party", "candidates"}, min_rows=2)
    summary = summary.rename(columns={summary.columns[0]: "state"})
    try:
        ratings = find_table(tables, {"state"}, {"pvi"}, min_rows=3)
    except ValueError:
        ratings = None
    try:
        territories = find_table(tables, {"territory", "territories"}, {"candidates"}, min_rows=1)
        territories = territories.rename(columns={territories.columns[0]: "territory"})
    except ValueError:
        territories = None
    return summary, ratings, territories


def incumbent_col(df: pd.DataFrame) -> str:
    return "governor" if "governor" in df.columns else "incumbent"


# ---------------------------------------------------------------------------
# 2026 races and candidates
# ---------------------------------------------------------------------------
def build_2026(html: str):
    summary, ratings, terr = state_tables(html)
    inc = incumbent_col(summary)

    race_rows, cand_rows = [], []
    for kind, table, name_col in [("state", summary, "state"), ("territory", terr, "territory")]:
        if table is None:
            continue
        for _, r in table.iterrows():
            name = clean(r[name_col])
            if kind == "territory":
                name = TERRITORY_NAMES.get(name.lower())
                if name is None:  # skips the District of Columbia (not a governor race)
                    continue
            incumbent = clean(r[inc])
            inc_party = party_code(r["party"])
            cands = parse_candidates(r["candidates"], with_pct=False)
            inc_running = any(
                last_name(c["candidate"]) == last_name(incumbent) and c["party_code"] == inc_party
                for c in cands
            )
            status = clean(r.get("status", ""))
            leaving = re.search(r"term-limited|retir|lost|resign|died|not running|withdr", status.lower())
            last_pct, last_party = parse_last_race(r.get("last race", ""))
            race_rows.append({
                "race": name, "type": kind, "incumbent": incumbent, "incumbent_party": inc_party,
                "first_elected": clean(r.get("first elected", "")), "status": status,
                "open_seat": (not inc_running) or bool(leaving), "last_race_pct": last_pct, "last_race_party": last_party,
                "n_candidates": len(cands),
            })
            for c in cands:
                cand_rows.append({
                    "race": name, "type": kind, **c,
                    "incumbent": inc_running and not leaving and last_name(c["candidate"]) == last_name(incumbent),
                })

    races = pd.DataFrame(race_rows)

    # Add Cook PVI and expert ratings (states only).
    if ratings is not None:
        rt = ratings.copy()
        rt["race"] = rt["state"].map(clean)
        rt["pvi"] = rt["pvi"].map(parse_pvi)
        rating_cols = [c for c in rt.columns
                       if c.split(" ")[0] in {"cook", "ie", "sabato", "wh", "rcp", "fox", "silver", "ddhq", "538", "politico"}]
        rt = rt.rename(columns={c: "rating_" + c.split(" ")[0] for c in rating_cols})
        keep = ["race", "pvi"] + ["rating_" + c.split(" ")[0] for c in rating_cols]
        rt = rt[keep].drop_duplicates("race")
        for c in keep[2:]:
            rt[c] = rt[c].map(clean)
        races = races.merge(rt, on="race", how="left")
    else:
        races["pvi"] = float("nan")

    # Competitive rule: lean within 5 PVI points (about 10 points of margin),
    # or the governor's party differs from the state's lean.
    lean = races["pvi"].apply(lambda p: "D" if p > 0 else ("R" if p < 0 else "EVEN"))
    races["competitive"] = (races["type"] == "state") & (
        (races["pvi"].abs() <= 5) | ((lean != "EVEN") & (lean != races["incumbent_party"]))
    )
    return races, pd.DataFrame(cand_rows)


# ---------------------------------------------------------------------------
# Past results (2014, 2018, 2022) for the same 36 states
# ---------------------------------------------------------------------------
def build_history(year: int, html: str):
    summary, ratings, terr = state_tables(html)
    inc = incumbent_col(summary)
    rows, terr_rows = [], []
    for _, r in summary.iterrows():
        state = clean(r["state"])
        cands = dedupe(parse_candidates(r["candidates"], with_pct=True))
        if not cands:
            continue
        incumbent = clean(r[inc])
        inc_party = party_code(r["party"])
        dem = sum(c["pct"] for c in cands if c["party_code"] == "D")
        rep = sum(c["pct"] for c in cands if c["party_code"] == "R")
        winner = max(cands, key=lambda c: c["pct"])
        rows.append({
            "year": year, "race": state, "incumbent": incumbent, "incumbent_party": inc_party,
            "incumbent_ran": any(last_name(c["candidate"]) == last_name(incumbent) for c in cands),
            "dem_pct": dem, "rep_pct": rep,
            "winner": winner["candidate"], "winner_party": winner["party_code"],
        })
    hist = pd.DataFrame(rows)
    if ratings is not None:
        rt = ratings.assign(race=ratings["state"].map(clean), pvi=ratings["pvi"].map(parse_pvi))
        hist = hist.merge(rt[["race", "pvi"]].drop_duplicates("race"), on="race", how="left")
    else:
        hist["pvi"] = float("nan")
    if terr is not None:
        for _, r in terr.iterrows():
            name = TERRITORY_NAMES.get(clean(r["territory"]).lower())
            if name is None:
                continue
            for c in dedupe(parse_candidates(r["candidates"], with_pct=True)):
                terr_rows.append({"year": year, "race": name, **c})
    return hist, pd.DataFrame(terr_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(refresh: bool):
    log = []

    races, cands = build_2026(fetch(YEAR_NOW, refresh, log))
    n_states = (races["type"] == "state").sum()
    assert n_states == 36, f"Expected 36 state races in 2026, found {n_states}"
    missing = set(STATES_2026) - set(races["race"])
    assert not missing, f"States missing from the 2026 table: {missing}"
    assert races.loc[races["type"] == "state", "pvi"].notna().all(), "A 2026 state has no PVI"
    print(f"2026: {n_states} states, {(races['type'] == 'territory').sum()} territories, "
          f"{len(cands)} candidates, {races['competitive'].sum()} competitive")

    hist_parts, terr_parts = [], []
    for y in YEARS_HISTORY:
        h, t = build_history(y, fetch(y, refresh, log))
        assert len(h) == 36, f"Expected 36 races in {y}, found {len(h)}"
        hist_parts.append(h)
        terr_parts.append(t)
        print(f"{y}: {len(h)} races, PVI missing for {h['pvi'].isna().sum()}")
    history = pd.concat(hist_parts, ignore_index=True)
    territory_results = pd.concat(terr_parts, ignore_index=True)

    not_up = []
    for y in YEARS_NOT_UP:
        h, _ = build_history(y, fetch(y, refresh, log))
        h = h[~h["race"].isin(STATES_2026)]
        not_up.append(h.assign(election_year=y)[["race", "election_year", "winner", "winner_party"]])
    not_up = (
        pd.concat(not_up)
        .sort_values("election_year")
        .drop_duplicates("race", keep="last")
        .rename(columns={"race": "state", "winner": "governor", "winner_party": "party"})
    )
    assert len(not_up) == 14, f"Expected 14 seats not up in 2026, found {len(not_up)}: {sorted(not_up['state'])}"
    print(f"Not up in 2026: {len(not_up)} seats ({(not_up['party'] == 'D').sum()} D, "
          f"{(not_up['party'] == 'R').sum()} R)")

    races.to_csv(RAW / "governor_races_2026.csv", index=False)
    cands.to_csv(RAW / "governor_candidates_2026.csv", index=False)
    history.to_csv(RAW / "governor_results_history.csv", index=False)
    not_up.to_csv(RAW / "governors_not_up_2026.csv", index=False)
    territory_results.to_csv(RAW / "territory_results.csv", index=False)

    log_path = RAW / "download_log.csv"
    new_log = pd.DataFrame(log).assign(script="01c-download_governor_data.py")
    if log_path.exists():
        new_log = pd.concat([pd.read_csv(log_path), new_log], ignore_index=True)
    new_log.to_csv(log_path, index=False)
    print("Saved to", RAW)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--refresh", action="store_true", help="download fresh copies of the pages")
    main(parser.parse_args().refresh)
