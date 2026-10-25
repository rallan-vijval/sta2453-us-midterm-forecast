"""
Purpose : Download each 2026 state governor page on Wikipedia and extract its
          general-election polls into one table.
Author  : Shrey Sati (Member 2)
Date    : 2026-10-25
Inputs  : https://en.wikipedia.org/wiki/2026_<State>_gubernatorial_election (36 pages)
          data/01-raw_data/governor_candidates_2026.csv  candidate names and parties (01c)
Outputs : data/01-raw_data/wikipedia/states_2026/<State>.html  page snapshots
          data/01-raw_data/governor_polls_2026.csv  one row per poll
          data/01-raw_data/download_log.csv  what was downloaded and when (appended)
Notes   : - Pages change, so each page is saved once and re-read from the snapshot on
            later runs. Use --refresh to download fresh copies.
          - A table counts as general-election polling if it has a "Poll source" column
            and columns for both the Democratic and the Republican nominee (matched by
            last name). Primary polls and hypothetical match-ups with other candidates
            therefore drop out.
          - Candidates are matched to columns by full name, else by a unique last name
            (Nevada has two Fords). A poll that gives a share to anyone not on the
            ballot (hypothetical match-ups, primary fields) is skipped, as are tables
            listing ranked-choice rounds (Alaska) and poll aggregators.
          - Every named candidate's share is kept in candidate_shares (JSON). The
            party columns sum them by party, so Alaska's three Republicans add up
            in rep_pct, and independents and minor parties such as Ken Block (RI) go
            in minor_pct. other_pct is the poll's own "Other" column.
          - A poll listed more than once (likely/registered voters, with/without leaners
            or minor candidates) is kept once: likely voters first, then the version
            naming every candidate. complete_ballot is False when a poll left out a
            minor candidate who is on the ballot. Versions without the Democratic or
            Republican nominee are hypothetical and dropped.
          - partisan is "D" or "R" when Wikipedia tags the pollster or sponsor as
            partisan, e.g. "Public Policy Polling (D)"; such polls often lean to their side.
            "D/R" marks a bipartisan pair of firms, usually a media poll; "none" means neither.
          - Every general-election poll on the page is kept, however old. How much
            older polls count is a modelling choice left to the forecast script.
            Rows whose end date cannot be read are dropped.
Run     : uv run python scripts/01b-download_polls.py [--refresh] [--download-only]
"""

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, STATES_2026

URL = "https://en.wikipedia.org/wiki/2026_{state}_gubernatorial_election"
SNAP = RAW / "wikipedia" / "states_2026"
# Wikipedia asks automated clients to identify themselves.
HEADERS = {
    "User-Agent": "STA2453-silicon-forecast/1.0 (University of Toronto course project; "
    "shrey.sati@mail.utoronto.ca)"
}
ELECTION_DAY = pd.Timestamp("2026-11-03")
MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def fetch(state: str, refresh: bool, log: list) -> str:
    """Return the page HTML, from the saved snapshot unless --refresh."""
    path = SNAP / f"{state.replace(' ', '_')}.html"
    url = URL.format(state=state.replace(" ", "_"))
    if path.exists() and not refresh:
        log.append(
            {
                "year": 2026,
                "url": url,
                "file": path.name,
                "source": "snapshot",
                "retrieved_utc": "",
                "bytes": path.stat().st_size,
            }
        )
        return path.read_text(encoding="utf-8")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    log.append(
        {
            "year": 2026,
            "url": url,
            "file": path.name,
            "source": "downloaded",
            "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "bytes": len(resp.content),
        }
    )
    time.sleep(1)
    return resp.text


# ---------------------------------------------------------------------------
# Cleaning single cells
# ---------------------------------------------------------------------------
def clean(x) -> str:
    """Remove footnote markers, odd spaces and party stripes from a cell or header."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    x = re.sub(r"\[[^\]]*\]", "", str(x))  # [a], [12], [citation needed]
    x = x.replace("▌", " ").replace("\xa0", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", x).strip()


def last_name(name: str) -> str:
    tokens = [
        t
        for t in re.sub(r"[.,]", " ", name).split()
        if t.lower() not in {"jr", "sr", "ii", "iii", "iv"}
    ]
    return tokens[-1].lower() if tokens else ""


def pct(x):
    """'39%' -> 39.0, '<1%' -> 0.5, '–' or '' -> NaN."""
    s = clean(x).replace(",", "")
    if s.startswith("<"):
        return 0.5
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def sample(x):
    """'815 (LV)' -> (815, 'LV'); '1,505 (RV)' -> (1505, 'RV'); missing -> (NaN, '')."""
    s = clean(x)
    m = re.search(r"\d[\d,]*", s)
    n = float(m.group(0).replace(",", "")) if m else np.nan
    p = re.search(r"\b(LV|RV|A|V)\b", s)
    return n, (p.group(1) if p else "")


def end_date(x):
    """'October 1–4, 2026' / 'Sep 28 – Oct 2, 2026' / 'October 4, 2026' -> last day as a Timestamp."""
    s = clean(x).lower()
    year = re.search(r"20\d\d", s)
    year = int(year.group(0)) if year else 2026
    s = re.sub(r",?\s*20\d\d", "", s)
    parts = [p.strip() for p in re.split(r"[–—-]", s) if p.strip()]
    if not parts:
        return pd.NaT
    month = None
    for part in reversed(parts):  # the last month named on or before the end
        names = [w for w in re.findall(r"[a-z]+", part) if w.rstrip(".") in MONTHS]
        if names:
            month = MONTHS[names[-1].rstrip(".")]
            break
    days = re.findall(r"\d{1,2}", parts[-1])
    if month is None or not days:
        return pd.NaT
    try:
        return pd.Timestamp(year=year, month=month, day=int(days[-1]))
    except ValueError:
        return pd.NaT


# ---------------------------------------------------------------------------
# Finding and reading poll tables
# ---------------------------------------------------------------------------
def header(col) -> str:
    """Join a (possibly multi-level) column header into one clean string."""
    parts = col if isinstance(col, tuple) else (col,)
    out = []
    for p in parts:
        p = clean(p)
        if p and not p.lower().startswith("unnamed") and p not in out:
            out.append(p)
    return " ".join(out)


def find_column(cols: list[str], *words) -> str | None:
    return next((c for c in cols if any(w in c.lower() for w in words)), None)


def page_tables(html: str) -> tuple[list[pd.DataFrame], int]:
    """Read every table on a page separately, after repairing broken span attributes.

    Some pages carry malformed markup such as rowspan='2"' that stops pandas from
    reading any table on the page, so each cell's rowspan/colspan is reduced to its
    digits, and each table is read on its own so one bad table cannot hide the rest.
    Returns the tables and the number that still could not be read.
    """
    soup = BeautifulSoup(html, "lxml")
    tables, failed = [], 0
    for tag in soup.find_all("table"):
        for cell in tag.find_all(["td", "th"]):
            for attr in ("rowspan", "colspan"):
                if cell.has_attr(attr):
                    digits = re.search(r"\d+", str(cell[attr]))
                    if digits:
                        cell[attr] = digits.group(0)
                    else:
                        del cell[attr]
        try:
            tables += pd.read_html(StringIO(str(tag)))
        except ValueError:
            failed += 1
    return tables, failed


# Columns that describe the poll rather than a candidate.
META_WORDS = (
    "poll source",
    "date",
    "sample",
    "margin",
    "other",
    "undecided",
    "unsure",
    "don't know",
    "lead",
    "round",
    "refused",
    "none",
    "would not vote",
)


def match_candidate(col: str, cands: pd.DataFrame) -> str | None:
    """The candidate a column header names: full name first, else a unique last name.

    Full names come first because two candidates can share a last name (Nevada has
    Aaron Ford and Danielle Ford).
    """
    low = col.lower()
    full = [r.candidate for r in cands.itertuples() if r.candidate.lower() in low]
    if len(full) == 1:
        return full[0]
    last = [r.candidate for r in cands.itertuples() if r.last and r.last in low.split()]
    return last[0] if len(last) == 1 else None


def poll_tables(
    tables: list[pd.DataFrame], cands: pd.DataFrame
) -> list[tuple[int, pd.DataFrame, dict, list]]:
    """Return (table number, table, {column: candidate}, [other-person columns]) for each
    general-election poll table: one with a "Poll source" column and columns for both the
    Democratic and the Republican nominee. Aggregator tables and tables that list each poll
    once per ranked-choice round (Alaska) are skipped.
    """
    out = []
    for i, t in enumerate(tables):
        t = t.copy()
        t.columns = [header(c) for c in t.columns]
        cols = list(t.columns)
        if not find_column(cols, "poll source") or find_column(
            cols, "aggregat", "rcv round", "round"
        ):
            continue
        matched, extra = {}, []
        for c in cols:
            if not c or any(w in c.lower() for w in META_WORDS):
                continue
            who = match_candidate(c, cands)
            if who:
                matched[c] = who
            else:
                extra.append(
                    c
                )  # a person not on the ballot (hypothetical or primary poll)
        parties = (
            set(cands.set_index("candidate").loc[list(matched.values()), "party_code"])
            if matched
            else set()
        )
        if {"D", "R"} <= parties:
            out.append((i, t, matched, extra))
    return out


def partisan(pollster: str) -> str:
    """'Public Policy Polling (D)' -> 'D'; sponsored by both sides -> 'D/R'; neither -> 'none'."""
    tags = sorted(set(re.findall(r"\((D|R)\)", pollster)))
    return "/".join(tags) or "none"


def read_polls(
    state: str, tables: list[pd.DataFrame], cands: pd.DataFrame
) -> list[dict]:
    rows = []
    party = cands.set_index("candidate")["party_code"].to_dict()
    for i, t, matched, extra in poll_tables(tables, cands):
        cols = list(t.columns)
        c_src = find_column(cols, "poll source")
        c_date = find_column(cols, "date")
        c_n = find_column(cols, "sample")
        c_moe = find_column(cols, "margin")
        c_other = find_column(cols, "other")
        c_und = find_column(cols, "undecided", "unsure", "don't know")
        for _, r in t.iterrows():
            pollster = clean(r[c_src])
            if not pollster or pollster.lower().startswith("poll source"):
                continue  # blank or repeated header row
            if any(not np.isnan(pct(r[c])) for c in extra):
                continue  # gives a share to someone not on the ballot
            shares = {cand: pct(r[c]) for c, cand in matched.items()}
            if any(np.isnan(v) for k, v in shares.items() if party[k] in ("D", "R")):
                continue  # a version without one of the major-party nominees is hypothetical
            n, population = sample(r[c_n]) if c_n else (np.nan, "")

            def by_party(code, shares=shares):
                vals = [
                    v for k, v in shares.items() if party[k] == code and not np.isnan(v)
                ]
                return float(sum(vals)) if vals else np.nan

            rows.append(
                {
                    "race": state,
                    "pollster": pollster,
                    "partisan": partisan(pollster),
                    "end_date": end_date(r[c_date]) if c_date else pd.NaT,
                    "sample_size": n,
                    "population": population,
                    "margin_of_error": pct(r[c_moe]) if c_moe else np.nan,
                    "dem_pct": by_party("D"),
                    "rep_pct": by_party("R"),
                    "minor_pct": by_party("O"),
                    "other_pct": pct(r[c_other]) if c_other else np.nan,
                    "undecided_pct": pct(r[c_und]) if c_und else np.nan,
                    "complete_ballot": not any(np.isnan(v) for v in shares.values()),
                    "candidate_shares": json.dumps(
                        {k: v for k, v in shares.items() if not np.isnan(v)}
                    ),
                    "table_number": i,
                    "date_text": clean(r[c_date]) if c_date else "",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(refresh: bool, download_only: bool):
    SNAP.mkdir(parents=True, exist_ok=True)
    cands = pd.read_csv(RAW / "governor_candidates_2026.csv")
    cands["last"] = cands["candidate"].map(last_name)
    log, rows = [], []
    for state in STATES_2026:
        try:
            html = fetch(state, refresh, log)
        except requests.RequestException as e:
            print(f"  {state}: download failed ({e})")
            continue
        if download_only:
            continue
        tables, failed = page_tables(html)
        found = read_polls(state, tables, cands[cands["race"] == state])
        rows += found
        note = f" ({failed} tables could not be read)" if failed else ""
        print(f"  {state}: {len(found)} poll rows{note}")

    log_path = RAW / "download_log.csv"
    new_log = pd.DataFrame(log).assign(script="01b-download_polls.py")
    if log_path.exists():
        new_log = pd.concat([pd.read_csv(log_path), new_log], ignore_index=True)
    new_log.to_csv(log_path, index=False)
    if download_only:
        print(f"Saved {len(log)} pages to {SNAP}")
        return

    polls = pd.DataFrame(rows)
    before = len(polls)
    # Keep every poll up to election day. This also drops rows with an unreadable date.
    polls = polls[polls["end_date"] <= ELECTION_DAY]
    # A poll is often listed more than once (likely and registered voters, with and
    # without leaners, with and without minor candidates). Keep one row per poll:
    # likely voters first, then the version naming every candidate, then the first listed.
    polls = polls.assign(
        _pop=polls["population"].map({"LV": 0, "RV": 1, "V": 2, "A": 3}).fillna(4),
        _incomplete=~polls["complete_ballot"],
    )
    polls = polls.sort_values(
        ["race", "pollster", "end_date", "_pop", "_incomplete", "table_number"],
        kind="stable",
    )
    polls = polls.drop_duplicates(subset=["race", "pollster", "end_date"])
    polls = polls.drop(columns=["_pop", "_incomplete"])
    polls = polls.sort_values(
        ["race", "end_date"], ascending=[True, False]
    ).reset_index(drop=True)
    polls["end_date"] = polls["end_date"].dt.date
    polls.to_csv(RAW / "governor_polls_2026.csv", index=False)
    print(
        f"Kept {len(polls)} of {before} poll rows (fieldwork ending "
        f"{polls['end_date'].min()} to {polls['end_date'].max()}), "
        f"in {polls['race'].nunique()} states. "
        f"No polls found for: {sorted(set(STATES_2026) - set(polls['race']))}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Download and parse 2026 governor polls from Wikipedia."
    )
    ap.add_argument(
        "--refresh", action="store_true", help="download fresh copies of the pages"
    )
    ap.add_argument(
        "--download-only",
        action="store_true",
        help="save the pages without parsing them",
    )
    args = ap.parse_args()
    main(args.refresh, args.download_only)
