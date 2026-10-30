"""
Purpose : Paths, constants and small helpers shared by scripts 02 to 06, so
          every script reads files from the same places and labels parties
          and answers the same way.
Author  : Vijval Rallan (Member 1), Shrey Sati (Member 2), Allwin Steve Paul Justin Anantharaj (Member 3)
Date    : 2026-10-22
Notes   : Not run on its own; other scripts do `from common import ...`.
"""

import re
import zlib
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (built from the repo root, never hard-coded)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "data" / "00-simulated_data"
RAW = ROOT / "data" / "01-raw_data"
CES_DIR = RAW / "ces"
ANA = ROOT / "data" / "02-analysis_data"
RESP = ANA / "responses"
FC = ROOT / "data" / "03-forecast"
PROMPTS = ROOT / "prompts"
CONFIG = ROOT / "config"

SEED = 2453

# ---------------------------------------------------------------------------
# Races
# ---------------------------------------------------------------------------
STATES_2026 = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Iowa",
    "Kansas", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Nebraska", "Nevada", "New Hampshire", "New Mexico", "New York", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Vermont", "Wisconsin", "Wyoming",
]
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Iowa": "IA", "Kansas": "KS", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Mexico": "NM",
    "New York": "NY", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Vermont": "VT", "Wisconsin": "WI", "Wyoming": "WY",
    "Guam": "GU", "Northern Mariana Islands": "MP", "U.S. Virgin Islands": "VI",
}
ABBR_STATE = {v: k for k, v in STATE_ABBR.items()}

# ---------------------------------------------------------------------------
# Answer options (must match the prompt files exactly)
# ---------------------------------------------------------------------------
TURNOUT_OPTIONS = ["Definitely will vote",
                   "Probably will vote", "Unlikely to vote", "Will not vote"]
TURNOUT_WEIGHT = dict(zip(TURNOUT_OPTIONS, [1.0, 0.7, 0.2, 0.0]))
HOUSE_OPTIONS = ["The Democratic Party candidate",
                 "The Republican Party candidate", "Other", "Don't know"]
HOUSE_PARTY = dict(zip(HOUSE_OPTIONS, ["D", "R", "O", "DK"]))
EXTRA_OPTIONS = ["Other", "Don't know"]

PARTY_LABEL = {"D": "Democratic", "R": "Republican", "O": "Independent"}
AGE_GROUPS = ["18-29", "30-44", "45-64", "65+"]
PARTY_GROUPS = ["D", "I", "R"]
PARTY_GROUP_WORDS = {"D": "Democrats", "I": "Independents", "R": "Republicans"}


def stable_seed(*parts) -> int:
    """A seed that is the same on every computer (Python's hash() is not)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2**32)


def age_group(age):
    return pd.cut(age, [17, 29, 44, 64, 200], labels=AGE_GROUPS).astype(str)


# ---------------------------------------------------------------------------
# CES files and columns
# ---------------------------------------------------------------------------
# Files to look for, newest naming first.
CES_FILE_PATTERNS = {
    2025: ["CES25*.csv", "CCES25*.csv", "*2025*Common*.csv", "CES25*.dta", "*2025*.dta"],
    2024: ["CCES24_Common_OUTPUT_vv_topost_final.csv", "CCES24*.csv", "CES24*.csv", "CCES24*.dta"],
}

# Field -> accepted raw column names (first one present is used). VERIFY in codebooks.
CES_COLUMNS = {
    "state": ["inputstate"],
    "birthyr": ["birthyr"],
    "gender": ["gender4", "gender"],
    "race": ["race"],
    "hispanic": ["hispanic"],
    "educ": ["educ"],
    "faminc": ["faminc_new", "faminc"],
    "pid7": ["pid7"],
    "ideo5": ["ideo5"],
    "votereg": ["votereg"],
    "weight": ["commonpostweight", "commonweight", "weight"],
    # 2024 presidential vote: kept hidden, never put in a prompt.
    "vote_2024": ["CC24_410", "presvote24post", "presvote24", "CC25_presvote24"],
}
CES_OPTIONAL = {"hispanic", "vote_2024"}


def find_ces_file(year: int):
    for pattern in CES_FILE_PATTERNS[year]:
        hits = sorted(p for p in CES_DIR.glob(
            pattern) if "sample" not in p.name)
        if hits:
            return hits[0]
    return None


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    return pd.read_csv(path, low_memory=False)


def pick_ces_columns(df: pd.DataFrame, year: int) -> dict:
    used, missing = {}, []
    for field, options in CES_COLUMNS.items():
        hit = next((c for c in options if c in df.columns), None)
        if hit:
            used[field] = hit
        elif field not in CES_OPTIONAL:
            missing.append(f"{field} (tried {options})")
    if missing:
        raise SystemExit(
            f"CES {year}: could not find these fields: {missing}. Look them up in the "
            f"codebook and add the column name to CES_COLUMNS in scripts/common.py."
        )
    return used


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_races(simulated: bool = False) -> pd.DataFrame:
    if simulated:
        return pd.read_parquet(SIM / "races.parquet")
    return pd.read_csv(RAW / "governor_races_2026.csv")


def load_candidates(simulated: bool = False) -> pd.DataFrame:
    """race, candidate, party_label, party_code, incumbent, include_in_prompt."""
    if simulated:
        c = pd.read_parquet(SIM / "candidates.parquet")
        c = c.rename(columns={"party": "party_code"})
        c["party_label"] = c["party_code"].map(PARTY_LABEL)
        c["include_in_prompt"] = True
        return c
    clean = ANA / "candidates.parquet"
    if clean.exists():  # written by 02-clean_data.py (corrections applied)
        return pd.read_parquet(clean)
    c = pd.read_csv(
        RAW / "governor_candidates_2026.csv").rename(columns={"party": "party_label"})
    c["include_in_prompt"] = c["party_code"].isin(["D", "R"])
    return c


def load_responses(method: str, simulated: bool = False) -> pd.DataFrame:
    """All saved responses for one method (A, B, C or V) from every member and day."""
    if simulated and method == "A":
        return pd.read_parquet(SIM / "responses_method_a.parquet")
    files = sorted(RESP.glob(f"*_{method}_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def read_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").strip()


def fill(template: str, **values) -> str:
    """Fill {placeholders} with str.replace: the JSON examples also contain braces."""
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template


# ---------------------------------------------------------------------------
# Candidate lists for prompts
# ---------------------------------------------------------------------------
def party_phrase(label: str) -> str:
    label = str(label).strip()
    if label.lower() in {"independent", "nonpartisan"} or label.lower().endswith("party"):
        return label
    return f"{label} Party"


def candidate_lines(cands: pd.DataFrame, with_party: bool = True, with_incumbent: bool = True) -> str:
    lines = []
    for _, c in cands.iterrows():
        bits = []
        if with_party:
            bits.append(party_phrase(c["party_label"]))
        if with_incumbent and bool(c.get("incumbent", False)):
            bits.append("incumbent governor")
        lines.append(f"- {c['candidate']}" +
                     (f" ({', '.join(bits)})" if bits else ""))
    return "\n".join(lines)


def quoted_names(cands: pd.DataFrame) -> str:
    return ", ".join(f'"{n}"' for n in cands["candidate"])


def strip_party_sentence(persona: str) -> str:
    """Persona text without party identity, for the party-drop check."""
    return re.sub(r"In politics,\s+you\s+consider\s+yourself\s+.*?\s+and\s+your\s+views\s+are",
                  "Your political views are", persona, flags=re.S)


def norm(s) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def match_option(answer, options: list[str]):
    """Map a model's answer to one allowed option, or None. Accepts small
    differences in case and punctuation, or a unique last-name match."""
    if answer is None or (isinstance(answer, float) and pd.isna(answer)):
        return None
    a = norm(answer)
    for o in options:
        if a == norm(o):
            return o
    hits = [o for o in options if norm(
        o).split() and norm(o).split()[-1] in a.split()]
    return hits[0] if len(hits) == 1 else None
