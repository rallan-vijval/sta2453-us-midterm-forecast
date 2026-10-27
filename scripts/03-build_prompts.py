"""
Purpose : Turn Vijwal's cleaned CES sample into the persona prompts that
          04-query_models.py sends, plus the 24 demographic groups per state
          used by Method B and the weighting frame used by Methods A and B.
Author  : Allwin Steve Paul Justin Anantharaj (Member 3)
Date    : 2026-10-24
Inputs  : data/01-raw_data/governor_candidates_2026.csv   (01c)
          data/02-analysis_data/personas.parquet   (02-clean_data.py: one row per sampled
          CES respondent, 100 per state; columns age, gender, state, race_ethnicity,
          education, income, party_alignment, ideology)
          data/02-analysis_data/frame_source.parquet  (02-clean_data.py: every eligible
          CES respondent with their survey weight)
Outputs : data/02-analysis_data/ces_sample.parquet  an unchanged copy of 02's output
          data/02-analysis_data/personas.parquet  the same people, one row per persona, with persona_text
          data/02-analysis_data/cells.parquet     24 groups per state (party x age x degree)
          data/02-analysis_data/frame.parquet     each group's share of the state's registered voters
          data/02-analysis_data/frame_states.csv  per state: respondents, effective sample size,
                                                  party mix, and the 2024 and 2025 Republican shares
          data/02-analysis_data/candidates.parquet  candidate list, with include_in_prompt
Notes   : The persona sentence is built with an f-string. The CES labels are
          turned into plain English first ("Lean Democrat" -> "an independent
          who leans Democratic"), and a label the script does not recognise
          stops the run with a list of what to add, so nothing odd reaches a model.
          The persona sentence keeps the shape "In politics, you consider
          yourself ... and your views are ...", which the party-drop validation
          check relies on.
          10 personas per state are flagged for the validation checks.
          The frame comes from every eligible CES respondent, weighted, not from
          the 100 personas: the personas show how each kind of voter votes, the
          frame says how many of each kind the state has. Built from the personas
          alone, an unlucky sample (Arkansas: 50 Democrats of 100) would be copied
          into the frame instead of corrected. Respondents are grouped with the
          same functions as the personas, so the groups always line up.
Run     : uv run python scripts/03-build_prompts.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AGE_GROUPS, ANA, PARTY_GROUP_WORDS, PARTY_GROUPS, RAW, SEED, STATES_2026, age_group  # noqa: E402

N_VALIDATION = 10
# Candidates named in the prompts: both major-party nominees everywhere (Alaska's
# top-four ballot has three Republicans), plus every other candidate shown in the
# race's Wikipedia infobox, whatever their party.
# Voters for anyone else can answer "Other".
PROMPT_EXTRA_CANDIDATES = {
    "Maine": ["Rick Bennett"],
    "Arkansas": ["Colt Shelby"],
    "Idaho": ["John Stegner"],
    "Nebraska": ["Rick Beard", "Brett Lindstrom"],
    "Rhode Island": ["Ken Block"],
}
# Spell out party abbreviations so the prompt is unambiguous; the raw file keeps
# Wikipedia's label. Minnesota's DFL is the state's Democratic Party.
PARTY_LABEL_FIX = {"DFL": "Democratic-Farmer-Labor"}
MIN_CELL_N = 3
NEEDED = ["age", "gender", "state", "race_ethnicity", "education", "income", "party_alignment", "ideology"]


# ---------------------------------------------------------------------------
# CES labels -> plain English. Each returns None for a label it does not know.
# ---------------------------------------------------------------------------
def missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) or str(x).strip() == ""


def gender_words(x):
    s = str(x).strip().lower()
    if missing(x) or s in {"other", "prefer not to say"}:
        return "person"
    if s in {"woman", "female"}:
        return "woman"
    if s in {"man", "male"}:
        return "man"
    if "non-binary" in s or "nonbinary" in s:
        return "non-binary person"
    return None


def race_words(x):
    s = str(x).strip().lower()
    if missing(x) or s in {"other", "skipped", "not asked"}:
        return ""  # sentence left out
    words = {"white": "White", "black": "Black", "hispanic": "Hispanic", "asian": "Asian",
             "native american": "Native American", "middle eastern": "Middle Eastern",
             "two or more races": "multiracial", "mixed": "multiracial"}
    return words.get(s)


def education_words(x):
    s = str(x).strip().lower()
    if missing(x):
        return ""
    if s in {"no hs", "no high school"} or "did not" in s or "less than high school" in s:
        return "less than a high school diploma"
    if "post" in s or "graduate degree" in s:
        return "a postgraduate degree"
    if "4-year" in s or "4 year" in s or "bachelor" in s:
        return "a four-year college degree"
    if "2-year" in s or "2 year" in s or "associate" in s:
        return "a two-year college degree"
    if "some college" in s:
        return "some college"
    if "high school" in s:
        return "a high school diploma"
    return None


def has_degree(x) -> bool:
    return education_words(x) in {"a four-year college degree", "a postgraduate degree"}


def income_words(x):
    s = str(x).strip()
    if missing(x) or s.lower() in {"prefer not to say", "skipped", "not asked"}:
        return ""
    s = s.replace(" - ", " to ").replace("-", " to ")
    if s.lower().startswith("less than"):
        return "under " + s[len("less than "):]
    return s if "$" in s else None


def party_words(x):
    """(phrase for the persona, party group D/I/R)."""
    s = str(x).strip().lower()
    if missing(x) or s in {"not sure", "don't know", "other", "skipped", "not asked"}:
        return "unsure which party you support", "I"
    side = "Democrat" if "democrat" in s else "Republican" if "republican" in s else None
    if side is None:
        return ("an independent", "I") if "independent" in s else (None, None)
    group = side[0]
    if "lean" in s:
        return f"an independent who leans {'Democratic' if group == 'D' else 'Republican'}", group
    if "not very strong" in s or "weak" in s:
        return f"a not very strong {side}", group
    if "strong" in s:
        return f"a strong {side}", group
    return f"a {side}", group


def ideology_words(x):
    s = str(x).strip().lower()
    if missing(x) or s in {"not sure", "don't know"}:
        return "not settled"
    return s if s in {"very liberal", "liberal", "moderate", "conservative", "very conservative"} else None


# ---------------------------------------------------------------------------
# The persona sentence
# ---------------------------------------------------------------------------
def persona_text(age, gender, state, race, education, income, party, ideology) -> str:
    article = "an" if str(age).startswith("8") or age in (11, 18) else "a"
    text = f"You are {article} {age}-year-old {gender} who lives in {state}."
    if race:
        text += f" You describe your race/ethnicity as {race}."
    if education and income:
        text += f" Your highest level of education is {education}, and your family income is {income}."
    elif education:
        text += f" Your highest level of education is {education}."
    elif income:
        text += f" Your family income is {income}."
    text += f" In politics, you consider yourself {party} and your views are {ideology}."
    text += " You are registered to vote."
    return text


def load_sample() -> pd.DataFrame:
    """Vijwal's sample. 02 saves it as personas.parquet; this script keeps a copy
    as ces_sample.parquet and then writes personas.parquet with the prompt text,
    so both scripts can be rerun in any order."""
    sample, personas = ANA / "ces_sample.parquet", ANA / "personas.parquet"
    if personas.exists():
        df = pd.read_parquet(personas)
        if "persona_text" not in df.columns:  # fresh output of 02-clean_data.py
            df.to_parquet(sample, index=False)
            print(f"Copied 02's output to {sample.name}: {len(df):,} respondents")
            return df
    if sample.exists():
        df = pd.read_parquet(sample)
        print(f"Reading {sample.name}: {len(df):,} respondents")
        return df
    raise SystemExit("No CES sample found: run 02-clean_data.py first.")


FRAME_KEYS = ["race", "party_group", "age_group", "degree"]


def load_frame_source() -> pd.DataFrame:
    """Every eligible CES respondent from 02, sorted into party x age x degree
    groups by the same functions used for the personas."""
    path = ANA / "frame_source.parquet"
    if not path.exists():
        raise SystemExit("No frame_source.parquet: run 02-clean_data.py first.")
    src = pd.read_parquet(path).rename(columns={"state": "race"})
    src = src[src["race"].isin(STATES_2026)].copy()
    src["party_group"] = [party_words(x)[1] for x in src["party_alignment"]]
    bad = {
        "party_alignment": src.loc[src["party_group"].isna(), "party_alignment"],
        "education": src.loc[src["education"].map(education_words).isna(), "education"],
    }
    bad = {k: sorted(v.astype(str).unique()) for k, v in bad.items() if len(v)}
    if bad:
        raise SystemExit("frame_source labels not recognised (add them to the matching *_words function):\n"
                         + "\n".join(f"  {k}: {v}" for k, v in bad.items()))
    src["age_group"] = age_group(pd.to_numeric(src["age"]))
    src["degree"] = src["education"].map(has_degree)
    return src


def state_summary(src: pd.DataFrame) -> pd.DataFrame:
    """Per state: how many respondents the frame rests on, its effective sample size
    after weighting, the weighted party mix, and the Republican share in each survey
    year (if the years disagree a lot, pooling them needs a second look)."""
    w = src["weight"]
    g = src.assign(w2=w**2, **{f"w_{p}": w * (src["party_group"] == p) for p in PARTY_GROUPS})
    s = g.groupby("race").agg(n_respondents=("weight", "size"), w=("weight", "sum"), w2=("w2", "sum"),
                              **{f"w_{p}": (f"w_{p}", "sum") for p in PARTY_GROUPS})
    s["n_effective"] = (s["w"] ** 2 / s["w2"]).round(0)
    for p in PARTY_GROUPS:
        s[f"share_{p}"] = (100 * s[f"w_{p}"] / s["w"]).round(1)
    by_year = (g.groupby(["race", "survey_year"])[["w_R", "weight"]].sum()
               .assign(r=lambda d: 100 * d["w_R"] / d["weight"])["r"].unstack())
    for year in by_year.columns:
        s[f"share_R_{year}"] = by_year[year].round(1)
    s["share_from_2024"] = (100 * g[g["survey_year"] == 2024].groupby("race")["weight"].sum() / s["w"]).round(1)
    return s.drop(columns=["w", "w2"] + [f"w_{p}" for p in PARTY_GROUPS]).reset_index()


def main():
    raw = load_sample()
    lost = [c for c in NEEDED if c not in raw.columns]
    if lost:
        raise SystemExit(f"02's output is missing columns {lost}; expected {NEEDED}.")
    df = raw.rename(columns={"state": "race"}).copy()
    df = df[df["race"].isin(STATES_2026)].reset_index(drop=True)

    df["gender_text"] = df["gender"].map(gender_words)
    df["race_text"] = df["race_ethnicity"].map(race_words)
    df["education_text"] = df["education"].map(education_words)
    df["income_text"] = df["income"].map(income_words)
    party = df["party_alignment"].map(party_words)
    df["party_text"] = [t[0] for t in party]  # (phrase, group) pairs; .str[0] fails on pandas 3
    df["party_group"] = [t[1] for t in party]
    df["ideology_text"] = df["ideology"].map(ideology_words)

    # Stop if any label was not recognised, and say which.
    unknown = {}
    for src, col in [("gender", "gender_text"), ("race_ethnicity", "race_text"), ("education", "education_text"),
                     ("income", "income_text"), ("party_alignment", "party_text"), ("ideology", "ideology_text")]:
        bad = sorted(df.loc[df[col].isna(), src].astype(str).unique())
        if bad:
            unknown[src] = bad
    if unknown:
        raise SystemExit("Labels not recognised (add them to the matching *_words function):\n"
                         + "\n".join(f"  {k}: {v}" for k, v in unknown.items()))

    df["age"] = pd.to_numeric(df["age"]).astype(int)
    df["age_group"] = age_group(df["age"])
    df["degree"] = df["education"].map(has_degree)
    df["persona_text"] = [
        persona_text(r.age, r.gender_text, r.race, r.race_text, r.education_text, r.income_text,
                     r.party_text, r.ideology_text)
        for r in df.itertuples()
    ]
    df.insert(0, "persona_id", [f"P{i:05d}" for i in range(len(df))])

    rng = np.random.default_rng(SEED)
    df["validation"] = False
    for _, idx in df.groupby("race").groups.items():
        pick = rng.choice(np.asarray(idx), size=min(N_VALIDATION, len(idx)), replace=False)
        df.loc[pick, "validation"] = True

    if "weight" not in df.columns:
        df["weight"] = 1.0  # 02 drew personas in proportion to the CES weight

    keep = ["persona_id", "race", "age", "age_group", "gender", "race_ethnicity", "education", "degree", "income",
            "party_alignment", "party_group", "ideology", "weight", "validation", "persona_text"]
    keep += [c for c in ["survey_year", "respondent_id"] if c in df.columns]
    personas = df[keep]
    personas.to_parquet(ANA / "personas.parquet", index=False)

    # Frame: weighted share of each party x age x degree group within each state,
    # from every eligible CES respondent. n_personas counts the AI personas in each
    # group; small_cell flags groups with too few personas to stand on their own.
    src = load_frame_source()
    full = pd.MultiIndex.from_product([STATES_2026, PARTY_GROUPS, AGE_GROUPS, [True, False]],
                                      names=FRAME_KEYS).to_frame(index=False)
    agg = (src.groupby(FRAME_KEYS)
           .agg(weight=("weight", "sum"), n_respondents=("weight", "size")).reset_index())
    n_personas = personas.groupby(FRAME_KEYS).size().rename("n_personas").reset_index()
    frame = full.merge(agg, how="left", on=FRAME_KEYS).merge(n_personas, how="left", on=FRAME_KEYS)
    frame[["weight", "n_respondents", "n_personas"]] = frame[["weight", "n_respondents", "n_personas"]].fillna(0)
    frame[["n_respondents", "n_personas"]] = frame[["n_respondents", "n_personas"]].astype(int)
    frame["share"] = (frame["weight"] / frame.groupby("race")["weight"].transform("sum")).fillna(0)
    frame["small_cell"] = frame["n_personas"] < MIN_CELL_N
    frame["cell_id"] = (frame["race"].str.replace(" ", "") + "-" + frame["party_group"] + "-"
                        + frame["age_group"].str.replace("+", "plus") + "-"
                        + np.where(frame["degree"], "deg", "nodeg"))
    frame = frame.drop(columns="weight")
    frame.to_parquet(ANA / "frame.parquet", index=False)
    states = state_summary(src)
    states.to_csv(ANA / "frame_states.csv", index=False)

    cells = frame[["race", "cell_id", "party_group", "age_group", "degree", "small_cell"]].copy()
    cells["cell_description"] = (
        "registered " + cells["party_group"].map(PARTY_GROUP_WORDS) + " aged " + cells["age_group"]
        + " " + np.where(cells["degree"], "with", "without") + " a four-year college degree"
    )
    cells.to_parquet(ANA / "cells.parquet", index=False)

    # Candidate list for the prompts (read by common.load_candidates).
    cands = pd.read_csv(RAW / "governor_candidates_2026.csv")
    cands = cands[cands["race"].isin(STATES_2026)].rename(columns={"party": "party_label"})
    cands["party_label"] = cands["party_label"].replace(PARTY_LABEL_FIX)
    extra = {(race, name) for race, names in PROMPT_EXTRA_CANDIDATES.items() for name in names}
    named = pd.Series([(r, c) in extra for r, c in zip(cands["race"], cands["candidate"])], index=cands.index)
    cands["include_in_prompt"] = cands["party_code"].isin(["D", "R"]) | named
    missing_extra = extra - set(zip(cands["race"], cands["candidate"]))
    if missing_extra:
        raise SystemExit(f"PROMPT_EXTRA_CANDIDATES names not on the candidate list: {sorted(missing_extra)}")
    short = cands[cands["include_in_prompt"]].groupby("race")["party_code"].agg(set)
    assert all({"D", "R"} <= s for s in short), "every state needs a Democrat and a Republican in the prompt"
    cands[["race", "candidate", "party_label", "party_code", "incumbent", "include_in_prompt"]].to_parquet(
        ANA / "candidates.parquet", index=False)

    print(f"personas: {len(personas):,} in {personas.race.nunique()} states, "
          f"{int(personas.validation.sum())} flagged for validation")
    print(f"party groups: {personas.party_group.value_counts().to_dict()}")
    print(f"groups: {len(cells)}; groups with fewer than {MIN_CELL_N} personas: {int(frame.small_cell.sum())}")
    print(f"frame: {len(src):,} CES respondents; effective sample size per state from "
          f"{states['n_effective'].min():.0f} to {states['n_effective'].max():.0f}")
    gap = (states.get("share_R_2024", np.nan) - states.get("share_R_2025", np.nan)).abs()
    if gap.notna().any():
        print("largest 2024 vs 2025 gaps in the Republican share (points):\n"
              + states.assign(gap=gap.round(1)).nlargest(5, "gap")[["race", "share_R_2024", "share_R_2025", "gap"]]
              .to_string(index=False))
    print("\nExample persona:\n" + personas["persona_text"].iloc[0])


if __name__ == "__main__":
    main()
