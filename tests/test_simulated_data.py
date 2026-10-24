"""
Purpose : Test the simulated datasets before any real data exist.
Author  : Shrey Sati (Member 2)
Date    : 2026-10-24
Inputs  : data/00-simulated_data/*.parquet (from scripts/00-simulate_data.py)
Run     : uv run pytest tests/test_simulated_data.py -v
Notes   : Every table has a pandera schema listing every one of its columns.
          The schemas are strict, so a table with a column the schema does not
          mention fails: no column can go untested by accident. One
          parametrized test validates every table against its schema. Rules
          that join two tables are plain pytest functions below the schemas.
"""

from functools import cache
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.pandas import Check, Column, DataFrameSchema

SIM = Path(__file__).resolve().parents[1] / "data" / "00-simulated_data"

# ---------------------------------------------------------------------------
# Allowed values, copied from scripts/00-simulate_data.py
# ---------------------------------------------------------------------------
STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Iowa",
    "Kansas", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Nebraska", "Nevada", "New Hampshire", "New Mexico", "New York", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Vermont", "Wisconsin", "Wyoming",
]  # fmt: skip
TERRITORIES = ["Guam", "Northern Mariana Islands", "U.S. Virgin Islands"]
PARTY7 = [
    "a strong Democrat", "a not very strong Democrat",
    "an independent who leans Democratic", "an independent",
    "an independent who leans Republican", "a not very strong Republican",
    "a strong Republican",
]  # fmt: skip
PARTY7_GROUP = dict(zip(PARTY7, ["D", "D", "D", "I", "R", "R", "R"]))
IDEOLOGY = ["very liberal", "liberal", "moderate", "conservative", "very conservative"]
AGE_GROUPS = ["18-29", "30-44", "45-64", "65+"]
EDUCATION = [
    "a high school diploma", "some college", "a two-year degree",
    "a four-year college degree", "a postgraduate degree",
]  # fmt: skip
INCOME = [
    "under $30,000", "$30,000-$49,999", "$50,000-$79,999",
    "$80,000-$119,999", "$120,000-$199,999", "$200,000 or more",
]  # fmt: skip
TURNOUT = [
    "Definitely will vote",
    "Probably will vote",
    "Unlikely to vote",
    "Will not vote",
]
HOUSE = [
    "The Democratic Party candidate",
    "The Republican Party candidate",
    "Other",
    "Don't know",
]
MEMBERS = ["member1", "member2", "member3"]
MODEL_SETS = ["closed_us", "open_chinese", "open_western"]

# Settings from the project plan (guidebook): change here if the plan changes.
QUOTA = {True: 150, False: 75}  # personas per state: competitive / safe
N_VALIDATION = 10  # fixed validation personas per state
REPEATS = 3  # answers per persona per model
MIN_PARSE_RATE = 0.90  # below this share of usable answers, a model is broken
CELLS_PER_STATE = 3 * 4 * 2  # party group x age group x degree


@cache
def load(name: str) -> pd.DataFrame:
    """Read a simulated table once; fail clearly if the simulation has not been run."""
    path = SIM / f"{name}.parquet"
    if not path.exists():
        pytest.fail(f"{path} missing. Run: uv run python scripts/00-simulate_data.py")
    return pd.read_parquet(path)


def schema(columns, **kwargs):
    """A strict DataFrameSchema: a column in the table but not in the schema fails."""
    return DataFrameSchema(columns, strict=True, **kwargs)


def sums_to(total, *cols):
    """Row-wise check that the given columns add up to `total`."""
    return Check(lambda d: (d[list(cols)].sum(axis=1) - total).abs() < 1e-6,
                 error=f"{' + '.join(cols)} must equal {total}")  # fmt: skip


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
# A percentage, allowing for floating-point rounding (100 - 60 - 40 can be -1e-14).
share = Column(float, Check.in_range(-1e-9, 100 + 1e-9))
state = Column(str, Check.isin(STATES))
race_or_territory = Column(str, Check.isin(STATES + TERRITORIES))
party_group = Column(str, Check.isin(["D", "I", "R"]))
age_group = Column(str, Check.isin(AGE_GROUPS))
# Who ran the model, and which call this is: shared by every model-output table.
model_cols = {
    "member": Column(str, Check.isin(MEMBERS)),
    "model_set": Column(str, Check.isin(MODEL_SETS)),
    "model_slug": Column(str, Check.str_length(min_value=1)),
    "repeat": Column(int, Check.in_range(1, REPEATS)),
}
estimate_cols = {
    "method": Column(str, Check.isin(["A", "B"])),
    "race": state,
    **model_cols,
    "dem_share": share,
    "rep_share": share,
    "other_share": share,
}

# ---------------------------------------------------------------------------
# Schemas: every column of every table, and every rule within one table
# ---------------------------------------------------------------------------
SCHEMAS = {
    "races": schema({
        "race": Column(str, Check.isin(STATES + TERRITORIES), unique=True),
        "type": Column(str, Check.isin(["state", "territory"])),
        "pvi": Column(float, Check.in_range(-50, 50), nullable=True),
        "incumbent_party": Column(str, Check.isin(["D", "R", "I"])),
        "open_seat": Column(bool),
        "competitive": Column(bool),
        "true_margin": Column(float),  # simulation only
    }, checks=Check(lambda d: d["pvi"].notna() | (d["type"] == "territory"),
                    error="every state needs a PVI; only territories may lack one")),
    "candidates": schema({
        "race": race_or_territory,
        "candidate": Column(str, Check.str_length(min_value=1)),
        "party": Column(str, Check.isin(["D", "R", "O"])),
        "incumbent": Column(bool),
    }, unique=["race", "candidate"]),
    "governors_not_up": schema({
        "state": Column(str, Check(lambda s: ~s.isin(STATES)), unique=True),
        "party": Column(str, Check.isin(["D", "R"])),
    }),
    "history": schema({
        "year": Column(int, Check.isin([2014, 2018, 2022])),
        "race": state,
        "pvi": Column(int, Check.in_range(-50, 50)),
        "incumbent_party": Column(str, Check.isin(["D", "R"])),
        "incumbent_ran": Column(bool),
        "dem_pct": share,
        "rep_pct": share,
    }, checks=Check(lambda d: d["dem_pct"] + d["rep_pct"] <= 100,
                    error="dem_pct + rep_pct is more than 100"),
       unique=["year", "race"]),
    "personas": schema({
        "persona_id": Column(str, unique=True),
        "race": state,
        "survey_year": Column(int, Check.isin([2024, 2025])),
        "age": Column(int, Check.in_range(18, 100)),
        "age_group": age_group,
        "gender": Column(str, Check.isin(["man", "woman"])),
        "race_ethnicity": Column(str, Check.isin(["White", "Black", "Hispanic", "Asian", "Other"])),
        "education": Column(str, Check.isin(EDUCATION)),
        "degree": Column(bool),
        "income": Column(str, Check.isin(INCOME)),
        "party_id": Column(str, Check.isin(PARTY7)),
        "party_group": party_group,
        "ideology": Column(str, Check.isin(IDEOLOGY)),
        "weight": Column(float, Check.gt(0)),
        "validation": Column(bool),
        "vote_2024": Column(str, Check.isin(["Harris", "Trump"])),
        "persona_text": Column(str, [
            Check(lambda s: ~s.str.contains("Trump|Harris|2024"),
                  error="past vote leaks into the prompt"),
            Check(lambda s: ~s.str.contains(r"[{}]|\$[A-Za-z_]"),
                  error="unfilled placeholder such as {age} or $state"),
        ]),
    }, checks=[
        Check(lambda d: d["party_id"].map(PARTY7_GROUP) == d["party_group"],
              error="party_group does not match party_id"),
        Check(lambda d: d["degree"] == d["education"].isin(EDUCATION[3:]),
              error="degree does not match education"),
        Check(lambda d: pd.cut(d["age"], [17, 29, 44, 64, 120], labels=AGE_GROUPS)
              .astype(str) == d["age_group"], error="age_group does not match age"),
    ]),
    "cells": schema({
        "race": state,
        "party_group": party_group,
        "age_group": age_group,
        "degree": Column(bool),
        "cell_id": Column(str, unique=True),
        "cell_description": Column(str, Check.str_startswith("registered ")),
    }, unique=["race", "party_group", "age_group", "degree"]),
    "frame": schema({
        "race": state,
        "cell_id": Column(str, unique=True),
        "party_group": party_group,
        "age_group": age_group,
        "degree": Column(bool),
        "share": Column(float, Check.in_range(0, 1)),
    }, checks=Check(lambda d: (d.groupby("race")["share"].sum() - 1).abs().max() < 1e-6,
                    error="shares must sum to 1 within each state"),
       unique=["race", "party_group", "age_group", "degree"]),
    "responses_method_a": schema({
        "run_date": Column(str, Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
        **model_cols,
        "method": Column(str, Check.eq("A")),
        "persona_id": Column(str),
        "race": state,
        "turnout": Column(str, Check.isin(TURNOUT), nullable=True),
        "governor_vote": Column(str, nullable=True),  # checked against candidates below
        "house_vote": Column(str, Check.isin(HOUSE), nullable=True),
        "raw_text": Column(str, Check.str_length(min_value=1)),
        "parsed": Column(bool),
        "prompt_tokens": Column(int, Check.gt(0)),
        "completion_tokens": Column(int, Check.gt(0)),
        "cost_usd": Column(float, Check.ge(0)),
    }, unique=["persona_id", "model_slug", "repeat", "run_date"]),
    "method_a_estimates": schema({
        **estimate_cols,
        "n_personas": Column(float, Check.gt(0)),
    }, checks=sums_to(100, "dem_share", "rep_share", "other_share"),
       unique=["race", "model_slug", "repeat"]),
    "method_b_estimates": schema({
        **estimate_cols,
        "n_cells": Column(int, Check.eq(CELLS_PER_STATE)),
    }, checks=sums_to(100, "dem_share", "rep_share", "other_share"),
       unique=["race", "model_slug", "repeat"]),
    "method_c_responses": schema({
        "race": race_or_territory,
        **model_cols,
        "dem_two_party_share": share,
        "uncertainty_plus_minus": Column(int, Check.gt(0)),
        "probability_dem_wins": Column(float, Check.in_range(0, 1)),
    }, unique=["race", "model_slug", "repeat"]),
    "polls": schema({
        "race": state,
        "pollster": Column(str, Check.str_length(min_value=1)),
        "end_date": Column(str, Check.in_range("2026-08-01", "2026-11-03")),
        "sample_size": Column(int, Check.gt(0)),
        "population": Column(str, Check.isin(["LV", "RV", "A"])),
        "dem_pct": share,
        "rep_pct": share,
    }, checks=Check(lambda d: d["dem_pct"] + d["rep_pct"] <= 100,
                    error="candidate shares add to more than 100")),
    "forecast": schema({
        "race": Column(str, Check.isin(STATES), unique=True),
        "type": Column(str, Check.eq("state")),
        "incumbent_party": Column(str, Check.isin(["D", "R"])),
        **{c: share for c in ["dem_share_lo95", "dem_share_lo80", "dem_share_mean",
                              "dem_share_hi80", "dem_share_hi95"]},
        "p_dem_win": Column(float, Check.in_range(0, 1)),
    }, checks=Check(lambda d: (d["dem_share_lo95"] <= d["dem_share_lo80"])
                    & (d["dem_share_lo80"] <= d["dem_share_mean"])
                    & (d["dem_share_mean"] <= d["dem_share_hi80"])
                    & (d["dem_share_hi80"] <= d["dem_share_hi95"]),
                    error="intervals must be nested around the mean")),
    "simulations": schema({
        "draw": Column(int, unique=True),
        "dem_governors": Column(int, Check.in_range(0, 50)),
        "rep_governors": Column(int, Check.in_range(0, 50)),
        "flips": Column(int, Check.in_range(0, len(STATES))),
    }, checks=sums_to(50, "dem_governors", "rep_governors")),
}  # fmt: skip


@pytest.mark.parametrize("name", SCHEMAS)
def test_table_matches_schema(name):
    SCHEMAS[name].validate(load(name), lazy=True)


def test_schemas_reject_broken_data():
    """A schema that accepts everything would pass silently, so feed it bad rows."""
    bad = load("forecast").copy()
    bad.loc[0, "dem_share_lo80"] = bad.loc[0, "dem_share_mean"] + 1
    with pytest.raises(pa.errors.SchemaErrors):
        SCHEMAS["forecast"].validate(bad, lazy=True)
    extra = load("simulations").assign(unexpected=1)  # a column nobody tests
    with pytest.raises(pa.errors.SchemaErrors):
        SCHEMAS["simulations"].validate(extra, lazy=True)


# ---------------------------------------------------------------------------
# Rules that join tables
# ---------------------------------------------------------------------------
def test_36_races_plus_14_not_up_make_50_governorships():
    races, not_up = load("races"), load("governors_not_up")
    assert sorted(races.loc[races["type"] == "state", "race"]) == sorted(STATES)
    assert len(STATES) + len(not_up) == 50


def test_every_state_race_has_one_democrat_and_one_republican():
    c = load("candidates")
    counts = c[c["race"].isin(STATES)].groupby(["race", "party"]).size().unstack()
    assert (counts["D"] == 1).all() and (counts["R"] == 1).all()


def test_persona_quota_and_validation_slice_per_state():
    races, p = load("races"), load("personas")
    target = races[races["type"] == "state"].set_index("race")["competitive"].map(QUOTA)
    assert p.groupby("race").size().reindex(target.index).equals(target)
    assert (p.groupby("race")["validation"].sum() == N_VALIDATION).all()


def test_simulated_variables_relate_as_intended():
    """The simulation claims ideology and the 2024 vote follow party; check they do."""
    p = load("personas")
    ideo = p["ideology"].map({v: i for i, v in enumerate(IDEOLOGY)})
    mean_ideo = ideo.groupby(p["party_group"]).mean()
    assert mean_ideo["D"] < mean_ideo["I"] < mean_ideo["R"]
    harris = (p["vote_2024"] == "Harris").groupby(p["party_group"]).mean()
    assert harris["D"] > 0.8 and harris["R"] < 0.2


def test_cells_and_frame_describe_the_same_groups():
    """Method B asks about a cell; the frame weights it. Both must mean the same group."""
    cells, frame = load("cells"), load("frame")
    keys = ["race", "cell_id", "party_group", "age_group", "degree"]
    assert (cells.groupby("race").size() == CELLS_PER_STATE).all()
    pd.testing.assert_frame_equal(
        cells[keys].sort_values("cell_id").reset_index(drop=True),
        frame[keys].sort_values("cell_id").reset_index(drop=True),
    )


def test_every_vote_is_a_candidate_in_that_race_or_other_or_dont_know():
    r, c = load("responses_method_a"), load("candidates")
    allowed = c.groupby("race")["candidate"].apply(set).to_dict()
    votes = r.dropna(subset=["governor_vote"])
    ok = [v in allowed[s] | {"Other", "Don't know"}
          for s, v in zip(votes["race"], votes["governor_vote"])]  # fmt: skip
    assert all(ok)


def test_failed_answers_are_kept_flagged_and_carry_no_vote():
    r = load("responses_method_a")
    assert r.loc[~r["parsed"], "governor_vote"].isna().all()
    assert r.loc[r["parsed"], "governor_vote"].notna().all()
    assert 0.01 < (~r["parsed"]).mean() < 0.05  # the ~2.5% planted problems are present
    assert (r.groupby("model_slug")["parsed"].mean() >= MIN_PARSE_RATE).all()


def test_every_persona_gets_every_repeat_from_every_model():
    r, p = load("responses_method_a"), load("personas")
    per = r.groupby(["persona_id", "model_slug"]).size()
    assert (per == REPEATS).all()
    assert len(per) == len(p) * r["model_slug"].nunique()


@pytest.mark.parametrize("name", ["method_a_estimates", "method_b_estimates"])
def test_estimates_cover_every_state_model_and_repeat(name):
    est, n_models = load(name), load("responses_method_a")["model_slug"].nunique()
    assert set(est["race"]) == set(STATES)
    assert len(est) == len(STATES) * n_models * REPEATS


def test_simulations_keep_the_14_fixed_seats():
    sims, not_up = load("simulations"), load("governors_not_up")
    fixed_d = (not_up["party"] == "D").sum()
    assert sims["dem_governors"].between(fixed_d, fixed_d + len(STATES)).all()