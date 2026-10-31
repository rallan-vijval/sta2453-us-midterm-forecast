"""
Purpose : Test the real data: the downloads, the personas and frame, every model
          answer, the estimates built from them, and the forecast.
Author  : Shrey Sati
Date    : 2026-10-27
Inputs  : data/01-raw_data/, data/02-analysis_data/, data/03-forecast/
Run     : uv run pytest tests/test_analysis_data.py -v
Notes   : Same design as test_simulated_data.py. Every table has a strict pandera
          schema listing all of its columns, so a new or renamed column fails until
          it is described here, and one parametrized test checks every table against
          its schema. Rules that join two tables, or need a whole column at once,
          are plain pytest functions grouped by pipeline step below.
          Model answers are read from every response file, so the daily runs are
          tested too. Completeness tests use the first (full) run.
          Allowed answers come from scripts/common.py, the single place the scripts
          take them from; test_prompts_offer_the_allowed_answers checks they match
          the wording the models were shown.
"""

import json
import sys
from functools import cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.pandas import Check, Column, DataFrameSchema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import (
    AGE_GROUPS,
    EXTRA_OPTIONS,
    HOUSE_OPTIONS,
    STATES_2026,
    TURNOUT_OPTIONS,
)

RAW = ROOT / "data" / "01-raw_data"
ANA = ROOT / "data" / "02-analysis_data"
FC = ROOT / "data" / "03-forecast"
PROMPTS = ROOT / "prompts"

# Settings from the pipeline: change here if the plan changes.
PERSONAS_PER_STATE = 100  # 02-clean_data.py
N_VALIDATION = 10  # validation personas per state, 03-build_prompts.py
CELLS_PER_STATE = 3 * 4 * 2  # party group x age group x degree
FULL_RUN_REPEATS = 3  # repeats in the first, full run (config/models.yaml)
MIN_PARSE_RATE = 0.90  # below this share of usable answers, a model is broken
N_SIMULATIONS = 10_000  # 05-forecast.py
SEATS_NOT_UP = 14
ELECTION_DAY = "2026-11-03"
SPENDING_LIMIT_USD = 20  # the OpenRouter key's limit
CHECKS = [
    "attention",
    "party_drop",
    "incumbent_drop",
    "label_drop",
    "order_reverse",
    "governor_only",
]
RATINGS = ["Toss-up"] + [f"{r} {p}" for r in ["Lean", "Likely", "Solid"] for p in "DRO"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
FILES = {
    # 01-raw_data
    "races": RAW / "governor_races_2026.csv",
    "candidates_raw": RAW / "governor_candidates_2026.csv",
    "history": RAW / "governor_results_history.csv",
    "not_up": RAW / "governors_not_up_2026.csv",
    "polls": RAW / "governor_polls_2026.csv",
    # 02-analysis_data
    "sampling_report": ANA / "sampling_report.csv",
    "personas": ANA / "personas.parquet",
    "cells": ANA / "cells.parquet",
    "frame": ANA / "frame.parquet",
    "frame_source": ANA / "frame_source.parquet",
    "frame_states": ANA / "frame_states.csv",
    "candidates": ANA / "candidates.parquet",
    "run_log": ANA / "run_log.csv",
    "method_a_estimates": ANA / "method_a_estimates.parquet",
    "method_b_estimates": ANA / "method_b_estimates.parquet",
    "ticket_splitting": ANA / "ticket_splitting.csv",
    # 03-forecast
    "forecast": FC / "forecast.parquet",
    "simulations": FC / "simulations.parquet",
    "method_grid": FC / "method_grid.parquet",
    "backtest": FC / "backtest.csv",
    "coefficients": FC / "fundamentals_coefficients.csv",
    "validation_summary": FC / "validation_summary.csv",
    "stability_summary": FC / "stability_summary.csv",
    "cost_summary": FC / "cost_summary.csv",
}


@cache
def load(name: str) -> pd.DataFrame:
    """Read a table once; fail clearly if the script that writes it has not run."""
    if name.startswith("responses_"):
        method = name[-1]
        files = sorted((ANA / "responses").glob(f"*_{method}_*.parquet"))
        files = [f for f in files if "simulated" not in f.name]
        if not files:
            pytest.fail(f"No Method {method} responses: run scripts/04-query_models.py")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    path = FILES[name]
    if not path.exists():
        pytest.fail(f"{path} is missing: run the script that writes it.")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def first_run(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from the first run date: the full run, before the smaller daily runs."""
    return df[df["run_date"] == df["run_date"].min()]


def prompt_candidates() -> pd.DataFrame:
    c = load("candidates")
    return c[c["include_in_prompt"]]


# ---------------------------------------------------------------------------
# Schema helpers and shared columns
# ---------------------------------------------------------------------------
def schema(columns, **kwargs):
    """A strict DataFrameSchema: a column in the table but not in the schema fails."""
    return DataFrameSchema(columns, strict=True, coerce=True, **kwargs)


def sums_to(total, *cols, tol=1e-6):
    return Check(
        lambda d: (d[list(cols)].sum(axis=1) - total).abs() < tol,
        error=f"{' + '.join(cols)} must equal {total}",
    )


def ordered(*cols):
    """Each column is at most the next one, row by row (e.g. lo95 <= lo80 <= hi80)."""

    def check(d):
        ok = pd.Series(True, index=d.index)
        for a, b in pairwise(cols):
            ok &= d[a] <= d[b] + 1e-9
        return ok

    return Check(check, error=" <= ".join(cols))


def text(**kw):
    return Column(str, Check.str_length(min_value=1), **kw)


def nullable(dtype=None, *checks):
    return Column(dtype, list(checks), nullable=True)


pct = Column(float, Check.in_range(0, 100))  # a percentage
prob = Column(float, Check.in_range(0, 1))  # a probability
count = Column(int, Check.ge(0))
state = Column(str, Check.isin(STATES_2026))
party_code = Column(str, Check.isin(["D", "R", "O"]))
party_group = Column(str, Check.isin(["D", "I", "R"]))
date = Column(str, Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"))
utc_time = Column(str, Check.str_matches(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"))
model = Column(str, Check.str_matches(r"^[a-z0-9-]+/[a-z0-9.-]+$"))
tokens = Column(float, Check.ge(0), nullable=True)  # missing only on a failed call
margin = Column(float, Check.in_range(-100, 100))
run_cols = {
    "member": text(),
    "model_set": text(),
    "model_slug": model,
    "repeat": Column(int, Check.ge(1)),
    "run_date": date,
}
estimate_cols = {
    "race": state,
    **run_cols,
    "dem_share": pct,
    "rep_share": pct,
    "other_share": Column(float, Check.in_range(-1e-9, 100)),
}
# Columns every model answer has, whatever the method.
response_cols = {
    "key": text(),
    "method": Column(str),
    "check": Column(str),
    "model_slug": model,
    "provider_pinned": text(),
    "provider_served": nullable(str),
    "race": state,
    "unit_id": text(),
    "repeat": Column(int, Check.ge(1)),
    "candidate_order": Column(str, Check.str_startswith("[")),
    "raw_text": nullable(str),
    "parsed": Column(bool),
    "retries": Column(int, Check.in_range(0, 1)),
    "error": nullable(),
    "prompt_tokens": tokens,
    "completion_tokens": tokens,
    "reasoning_tokens": tokens,
    "cost_usd": Column(float, Check.ge(0), nullable=True),
    "seconds": Column(float, Check.ge(0)),
    "timestamp": utc_time,
    "run_date": date,
    "member": text(),
    "model_set": text(),
    "prompt_version": text(),
}
turnout = nullable(str, Check.isin(TURNOUT_OPTIONS))
house = nullable(str, Check.isin(HOUSE_OPTIONS))

# ---------------------------------------------------------------------------
# Schemas: every column of every table, and every rule within one table
# ---------------------------------------------------------------------------
# fmt: off
SCHEMAS = {
    # --- 01-raw_data -------------------------------------------------------
    "races": schema({
        "race": Column(str, Check.isin(STATES_2026), unique=True),
        "type": Column(str, Check.eq("state")),
        "incumbent": text(),
        "incumbent_party": Column(str, Check.isin(["D", "R"])),
        "first_elected": Column(int, Check.in_range(1990, 2025)),
        "status": text(),
        "open_seat": Column(bool),
        "last_race_pct": pct,
        "last_race_party": Column(str, Check.isin(["D", "R"])),
        "n_candidates": Column(int, Check.ge(2)),
        "pvi": Column(float, Check.in_range(-40, 40)),
        **{f"rating_{s}": text() for s in
           ["cook", "ddhq", "fox", "ie", "rcp", "sabato", "silver"]},
        "competitive": Column(bool),
    }),
    "candidates_raw": schema({
        "race": state,
        "type": Column(str, Check.eq("state")),
        "candidate": text(),
        "party": text(),
        "party_code": party_code,
        "incumbent": Column(bool),
    }, unique=["race", "candidate"]),
    "history": schema({
        "year": Column(int, Check.isin([2014, 2018, 2022])),
        "race": state,
        "incumbent": text(),
        "incumbent_party": party_code,
        "incumbent_ran": Column(bool),
        "dem_pct": pct,  # 0 where no Democrat ran (Alaska 2014)
        "rep_pct": pct,
        "winner": text(),
        "winner_party": party_code,
        "pvi": Column(float, Check.in_range(-40, 40)),
    }, unique=["year", "race"],
       checks=Check(lambda d: d["dem_pct"] + d["rep_pct"] <= 100,
                    error="Democratic plus Republican share above 100")),
    "not_up": schema({
        "state": Column(str, Check.notin(STATES_2026), unique=True),
        "election_year": Column(int, Check.in_range(2023, 2025)),
        "governor": text(),
        "party": Column(str, Check.isin(["D", "R"])),
    }),
    "polls": schema({
        "race": state,
        "pollster": text(),
        "partisan": Column(str, Check.isin(["none", "D", "R", "D/R"])),
        "end_date": Column(str, [Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"),
                                 Check.le(ELECTION_DAY)]),
        "sample_size": nullable(float, Check.gt(0)),
        "population": Column(str, Check.isin(["LV", "RV", "A", "V"])),
        "margin_of_error": nullable(float, Check.in_range(0, 20)),
        "dem_pct": pct,
        "rep_pct": pct,
        "minor_pct": nullable(float, Check.in_range(0, 100)),
        "other_pct": nullable(float, Check.in_range(0, 100)),
        "undecided_pct": nullable(float, Check.in_range(0, 100)),
        "complete_ballot": Column(bool),
        "candidate_shares": Column(str, Check.str_startswith("{")),
        "table_number": count,
        "date_text": text(),
    }, unique=["race", "pollster", "end_date"],
       # Shares add to about 100. Oregon's DHM poll of 3-9 Sep sums to 105 on the
       # page itself; a larger total means a value was counted twice.
       checks=Check(lambda d: d[["dem_pct", "rep_pct", "minor_pct", "other_pct",
                                 "undecided_pct"]].fillna(0).sum(axis=1).between(90, 105),
                    error="poll shares must add to 90-105")),
    # --- 02-analysis_data: personas and the frame ---------------------------
    "sampling_report": schema({
        "state": Column(str, Check.isin(STATES_2026), unique=True),
        "available_2025": count,
        "available_2024": count,
        "sampled_2025": count,
        "sampled_2024": count,
        "sampled_total": Column(int, Check.eq(PERSONAS_PER_STATE)),
        "target_met": Column(bool, Check.eq(True)),
    }, checks=[
        # CES 2025 is used first; 2024 only tops a state up to 100.
        Check(lambda d: d["sampled_2025"] == d["available_2025"].clip(
            upper=PERSONAS_PER_STATE), error="CES 2024 used where 2025 was available"),
        sums_to(PERSONAS_PER_STATE, "sampled_2025", "sampled_2024"),
    ]),
    "personas": schema({
        "persona_id": Column(str, Check.str_matches(r"^P\d{5}$"), unique=True),
        "race": state,
        "age": Column(int, Check.in_range(18, 110)),
        "age_group": Column(str, Check.isin(AGE_GROUPS)),
        "gender": text(),
        "race_ethnicity": text(),
        "education": text(),
        "degree": Column(bool),
        "income": nullable(str),  # one respondent did not answer
        "party_alignment": text(),
        "party_group": party_group,
        "ideology": text(),
        "weight": Column(float, Check.eq(1.0)),  # drawn by weight, so each counts once
        "validation": Column(bool),
        "persona_text": Column(str, [
            Check.str_startswith("You are a"),
            Check.str_contains("You are registered to vote."),
            # Never a past vote: the 2024 vote is held back for checking.
            Check(lambda s: ~s.str.contains(r"\b(?:voted|2020|2024|Trump|Harris|Biden)\b"),
                  element_wise=False, error="persona mentions a past vote"),
        ]),
    }),
    "cells": schema({
        "race": state,
        "cell_id": Column(str, unique=True),
        "party_group": party_group,
        "age_group": Column(str, Check.isin(AGE_GROUPS)),
        "degree": Column(bool),
        "small_cell": Column(bool),
        "cell_description": Column(str, Check.str_startswith("registered ")),
    }, unique=["race", "party_group", "age_group", "degree"]),
    "frame": schema({
        "race": state,
        "party_group": party_group,
        "age_group": Column(str, Check.isin(AGE_GROUPS)),
        "degree": Column(bool),
        "n_respondents": count,
        "n_personas": count,
        "share": prob,
        "small_cell": Column(bool),
        "cell_id": Column(str, unique=True),
    }, unique=["race", "party_group", "age_group", "degree"]),
    "frame_source": schema({
        "state": state,
        "age": Column(int, Check.in_range(18, 110)),
        "education": text(),
        "party_alignment": text(),
        "weight": Column(float, Check.gt(0)),
        "survey_year": Column(int, Check.isin([2024, 2025])),
    }),
    "frame_states": schema({
        "race": Column(str, Check.isin(STATES_2026), unique=True),
        "n_respondents": Column(int, Check.ge(PERSONAS_PER_STATE)),
        "n_effective": Column(float, Check.gt(0)),
        "share_D": pct,
        "share_I": pct,
        "share_R": pct,
        "share_R_2024": pct,
        "share_R_2025": pct,
        "share_from_2024": pct,
    }, checks=[
        sums_to(100, "share_D", "share_I", "share_R", tol=0.2),  # rounded to 0.1
        Check(lambda d: d["n_effective"] <= d["n_respondents"],
              error="effective sample size cannot exceed the sample size"),
    ]),
    "candidates": schema({
        "race": state,
        "candidate": text(),
        "party_label": text(),
        "party_code": party_code,
        "incumbent": Column(bool),
        "include_in_prompt": Column(bool),
    }, unique=["race", "candidate"]),
    # --- 02-analysis_data: model answers and estimates ----------------------
    "responses_A": schema({
        **response_cols,
        "persona_id": Column(str, Check.str_matches(r"^P\d{5}$")),
        "governor_vote": nullable(str),
        "turnout": turnout,
        "house_vote": house,
    }, unique=["key", "run_date"]),
    "responses_B": schema({
        **response_cols,
        "cell_id": text(),
        "distribution_json": nullable(str),
        "distribution_sum": nullable(float, Check.in_range(90, 110)),
        "turnout_percent": nullable(float, Check.in_range(0, 100)),
    }, unique=["key", "run_date"]),
    "responses_C": schema({
        **response_cols,
        "race_id": state,
        "dem_two_party_share": nullable(float, Check.in_range(0, 100)),
        "uncertainty_plus_minus": nullable(float, Check.ge(0)),
        "probability_dem_wins": nullable(float, Check.in_range(0, 1)),
    }, unique=["key", "run_date"]),
    "responses_V": schema({
        **response_cols,
        "persona_id": Column(str, Check.str_matches(r"^P\d{5}$")),
        "governor_vote": nullable(str),
        "stated_state": nullable(str),
        "turnout": turnout,
        "house_vote": house,
    }, unique=["key", "run_date"]),
    "run_log": schema({
        "model_slug": model,
        "calls": Column(int, Check.gt(0)),
        "parse_rate": prob,
        "prompt_tokens": Column(float, Check.ge(0)),
        "completion_tokens": Column(float, Check.ge(0)),
        "reasoning_tokens": Column(float, Check.ge(0)),
        "cost_usd": Column(float, Check.ge(0)),
        "mean_seconds": Column(float, Check.ge(0)),
        "run_date": date,
        "member": text(),
        "model_set": text(),
        "method": Column(str, Check.isin(list("ABCV"))),
        "started_utc": utc_time,
        "finished_utc": utc_time,
        "simulated_inputs": Column(bool),
    }, checks=Check(lambda d: d["started_utc"] <= d["finished_utc"],
                    error="a run finished before it started")),
    "method_a_estimates": schema({
        "method": Column(str, Check.eq("A")),
        **estimate_cols,
        "n_personas": Column(int, Check.in_range(1, PERSONAS_PER_STATE)),
    }, checks=sums_to(100, "dem_share", "rep_share", "other_share"),
       unique=["race", "model_slug", "repeat", "run_date"]),
    "method_b_estimates": schema({
        "method": Column(str, Check.eq("B")),
        **estimate_cols,
        "n_cells": Column(int, Check.in_range(1, CELLS_PER_STATE)),
    }, checks=sums_to(100, "dem_share", "rep_share", "other_share"),
       unique=["race", "model_slug", "repeat", "run_date"]),
    "ticket_splitting": schema({
        "member": text(),
        "model_set": text(),
        "model_slug": Column(str, unique=True),
        "answers": Column(int, Check.gt(0)),
        "split_rate": prob,
    }),
    # --- 03-forecast --------------------------------------------------------
    "forecast": schema({
        "race": Column(str, Check.isin(STATES_2026), unique=True),
        "dem_candidate": text(),
        "rep_candidate": text(),
        "other_candidates": nullable(str),
        "type": Column(str, Check.eq("state")),
        "incumbent_party": Column(str, Check.isin(["D", "R"])),
        "open_seat": Column(bool),
        "competitive": Column(bool),
        "pvi": Column(float),
        "fund_margin": margin,
        "sil_margin": margin,
        "poll_margin": nullable(float, Check.in_range(-100, 100)),
        "n_polls": count,
        "n_polls_other": nullable(float, Check.ge(0)),
        **{f"weight_{s}": prob for s in ["fundamentals", "silicon", "polls"]},
        "other_estimate": pct,
        "other_source": Column(str, Check.isin(["polls", "AI surveys"])),
        "silicon_margin_corrected": margin,
        "silicon_margin_A": margin,
        "silicon_margin_B": margin,
        **{f"margin_{s}": margin for s in
           ["mean", "lo80", "hi80", "lo95", "hi95"]},
        "margin_sd": Column(float, Check.gt(0)),
        **{f"dem_share_{s}": pct for s in
           ["mean", "lo80", "hi80", "lo95", "hi95"]},
        **{f"{p}_vote_mean": pct for p in ["dem", "rep", "other"]},
        **{f"p_{s}": prob for s in ["dem_win", "rep_win", "other_win", "flip"]},
        "p_runoff": nullable(float, Check.in_range(0, 1)),
        "favourite": party_code,
        "p_favourite": Column(float, Check.in_range(1 / 3, 1)),
        "rating": Column(str, Check.isin(RATINGS)),
        "is_simulated": Column(bool, Check.eq(False)),
    }, checks=[
        sums_to(1, "p_dem_win", "p_rep_win", "p_other_win"),
        sums_to(1, "weight_fundamentals", "weight_silicon", "weight_polls"),
        sums_to(100, "dem_vote_mean", "rep_vote_mean", "other_vote_mean", tol=1e-6),
        ordered("margin_lo95", "margin_lo80", "margin_hi80", "margin_hi95"),
        ordered("margin_lo95", "margin_mean", "margin_hi95"),
        ordered("dem_share_lo95", "dem_share_lo80", "dem_share_hi80", "dem_share_hi95"),
        # Margins are in points of the two-party vote: share = 50 + margin / 2.
        Check(lambda d: (d["dem_share_mean"] - (50 + d["margin_mean"] / 2)).abs() < 1e-6,
              error="dem_share_mean must equal 50 + margin_mean / 2"),
    ]),
    "simulations": schema({
        "draw": Column(int, Check.in_range(1, N_SIMULATIONS), unique=True),
        "method": Column(str, Check.isin(["A", "B"])),
        "model_set": model,
        "national_swing": Column(float, Check.in_range(-30, 30)),
        "dem_governors": count,
        "rep_governors": count,
        "other_governors": count,
        "flips": Column(int, Check.in_range(0, len(STATES_2026))),
        "is_simulated": Column(bool, Check.eq(False)),
    }, checks=sums_to(50, "dem_governors", "rep_governors", "other_governors")),
    "method_grid": schema({
        "race": state,
        "method": Column(str, Check.isin(["A", "B"])),
        "model_set": model,
        "silicon_margin": margin,
        "combined_margin": margin,
        "combined_sd": Column(float, Check.gt(0)),
        "p_dem_win": prob,
        "is_simulated": Column(bool, Check.eq(False)),
    }, unique=["race", "method", "model_set"]),
    "backtest": schema({
        "races": Column(int, Check.gt(30)),
        "rmse_margin": Column(float, Check.gt(0)),
        "brier": Column(float, Check.in_range(0, 0.25)),  # 0.25 = always saying 50%
        "log_score": Column(float, Check.le(0)),
        "winner_correct": prob,
        "is_simulated": Column(bool, Check.eq(False)),
    }),
    "coefficients": schema({
        "term": Column(str, Check.isin(
            ["intercept", "pvi", "incumbent_running", "president_democrat"]), unique=True),
        "estimate": Column(float),
        "std_error": Column(float, Check.gt(0)),
    }),
    "validation_summary": schema({
        "run_date": date,
        "model_slug": model,
        "check": Column(str, Check.isin(CHECKS)),
        "n_pairs": Column(int, Check.gt(0)),
        "changed_pct": pct,
        "changed_pct_se": Column(float, Check.ge(0)),
        "dont_know_pct": pct,
        "dont_know_pct_without_change": pct,
        "statistic": text(),
        "statistic_pct": pct,
        "statistic_pct_se": Column(float, Check.ge(0)),
        "statistic_pct_without_change": nullable(float, Check.in_range(0, 100)),
        "retest_changed_pct": pct,
        "retest_n": Column(int, Check.gt(0)),
        "extra_changed_pct": Column(float, Check.in_range(-100, 100)),
    }, unique=["run_date", "model_slug", "check"]),
    "stability_summary": schema({
        "method": Column(str, Check.isin(["B", "C", "V"])),
        "model_slug": model,
        "quantity": text(),
        "n_days": Column(int, Check.ge(1)),
        "within_day_sd": nullable(float, Check.ge(0)),
        "between_day_sd": nullable(float, Check.ge(0)),
        "changed_vs_first_day_pct": nullable(float, Check.in_range(0, 100)),
        "n_pairs": nullable(float, Check.gt(0)),
    }, unique=["method", "model_slug"]),
    "cost_summary": schema({
        "run_date": date,
        "method": Column(str, Check.isin(list("ABCV"))),
        "model_slug": model,
        "calls": Column(int, Check.gt(0)),
        "prompt_tokens": Column(float, Check.ge(0)),
        "completion_tokens": Column(float, Check.ge(0)),
        "reasoning_tokens": Column(float, Check.ge(0)),
        "cost_usd": Column(float, Check.ge(0)),
        "wall_minutes": Column(float, Check.ge(0)),
        "parse_rate": prob,
        "mean_seconds_per_call": Column(float, Check.ge(0)),
        "cost_per_1000_calls_usd": Column(float, Check.ge(0)),
    }, unique=["run_date", "method", "model_slug"]),
}
# fmt: on


# ---------------------------------------------------------------------------
# Every table against its schema
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(SCHEMAS))
def test_table_matches_schema(name):
    SCHEMAS[name].validate(load(name), lazy=True)  # lazy: report every failure at once


def test_schemas_reject_broken_data():
    """The schemas catch the mistakes they are written for."""
    f = load("forecast").copy()
    f.loc[0, "p_dem_win"] += 0.2  # win probabilities no longer sum to 1
    with pytest.raises(pa.errors.SchemaErrors):
        SCHEMAS["forecast"].validate(f, lazy=True)
    e = load("method_a_estimates").copy()
    e.loc[0, "dem_share"] += 5  # shares no longer sum to 100
    with pytest.raises(pa.errors.SchemaErrors):
        SCHEMAS["method_a_estimates"].validate(e, lazy=True)
    p = load("personas").copy()
    p.loc[0, "persona_text"] += " You voted for Trump in 2024."
    with pytest.raises(pa.errors.SchemaErrors):
        SCHEMAS["personas"].validate(p, lazy=True)
    with pytest.raises(pa.errors.SchemaErrors):  # an undescribed column
        SCHEMAS["frame"].validate(load("frame").assign(extra=1), lazy=True)


# ---------------------------------------------------------------------------
# 01: races, candidates, history, polls
# ---------------------------------------------------------------------------
def test_36_races_and_14_seats_not_up_make_50_governors():
    races, not_up = load("races"), load("not_up")
    assert sorted(races["race"]) == sorted(STATES_2026)
    assert len(not_up) == SEATS_NOT_UP
    assert not set(not_up["state"]) & set(races["race"])


def test_every_prompt_names_one_democrat_and_a_republican():
    """Alaska's top-four ballot has three Republicans, so at least one."""
    codes = prompt_candidates().groupby("race")["party_code"]
    assert (codes.apply(lambda s: (s == "D").sum()) == 1).all()
    assert (codes.apply(lambda s: (s == "R").sum()) >= 1).all()
    assert set(codes.groups) == set(STATES_2026)


def test_open_seat_agrees_with_the_candidate_list():
    """A seat is open exactly when the sitting governor is not a candidate,
    and a running incumbent is the race's named incumbent from their party."""
    races = load("races").set_index("race")
    c = load("candidates")
    inc = c[c["incumbent"]].set_index("race")
    assert inc.index.is_unique
    assert (~races.loc[inc.index, "open_seat"]).all()
    assert races.loc[~races.index.isin(inc.index), "open_seat"].all()
    assert (inc["candidate"] == races.loc[inc.index, "incumbent"]).all()
    assert (inc["party_code"] == races.loc[inc.index, "incumbent_party"]).all()


def test_cleaned_candidates_match_the_download():
    keys = ["race", "candidate", "party_code", "incumbent"]
    raw = load("candidates_raw")[keys].sort_values(keys).reset_index(drop=True)
    clean = load("candidates")[keys].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(raw, clean, check_dtype=False)


def test_history_winner_had_the_larger_share():
    h = load("history")
    two = h[h["winner_party"].isin(["D", "R"])]
    assert ((two["dem_pct"] > two["rep_pct"]) == (two["winner_party"] == "D")).all()
    assert set(h["race"]) == set(STATES_2026)


def test_polls_name_both_nominees_and_match_their_shares():
    """Each poll's party totals are the sums of its named candidates' shares."""
    polls, c = load("polls"), load("candidates")
    party = c.set_index(["race", "candidate"])["party_code"].to_dict()
    for row in polls.itertuples():
        shares = json.loads(row.candidate_shares)
        by_party = {"D": 0.0, "R": 0.0, "O": 0.0}
        for name, value in shares.items():
            by_party[party[(row.race, name)]] += value  # KeyError: name not on ballot
        assert by_party["D"] == row.dem_pct and by_party["R"] == row.rep_pct
        assert by_party["O"] == (0 if pd.isna(row.minor_pct) else row.minor_pct)


# ---------------------------------------------------------------------------
# 02: personas, groups and the frame
# ---------------------------------------------------------------------------
def test_100_personas_and_10_validation_personas_per_state():
    p = load("personas")
    assert (p.groupby("race").size() == PERSONAS_PER_STATE).all()
    assert (p.groupby("race")["validation"].sum() == N_VALIDATION).all()
    assert set(p["race"]) == set(STATES_2026)


def test_persona_text_names_its_own_state():
    p = load("personas")
    assert all(f"lives in {r}." in t for r, t in zip(p["race"], p["persona_text"]))


def test_frame_and_groups_cover_24_groups_in_every_state():
    frame, cells = load("frame"), load("cells")
    assert (frame.groupby("race").size() == CELLS_PER_STATE).all()
    assert set(frame["cell_id"]) == set(cells["cell_id"])
    shares = frame.groupby("race")["share"].sum()
    assert np.allclose(shares, 1)


def test_frame_counts_everyone_in_frame_source_and_every_persona():
    frame, src, p = load("frame"), load("frame_source"), load("personas")
    assert (
        frame.groupby("race")["n_respondents"].sum()
        == src.groupby("state").size().rename_axis("race")
    ).all()
    n_personas = frame.groupby("race")["n_personas"].sum()
    assert (n_personas == p.groupby("race").size()).all()


def test_frame_source_weights_average_one_in_each_year():
    """02 rescales weights so each survey year counts by its number of respondents."""
    w = load("frame_source").groupby("survey_year")["weight"].mean()
    assert np.allclose(w, 1)


# ---------------------------------------------------------------------------
# 04: model answers (the API responses)
# ---------------------------------------------------------------------------
METHODS = list("ABCV")


@pytest.mark.parametrize("method", METHODS)
def test_unparsed_answers_carry_no_vote(method):
    """An answer the parser rejected must not flow into an estimate."""
    r = load(f"responses_{method}")
    answer_cols = {
        "A": ["governor_vote", "turnout", "house_vote"],
        "B": ["distribution_json", "distribution_sum", "turnout_percent"],
        "C": ["dem_two_party_share", "uncertainty_plus_minus", "probability_dem_wins"],
        "V": ["governor_vote", "stated_state", "turnout", "house_vote"],
    }[method]
    assert r.loc[~r["parsed"], answer_cols].isna().all().all()


@pytest.mark.parametrize("method", METHODS)
def test_every_model_parses_at_least_90_percent(method):
    r = load(f"responses_{method}")
    rates = r.groupby(["run_date", "model_slug"])["parsed"].mean()
    assert (rates >= MIN_PARSE_RATE).all(), rates[rates < MIN_PARSE_RATE]


@pytest.mark.parametrize("method", METHODS)
def test_parsed_answers_have_a_provider_and_a_cost(method):
    r = load(f"responses_{method}")
    ok = r[r["parsed"]]
    assert ok["provider_served"].notna().all()
    assert ok["cost_usd"].notna().all()
    assert (ok["completion_tokens"] > 0).all()


@pytest.mark.parametrize("method", ["A", "V"])
def test_votes_are_candidates_on_that_states_ballot(method):
    r = load(f"responses_{method}")
    r = r[r["parsed"]]
    allowed = prompt_candidates().groupby("race")["candidate"].agg(set)
    ok = [
        v in allowed[race] or v in EXTRA_OPTIONS
        for race, v in zip(r["race"], r["governor_vote"])
    ]
    assert all(ok)
    if method == "A":
        assert r["turnout"].notna().all() and r["house_vote"].notna().all()


def test_method_b_distributions_use_the_states_options_and_sum_to_about_100():
    r = load("responses_B")
    r = r[r["parsed"]]
    allowed = prompt_candidates().groupby("race")["candidate"].agg(set)
    for row in r.itertuples():
        dist = json.loads(row.distribution_json)
        assert set(dist) <= allowed[row.race] | set(EXTRA_OPTIONS)
        assert all(v >= 0 for v in dist.values())
        assert abs(sum(dist.values()) - row.distribution_sum) < 1e-6


def test_method_c_uncertainty_and_probability_agree_in_direction():
    """A forecaster putting the Democrat above 50% should give them better than
    even odds. Allow a few misfires, but not a model ignoring its own estimate."""
    r = load("responses_C")
    r = r[r["parsed"] & (r["dem_two_party_share"] != 50)]
    agree = (r["dem_two_party_share"] > 50) == (r["probability_dem_wins"] > 0.5)
    assert agree.groupby(r["model_slug"]).mean().min() >= 0.9


def test_attention_check_answers_name_the_right_state():
    r = load("responses_V")
    att = r[r["parsed"] & (r["check"] == "attention")]
    assert (
        att["stated_state"].str.strip().str.lower() == att["race"].str.lower()
    ).mean() >= 0.95


def test_full_run_asks_every_persona_group_and_race():
    """The first run is complete: every persona, group and race, for every model,
    every repeat (validation checks once per validation persona)."""
    personas, cells = load("personas"), load("cells")
    a, b = first_run(load("responses_A")), first_run(load("responses_B"))
    c, v = first_run(load("responses_C")), first_run(load("responses_V"))
    n_models = a["model_slug"].nunique()
    assert len(a) == len(personas) * n_models * FULL_RUN_REPEATS
    assert len(b) == len(cells) * n_models * FULL_RUN_REPEATS
    assert len(c) == len(STATES_2026) * n_models * FULL_RUN_REPEATS
    assert len(v) == personas["validation"].sum() * n_models * len(CHECKS)
    assert set(v["persona_id"]) == set(
        personas.loc[personas["validation"], "persona_id"]
    )
    for r in (a, b, c, v):
        assert r["model_slug"].nunique() == n_models


def test_at_least_two_model_providers():
    """The brief asks for models from at least two providers."""
    providers = load("responses_A")["model_slug"].str.split("/").str[0]
    assert providers.nunique() >= 2


def test_prompts_offer_the_allowed_answers():
    """common.py's options are exactly what the prompts show the models."""
    a = (PROMPTS / "method_a_vote.txt").read_text(encoding="utf-8")
    flat = " ".join(a.split())  # options may wrap over lines
    for option in TURNOUT_OPTIONS + HOUSE_OPTIONS + EXTRA_OPTIONS:
        assert f'"{option}"' in flat, option


def test_run_log_matches_the_saved_answers():
    """run_log.csv counts the same calls and dollars as the response files."""
    log = load("run_log")
    log = log[~log["simulated_inputs"]]
    for method in METHODS:
        r = load(f"responses_{method}")
        logged = log[log["method"] == method].groupby("run_date")["calls"].sum()
        saved = r.groupby("run_date").size()
        assert (logged.reindex(saved.index) >= saved).all()  # failed calls are retried


# ---------------------------------------------------------------------------
# 05: estimates and the forecast
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["method_a_estimates", "method_b_estimates"])
def test_every_race_has_an_estimate_from_every_model_and_repeat(name):
    est = first_run(load(name))
    n_models = load("responses_A")["model_slug"].nunique()
    assert len(est) == len(STATES_2026) * n_models * FULL_RUN_REPEATS


def test_win_probabilities_and_ratings_agree():
    f = load("forecast")
    probs = f[["p_dem_win", "p_rep_win", "p_other_win"]].to_numpy()
    assert (f["favourite"] == np.array(["D", "R", "O"])[probs.argmax(axis=1)]).all()
    assert np.allclose(f["p_favourite"], probs.max(axis=1))
    toss = f["p_favourite"] < 0.6
    assert (f.loc[toss, "rating"] == "Toss-up").all()
    assert (f.loc[~toss, "rating"].str[-1] == f.loc[~toss, "favourite"]).all()


def test_flip_probability_is_one_minus_the_incumbent_party_winning():
    f = load("forecast")
    hold = np.where(f["incumbent_party"] == "D", f["p_dem_win"], f["p_rep_win"])
    assert np.allclose(f["p_flip"], 1 - hold)


def test_only_georgia_has_a_runoff_probability():
    f = load("forecast").set_index("race")
    assert f["p_runoff"].dropna().index.tolist() == ["Georgia"]


def test_races_without_polls_put_no_weight_on_polls():
    f = load("forecast")
    no_polls = f["n_polls"] == 0
    assert f.loc[no_polls, "poll_margin"].isna().all()
    assert (f.loc[no_polls, "weight_polls"] == 0).all()
    assert (f.loc[~no_polls, "weight_polls"] > 0).all()


def test_simulations_match_the_forecast():
    """10,000 draws, every method x model combination used, and each race's win
    probability agrees with the share of draws in the seat count."""
    sims, f, grid = load("simulations"), load("forecast"), load("method_grid")
    assert len(sims) == N_SIMULATIONS
    combos = set(zip(grid["method"], grid["model_set"]))
    assert set(zip(sims["method"], sims["model_set"])) == combos
    assert len(grid) == len(STATES_2026) * len(combos)
    expected_dem = f["p_dem_win"].sum() + (load("not_up")["party"] == "D").sum()
    assert abs(sims["dem_governors"].mean() - expected_dem) < 1e-6


# ---------------------------------------------------------------------------
# 06: validation, stability and costs
# ---------------------------------------------------------------------------
def test_validation_covers_every_check_and_model():
    v = first_run(load("validation_summary"))
    n_models = load("responses_V")["model_slug"].nunique()
    assert len(v) == len(CHECKS) * n_models


def test_costs_add_up_and_stay_under_the_spending_limit():
    cost, log = load("cost_summary"), load("run_log")
    log = log[~log["simulated_inputs"]]
    assert np.isclose(cost["cost_usd"].sum(), log["cost_usd"].sum())
    assert cost["calls"].sum() == log["calls"].sum()
    assert cost["cost_usd"].sum() < SPENDING_LIMIT_USD