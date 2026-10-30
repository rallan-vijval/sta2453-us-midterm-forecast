"""
Purpose : Send every prompt to every model in one member's config through
          OpenRouter, and save each answer with its cost and timing.
Author  : Shrey Sati (Member 2)
Date    : 2026-10-22
Inputs  : config/models_memberX.yaml          the member's model list and settings
          prompts/*.txt, prompts/validation/*.txt
          data/02-analysis_data/personas.parquet, cells.parquet, candidates.parquet
          data/01-raw_data/governor_races_2026.csv
          .env with OPENROUTER_API_KEY (or the environment, in GitHub Actions)
Outputs : data/02-analysis_data/responses/{member}_{method}_{date}.parquet
          data/02-analysis_data/run_log.csv   one row per model per run
Methods : A   each persona: turnout, governor vote, House vote
          B   each demographic group: percentage split across the candidates
          C   each race (states): a direct "forecaster" answer
          V   validation checks on the 10 flagged personas per state
Notes   : - Placeholders are filled with str.replace (the JSON examples contain braces).
          - Candidate order is shuffled per call with a seed built from the
            persona, model and repeat, so a rerun asks in the same order.
          - Each provider is pinned and fallbacks are off, so the host that
            answered is the host in the config.
          - An answer that cannot be parsed is retried once, then saved with
            parsed = False. No row is ever dropped.
          - A rerun skips anything already saved for that day (resume).
Run     : uv run python scripts/04-query_models.py --config config/models_member3.yaml --method A --states all
          Add --states MI,GA --limit 10 --repeats 1 for a pilot, --dry-run to print
          prompts without calling any model, --max-cost 2 to stop after US$2.
"""

from common import (
    ABBR_STATE,
    ANA,
    EXTRA_OPTIONS,
    HOUSE_OPTIONS,
    RESP,
    ROOT,
    SIM,
    STATES_2026,
    TURNOUT_OPTIONS,
    candidate_lines,
    fill,
    load_candidates,
    load_races,
    match_option,
    quoted_names,
    read_prompt,
    stable_seed,
    strip_party_sentence,
)
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

URL = "https://openrouter.ai/api/v1/chat/completions"
CHUNK = 400
VALIDATION_CHECKS = [
    "attention",
    "party_drop",
    "incumbent_drop",
    "label_drop",
    "order_reverse",
    "governor_only",
]


# ---------------------------------------------------------------------------
# Building the jobs (one job = one API call)
# ---------------------------------------------------------------------------
def ordered_candidates(cands: pd.DataFrame, *seed_parts) -> pd.DataFrame:
    order = list(range(len(cands)))
    random.Random(stable_seed(*seed_parts)).shuffle(order)
    return cands.iloc[order]


def build_jobs(method: str, cfg: dict, args, data: dict) -> list[dict]:
    cands_all = data["candidates"]
    cands_all = cands_all[cands_all["include_in_prompt"]]
    by_race = {r: g.reset_index(drop=True)
               for r, g in cands_all.groupby("race")}
    repeats = args.repeats or cfg.get("repeats", 3)
    races = set(data["races"])
    if args.states != "all":
        races &= {
            ABBR_STATE.get(s.strip().upper(), s.strip()) for s in args.states.split(",")
        }
    system = read_prompt("system.txt")
    jobs = []
    for m in cfg["models"]:
        slug = m["slug"]
        if method in ("A", "V"):
            p = data["personas"]
            p = p[p["race"].isin(races)]
            if method == "V" or args.validation_only:
                p = p[p["validation"]]
            if args.limit:
                p = p.groupby("race").head(args.limit)
            template = read_prompt("method_a_vote.txt")
            for row in p.itertuples():
                cands = by_race.get(row.race)
                if cands is None:
                    continue
                if method == "A":
                    for rep in range(1, repeats + 1):
                        c = ordered_candidates(
                            cands, row.persona_id, slug, rep)
                        user = fill(
                            template,
                            persona=row.persona_text,
                            state=row.race,
                            candidate_list=candidate_lines(c),
                            candidate_names=quoted_names(c),
                        )
                        jobs.append(
                            {
                                "key": f"{row.persona_id}|{slug}|{rep}",
                                "method": "A",
                                "model": m,
                                "system": system,
                                "user": user,
                                "race": row.race,
                                "unit_id": row.persona_id,
                                "repeat": rep,
                                "check": "",
                                "candidate_order": list(c["candidate"]),
                                "options": list(c["candidate"]) + EXTRA_OPTIONS,
                            }
                        )
                else:
                    base = ordered_candidates(
                        cands, row.persona_id, slug, 1
                    )  # same order as Method A repeat 1
                    for check in VALIDATION_CHECKS:
                        c = base.iloc[::-
                                      1] if check == "order_reverse" else base
                        values = {
                            "persona": row.persona_text,
                            "persona_without_party": strip_party_sentence(
                                row.persona_text
                            ),
                            "state": row.race,
                            "candidate_names": quoted_names(c),
                            "candidate_list": candidate_lines(c),
                            "candidate_list_without_incumbent": candidate_lines(
                                c, with_incumbent=False
                            ),
                            "candidate_list_without_party": candidate_lines(
                                c, with_party=False, with_incumbent=False
                            ),
                            "candidate_list_reversed": candidate_lines(c),
                        }
                        user = fill(read_prompt(
                            f"validation/{check}.txt"), **values)
                        jobs.append(
                            {
                                "key": f"{row.persona_id}|{slug}|{check}",
                                "method": "V",
                                "model": m,
                                "system": system,
                                "user": user,
                                "race": row.race,
                                "unit_id": row.persona_id,
                                "repeat": 1,
                                "check": check,
                                "candidate_order": list(c["candidate"]),
                                "options": list(c["candidate"]) + EXTRA_OPTIONS,
                            }
                        )
        elif method == "B":
            cells = data["cells"]
            cells = cells[cells["race"].isin(races)]
            if args.limit:
                cells = cells.groupby("race").head(args.limit)
            template = read_prompt("method_b_cell.txt")
            system_b = read_prompt("system_group.txt")
            for row in cells.itertuples():
                cands = by_race.get(row.race)
                if cands is None:
                    continue
                for rep in range(1, repeats + 1):
                    c = ordered_candidates(cands, row.cell_id, slug, rep)
                    keys = list(c["candidate"]) + EXTRA_OPTIONS
                    dist = "{" + \
                        ", ".join(f'"{k}": <number>' for k in keys) + "}"
                    user = fill(
                        template,
                        cell_description=row.cell_description,
                        state=row.race,
                        candidate_list=candidate_lines(c),
                        distribution_template=dist,
                    )
                    jobs.append(
                        {
                            "key": f"{row.cell_id}|{slug}|{rep}",
                            "method": "B",
                            "model": m,
                            "system": system_b,
                            "user": user,
                            "race": row.race,
                            "unit_id": row.cell_id,
                            "repeat": rep,
                            "check": "",
                            "candidate_order": list(c["candidate"]),
                            "options": keys,
                        }
                    )
        elif method == "C":
            template = read_prompt("method_c_forecaster.txt")
            system_c = read_prompt("system_forecaster.txt")
            all_races = set(data["races_all"]
                            ) if args.states == "all" else races
            for race in sorted(all_races):
                cands = by_race.get(race)
                if cands is None:
                    continue
                for rep in range(1, repeats + 1):
                    c = ordered_candidates(cands, race, slug, rep)
                    user = fill(template, state=race,
                                candidate_list=candidate_lines(c))
                    jobs.append(
                        {
                            "key": f"{race}|{slug}|{rep}",
                            "method": "C",
                            "model": m,
                            "system": system_c,
                            "user": user,
                            "race": race,
                            "unit_id": race,
                            "repeat": rep,
                            "check": "",
                            "candidate_order": list(c["candidate"]),
                            "options": [],
                        }
                    )
    return jobs


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def extract_json(text: str):
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def parse(job: dict, text: str) -> tuple[dict, bool]:
    """Return (parsed fields, ok)."""
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return {}, False
    method, check = job["method"], job["check"]
    if method == "A" or (method == "V" and check != "attention"):
        out = {"governor_vote": match_option(
            obj.get("governor_vote"), job["options"])}
        out["turnout"] = match_option(obj.get("turnout"), TURNOUT_OPTIONS)
        needs = ["governor_vote", "turnout"]
        if method == "A" or check != "governor_only":
            out["house_vote"] = match_option(
                obj.get("house_vote"), HOUSE_OPTIONS)
            needs.append("house_vote")
        return out, all(out[k] is not None for k in needs)
    if check == "attention":
        out = {
            "governor_vote": match_option(obj.get("governor_vote"), job["options"]),
            "stated_state": str(obj.get("state", "")).strip(),
        }
        return out, out["governor_vote"] is not None and out["stated_state"] != ""
    if method == "B":
        dist = obj.get("distribution")
        if not isinstance(dist, dict):
            return {}, False
        clean = {}
        for k, v in dist.items():
            opt = match_option(k, job["options"])
            try:
                val = float(v)
            except (TypeError, ValueError):
                return {}, False
            if opt is None or val < 0:
                return {}, False
            clean[opt] = clean.get(opt, 0) + val
        total = sum(clean.values())
        try:
            turnout = float(obj.get("turnout_percent"))
        except (TypeError, ValueError):
            turnout = None
        ok = 90 <= total <= 110 and turnout is not None and 0 <= turnout <= 100
        return {
            "distribution_json": json.dumps(clean),
            "distribution_sum": total,
            "turnout_percent": turnout,
        }, ok
    if method == "C":
        try:
            share = float(obj["dem_two_party_share"])
            unc = float(obj["uncertainty_plus_minus"])
            prob = float(obj["probability_dem_wins"])
        except (KeyError, TypeError, ValueError):
            return {}, False
        ok = 0 <= share <= 100 and unc >= 0 and 0 <= prob <= 1
        return {
            "dem_two_party_share": share,
            "uncertainty_plus_minus": unc,
            "probability_dem_wins": prob,
        }, ok
    return {}, False


# ---------------------------------------------------------------------------
# Calling OpenRouter
# ---------------------------------------------------------------------------
def request_body(job: dict, cfg: dict) -> dict:
    m = job["model"]
    body = {
        "model": m["slug"],
        "messages": [
            {"role": "system", "content": job["system"]},
            {"role": "user", "content": job["user"]},
        ],
        "max_tokens": m.get("max_tokens", cfg.get("max_tokens", 300)),
        "usage": {"include": True},
    }
    # A model can override the temperature; `temperature: null` sends none, for
    # models (such as OpenAI reasoning models) that only use their default.
    temperature = m.get("temperature", cfg.get("temperature", 1.0))
    if temperature is not None:
        body["temperature"] = temperature
    if m.get("json_mode", True):
        body["response_format"] = {"type": "json_object"}
    if m.get("reasoning_effort"):
        body["reasoning"] = {"effort": m["reasoning_effort"], "exclude": True}
    if m.get("provider_order"):
        body["provider"] = {
            "order": m["provider_order"],
            "allow_fallbacks": False,
            "require_parameters": cfg.get("require_parameters", True),
        }
    return body


async def post(client: httpx.AsyncClient, body: dict, headers: dict) -> dict:
    for attempt in range(6):
        try:
            r = await client.post(URL, json=body, headers=headers, timeout=120)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            err = f"network: {type(e).__name__}"
        else:
            if r.status_code == 200:
                data = r.json()
                if "choices" in data:
                    return data
                err = f"no choices: {str(data)[:300]}"
            elif r.status_code in (408, 429, 500, 502, 503, 504):
                err = f"HTTP {r.status_code}"
            else:
                return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
        await asyncio.sleep(min(60, 2**attempt + random.random()))
    return {"error": f"gave up after retries ({err})"}


async def run_job(job, cfg, client, sem, headers):
    async with sem:
        t0 = time.time()
        body = request_body(job, cfg)
        texts, usage, provider, error, fields, ok, tries = (
            [],
            {},
            None,
            None,
            {},
            False,
            0,
        )
        for tries in (1, 2):  # one retry if the answer cannot be parsed
            data = await post(client, body, headers)
            if "error" in data:
                error = data["error"]
                break
            text = (data["choices"][0].get("message")
                    or {}).get("content") or ""
            texts.append(text)
            u = data.get("usage") or {}
            for k in ("prompt_tokens", "completion_tokens", "cost"):
                usage[k] = usage.get(k, 0) + (u.get(k) or 0)
            details = u.get("completion_tokens_details") or {}
            usage["reasoning_tokens"] = usage.get("reasoning_tokens", 0) + (
                details.get("reasoning_tokens") or 0
            )
            provider = data.get("provider")
            fields, ok = parse(job, text)
            if ok:
                break
        row = {
            "key": job["key"],
            "method": job["method"],
            "check": job["check"],
            "model_slug": job["model"]["slug"],
            "provider_pinned": ",".join(job["model"].get("provider_order", [])),
            "provider_served": provider,
            "race": job["race"],
            "unit_id": job["unit_id"],
            "repeat": job["repeat"],
            "candidate_order": json.dumps(job["candidate_order"]),
            "raw_text": texts[-1] if texts else None,
            "parsed": bool(ok),
            "retries": tries - 1,
            "error": error,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "cost_usd": usage.get("cost"),
            "seconds": round(time.time() - t0, 2),
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        row[
            "persona_id"
            if job["method"] in ("A", "V")
            else "cell_id"
            if job["method"] == "B"
            else "race_id"
        ] = job["unit_id"]
        # An answer that failed parsing keeps its raw text but contributes no fields,
        # so a half-read answer can never reach the forecast.
        row.update(fields if ok else {k: None for k in fields})
        return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_inputs(simulated: bool) -> dict:
    src = SIM if simulated else ANA
    races = load_races(simulated)
    return {
        "personas": pd.read_parquet(src / "personas.parquet"),
        "cells": pd.read_parquet(src / "cells.parquet"),
        "candidates": load_candidates(simulated),
        "races": [r for r in races["race"] if r in STATES_2026],
        "races_all": list(races["race"]),
    }


async def main(args):
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(Path(args.config).read_text())
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and not args.dry_run:
        raise SystemExit("OPENROUTER_API_KEY is not set (put it in .env).")
    data = load_inputs(args.simulated)
    method = args.method.upper()
    jobs = build_jobs(method, cfg, args, data)

    run_date = args.run_date or date.today().isoformat()
    RESP.mkdir(parents=True, exist_ok=True)
    tag = "_simulated" if args.simulated else ""
    out_path = RESP / f"{cfg['member']}_{method}_{run_date}{tag}.parquet"
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    done_keys = (
        set(done.loc[done.get("parsed", pd.Series(
            dtype=bool)).astype(bool), "key"])
        if len(done)
        else set()
    )
    todo = [j for j in jobs if j["key"] not in done_keys]
    print(
        f"{cfg['member']} / method {method}: {len(jobs):,} calls planned, {len(jobs) - len(todo):,} already saved, "
        f"{len(todo):,} to run{' (dry run)' if args.dry_run else ''}."
    )
    if args.dry_run:
        if todo:
            print(
                "\n--- first prompt ---\nSYSTEM:\n"
                + todo[0]["system"]
                + "\n\nUSER:\n"
                + todo[0]["user"]
                + "\n---"
            )
        return

    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/rallan-vijval/sta2453-us-midterm-forecast",
        "X-Title": "STA2453 governors 2026",
    }
    sem = asyncio.Semaphore(cfg.get("concurrency", 8))
    started = datetime.now(UTC).isoformat(timespec="seconds")
    spent, rows = 0.0, []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i: i + CHUNK]
            rows += await asyncio.gather(
                *(run_job(j, cfg, client, sem, headers) for j in chunk)
            )
            new = pd.DataFrame(rows)
            new = new.assign(
                run_date=run_date,
                member=cfg["member"],
                model_set=cfg["model_set"],
                prompt_version=args.prompt_version,
            )
            combined = pd.concat(
                [done[~done["key"].isin(new["key"])]
                 if len(done) else done, new],
                ignore_index=True,
            )
            combined.to_parquet(out_path, index=False)
            spent = float(
                pd.to_numeric(new["cost_usd"], errors="coerce").fillna(0).sum()
            )
            print(
                f"  {min(i + CHUNK, len(todo)):,}/{len(todo):,} done, "
                f"{new['parsed'].mean():.0%} parsed, US${spent:.3f} so far"
            )
            if args.max_cost and spent >= args.max_cost:
                print(
                    f"Stopped: spending reached the --max-cost limit of US${args.max_cost}."
                )
                break

    if rows:
        new = pd.DataFrame(rows)
        log = (
            new.groupby("model_slug")
            .agg(
                calls=("key", "size"),
                parse_rate=("parsed", "mean"),
                prompt_tokens=("prompt_tokens", "sum"),
                completion_tokens=("completion_tokens", "sum"),
                reasoning_tokens=("reasoning_tokens", "sum"),
                cost_usd=("cost_usd", "sum"),
                mean_seconds=("seconds", "mean"),
            )
            .reset_index()
        )
        log = log.assign(
            run_date=run_date,
            member=cfg["member"],
            model_set=cfg["model_set"],
            method=method,
            started_utc=started,
            finished_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            simulated_inputs=args.simulated,
        )
        log_path = ANA / "run_log.csv"
        log = (
            pd.concat([pd.read_csv(log_path), log], ignore_index=True)
            if log_path.exists()
            else log
        )
        log.to_csv(log_path, index=False)
        print(f"Saved {out_path.name}; run_log.csv updated.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Query OpenRouter models for one member's config."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--method",
        required=True,
        choices=["A", "B", "C", "V", "a", "b", "c", "v"],
        help="A personas, B groups, C forecaster, V validation checks",
    )
    ap.add_argument("--states", default="all",
                    help="'all' or a list such as MI,GA")
    ap.add_argument(
        "--limit", type=int, default=0, help="personas or groups per state (pilot)"
    )
    ap.add_argument(
        "--repeats", type=int, default=0, help="override the config's repeats"
    )
    ap.add_argument(
        "--validation-only",
        action="store_true",
        help="Method A on the 10 validation personas per state",
    )
    ap.add_argument(
        "--max-cost", type=float, default=0, help="stop after this many US dollars"
    )
    ap.add_argument(
        "--run-date", default="", help="label for the output file (default: today)"
    )
    ap.add_argument("--prompt-version", default="v1")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the first prompt; call no model"
    )
    ap.add_argument(
        "--simulated", action="store_true", help="use data/00-simulated_data inputs"
    )
    asyncio.run(main(ap.parse_args()))
