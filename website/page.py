"""
Purpose : Everything the website shows that is not prose: data loading, the colour
          palette, one table style, the headline numbers, the seat bar and the
          ratings strip. index.qmd and the section files call these functions, so
          every table and figure looks the same and nothing is computed twice.
Author  : Vijval Rallan, Shrey Sati, Allwin Steve Paul Justin Anantharaj
Date    : 2026-10-28
Inputs  : data/03-forecast/*, data/02-analysis_data/*, data/01-raw_data/*, other/charts/*.png
Notes   : Reads files only; the forecast itself is made by scripts/05-forecast.py.
          The colours match scripts/07-plot_forecast.py and website/styles.scss.
"""

import html
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from great_tables import GT, style
from IPython.display import HTML, Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "01-raw_data"
ANA = ROOT / "data" / "02-analysis_data"
FC = ROOT / "data" / "03-forecast"
CHARTS = ROOT / "other" / "charts"

# ---------------------------------------------------------------------------
# Palette (same as styles.scss and 07-plot_forecast.py)
# ---------------------------------------------------------------------------
GROUND, SURFACE, RULE = "#1A1F26", "#222934", "#343C48"
TEXT, MUTED = "#E6E8EB", "#9AA3AF"
RATING_COLOUR = {
    "Solid D": "#2F6BC2", "Likely D": "#5C8FD8", "Lean D": "#A3C3EE",
    "Toss-up": "#E3B341",
    "Lean R": "#F2B1A8", "Likely R": "#E06E60", "Solid R": "#B9412F",
    "Lean O": "#C9B6E8", "Likely O": "#A58BD0", "Solid O": "#8668BF",
}  # fmt: skip
# The darkest fills are too dim as text on the dark page, so words and numbers use a lighter shade.
LIGHT_D, LIGHT_R, LIGHT_O = "#7EA9EA", "#EE8A7E", "#B9A2DE"
RATING_TEXT = {**RATING_COLOUR, "Solid D": LIGHT_D, "Likely D": LIGHT_D, "Solid R": LIGHT_R,
               "Likely R": LIGHT_R, "Solid O": LIGHT_O, "Likely O": LIGHT_O}  # fmt: skip
RATING_ORDER = [
    "Solid D",
    "Likely D",
    "Lean D",
    "Toss-up",
    "Lean R",
    "Likely R",
    "Solid R",
]
PARTY_NAME = {"D": "Democratic", "R": "Republican", "O": "Independent"}
MODEL_NAME = {
    "openai/gpt-6-luna": "GPT-6 Luna (OpenAI)",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "mistralai/mistral-small-24b-instruct-2501": "Mistral Small 24B",
}
METHOD_NAME = {
    "A": "A: each persona votes",
    "B": "B: each group's split",
    "C": "C: model as forecaster",
    "V": "Validation checks",
}
CHECK_NAME = {
    "attention": "Asked which state they live in",
    "governor_only": "House question removed",
    "incumbent_drop": "Incumbent label removed",
    "order_reverse": "Candidate order reversed",
    "label_drop": "Party labels on candidates removed",
    "party_drop": "Persona's own party removed",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@cache
def forecast() -> pd.DataFrame:
    f = pd.read_parquet(FC / "forecast.parquet")
    return f.assign(closeness=(f["p_favourite"] - 0.5).abs()).sort_values("closeness")


@cache
def simulations() -> pd.DataFrame:
    return pd.read_parquet(FC / "simulations.parquet")


@cache
def run_info() -> dict:
    return json.loads((FC / "run_info.json").read_text())


@cache
def not_up() -> pd.DataFrame:
    return pd.read_csv(RAW / "governors_not_up_2026.csv")


@cache
def responses(method: str) -> pd.DataFrame:
    files = sorted((ANA / "responses").glob(f"*_{method}_*.parquet"))
    r = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    r["parsed"] = r["parsed"].astype(bool)
    return r


@cache
def method_c_margin() -> pd.Series:
    """Method C, the model asked as a forecaster: mean Democratic margin per race
    (two-party share turned into a margin), over models, repeats and days."""
    c = responses("C")
    c = c[c["parsed"]]
    return (2 * c.groupby("race")["dem_two_party_share"].mean() - 100).rename(
        "method_c"
    )


def chart(name: str, scale: float = 0.55) -> Image:
    """A chart drawn by scripts/07-plot_forecast.py, shown at a size where its text matches
    the page. The charts are saved at 200 pixels per inch; at 0.55 screen pixels per image
    pixel their 11.5-point axis labels come out at about 17 pixels, like the body text.
    Charts with small labels (the map, the scatter) get a larger scale.
    CSS still shrinks any chart to fit the column or a phone screen."""
    path = CHARTS / f"{name}.png"
    width_px = int.from_bytes(
        path.read_bytes()[16:20], "big"
    )  # a PNG stores its width at bytes 16-19
    return Image(filename=str(path), width=round(width_px * scale))


# ---------------------------------------------------------------------------
# Words and numbers
# ---------------------------------------------------------------------------
def pct(p: float) -> str:
    """A probability as a whole percentage; 10,000 simulations cannot show certainty."""
    if p >= 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{p:.0%}"


def lean(m: float, decimals: int = 0) -> str:
    """A margin (Democrat minus Republican, points) as D+5 or R+12."""
    if pd.isna(m):
        return ""
    if abs(m) < 0.5 * 10**-decimals:
        return "Even"
    return f"{'D' if m > 0 else 'R'}+{abs(m):.{decimals}f}"


def day(t: pd.Timestamp, short: bool = False) -> str:
    """'28 October' (or '28 Oct'). Built by hand because Windows strftime has no %-d."""
    return f"{t.day} {t:%b}" if short else f"{t.day} {t:%B}"


def when(t: pd.Timestamp) -> str:
    """'28 October 2026, 12:10 PM Toronto time'."""
    hour = t.hour % 12 or 12
    return f"{day(t)} {t.year}, {hour}:{t:%M} {t:%p} Toronto time"


def dollars(x: float) -> str:
    """US dollars to the cent; amounts below half a cent read "<$0.01", not "$0.00"."""
    if pd.isna(x):
        return ""
    return "<$0.01" if x < 0.005 else f"${x:,.2f}"


def headline() -> dict:
    """Numbers for the header and for inline text."""
    f, sims, info = forecast(), simulations(), run_info()
    dem = sims["dem_governors"]
    lead = (
        "D" if (f["favourite"] == "D").sum() >= (f["favourite"] == "R").sum() else "R"
    )
    return {
        "lead_party": {"D": "Democrats", "R": "Republicans"}[lead],
        "lead_count": int((f["favourite"] == lead).sum()),
        "updated": when(pd.Timestamp(info["run_utc"]).tz_convert("America/Toronto")),
        "dem_favoured": int((f["favourite"] == "D").sum()),
        "rep_favoured": int((f["favourite"] == "R").sum()),
        "toss_ups": int((f["rating"] == "Toss-up").sum()),
        "dem_median": int(dem.median()),
        "dem_lo80": int(dem.quantile(0.1)),
        "dem_hi80": int(dem.quantile(0.9)),
        "p_dem_majority": pct(info["p_dem_majority"]),
        "p_rep_majority": pct(info["p_rep_majority"]),
        "n_races": len(f),
        "n_sims": f"{info['n_simulations']:,}",
        "flips_median": int(info["flips_median"]),
    }


# ---------------------------------------------------------------------------
# One table style for the whole site
# ---------------------------------------------------------------------------
def table(df: pd.DataFrame, **kw) -> GT:
    """A great_tables table in the site's style; format columns after calling this."""
    return (
        GT(df, **kw)
        .tab_options(
            table_background_color=GROUND,
            table_font_color=TEXT,
            table_font_names=["Karla", "Helvetica Neue", "Arial", "sans-serif"],
            table_font_size="15px",
            table_width="100%",
            table_border_top_style="none",
            table_border_bottom_color=RULE,
            column_labels_font_weight="600",
            column_labels_background_color=GROUND,
            column_labels_border_top_style="none",
            column_labels_border_bottom_color=MUTED,
            column_labels_border_bottom_width="1px",
            table_body_border_bottom_color=RULE,
            table_body_hlines_color=RULE,
            row_group_border_top_color=RULE,
            row_group_border_bottom_color=RULE,
            row_group_font_weight="600",
            row_group_background_color=SURFACE,
            stub_border_color=RULE,
            data_row_padding="6px",
            stub_background_color=GROUND,
            stub_font_weight="500",
            source_notes_font_size="12px",
            grand_summary_row_background_color=SURFACE,
        )
        .tab_style(style=style.text(color=MUTED, weight="500"), locations=_labels())
    )


def _labels():
    from great_tables import loc

    return loc.column_labels()


# ---------------------------------------------------------------------------
# Header: the 50 governorships as a row of squares
# ---------------------------------------------------------------------------
def seat_bar() -> HTML:
    """Fifty squares, one per governorship: the 14 not up this year (outlined) and the
    36 races coloured by their rating, Democratic on the left."""
    f, nu = forecast(), not_up()
    rows = [(f"{s}: governor {g} ({PARTY_NAME[p]}), not up in 2026", f"held-{p}", "")
            for s, g, p in zip(nu["state"], nu["governor"], nu["party"])]  # fmt: skip
    for r in f.itertuples():
        side = {"D": r.p_dem_win, "R": r.p_rep_win, "O": r.p_other_win}[r.favourite]
        rows.append(
            (
                f"{r.race}: {r.rating}, {PARTY_NAME[r.favourite]} {pct(side)}",
                "race",
                r.rating,
            )
        )
    held_d = [x for x in rows if x[1] == "held-D"]
    held_r = [x for x in rows if x[1] == "held-R"]
    races = [x for x in rows if x[1] == "race"]
    order = {lab: i for i, lab in enumerate(RATING_ORDER)}
    races.sort(key=lambda x: order.get(x[2], 3))
    cells = []
    for title, kind, rating in held_d + races + held_r:
        colour = RATING_COLOUR.get(rating, MUTED) if kind == "race" else ""
        cls = "seat seat-held seat-held-" + kind[-1] if kind != "race" else "seat"
        style_attr = f' style="--c:{colour}"' if colour else ""
        cells.append(
            f'<span class="{cls}"{style_attr} title="{html.escape(title)}"></span>'
        )
    cells.insert(
        25, '<span class="seat-majority" aria-hidden="true"></span>'
    )  # 26 seats is a majority
    return HTML(
        '<div class="seat-bar" role="img" aria-label="The 50 governorships: 14 not up '
        'this year and 36 races coloured by rating, Democratic on the left">'
        + "".join(cells)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# The ratings strip
# ---------------------------------------------------------------------------
def _dot_x(r) -> float:
    """Horizontal position (0-100) of a race's dot: the centre line is a coin flip,
    the edges are certainty; Democrats to the left."""
    p = {"D": r.p_dem_win, "R": r.p_rep_win, "O": r.p_other_win}[r.favourite]
    shift = (p - 0.5) / 0.5 * 46  # up to 46% of the width from the centre
    return (
        50 - shift if r.favourite == "D" else 50 + shift if r.favourite == "R" else 50
    )


def _sources_svg(r, c_margin: float) -> str:
    """Small chart: what each source says about the margin, on one axis (R+40 to D+40)."""
    rows = [
        ("Past results", r.fund_margin),
        ("AI survey, Method A", r.silicon_margin_A),
        ("AI survey, Method B", r.silicon_margin_B),
        ("AI surveys, corrected", r.silicon_margin_corrected),
        ("Model as forecaster (C)", c_margin),
        ("Poll average", r.poll_margin),
        ("Forecast", r.margin_mean),
    ]
    # Democrats on the left, as in the ratings strip; values in a column on the right.
    w, left, right, h_row, lim = 560, 170, 70, 20, 40

    def scale(m: float) -> float:
        return left + (w - left - right) * (0.5 - np.clip(m, -lim, lim) / (2 * lim))

    aria = f"What each source says about the {html.escape(r.race)} margin"
    parts = [
        f'<svg class="sources" viewBox="0 0 {w} {h_row * len(rows) + 22}" role="img" aria-label="{aria}">'
    ]
    for k in (-40, -20, 0, 20, 40):
        x = scale(k)
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="4" y2="{h_row * len(rows)}" class="grid{" zero" if k == 0 else ""}"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{h_row * len(rows) + 16}" class="tick">{lean(k) if k else "Even"}</text>'
        )
    for i, (label, m) in enumerate(rows):
        y = 12 + i * h_row
        parts.append(f'<text x="0" y="{y + 4}" class="lab">{label}</text>')
        if pd.isna(m):
            parts.append(
                f'<text x="{scale(0):.1f}" y="{y + 4}" class="none">no polls</text>'
            )
            continue
        if label == "Forecast":
            lo, hi = scale(r.margin_hi80), scale(r.margin_lo80)
            parts.append(
                f'<line x1="{lo:.1f}" x2="{hi:.1f}" y1="{y}" y2="{y}" class="range"/>'
            )
        cls = "fc" if label == "Forecast" else "pt"
        parts.append(f'<circle cx="{scale(m):.1f}" cy="{y}" r="{5 if cls == "fc" else 4}" class="{cls} {"d" if m >= 0 else "r"}">'
                     f"<title>{label}: {lean(m, 1)}</title></circle>")  # fmt: skip
        parts.append(f'<text x="{w}" y="{y + 4}" class="val">{lean(m, 1)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _detail(r, c_margin: float) -> str:
    cands = [("Democrat", r.dem_candidate, r.dem_vote_mean, r.p_dem_win),
             ("Republican", r.rep_candidate, r.rep_vote_mean, r.p_rep_win)]  # fmt: skip
    if isinstance(r.other_candidates, str):
        cands.append(
            (
                "Independent or third party",
                r.other_candidates,
                r.other_vote_mean,
                r.p_other_win,
            )
        )
    rows = "".join(
        f"<tr><th scope='row'>{html.escape(name)}<span>{who}</span></th>"
        f"<td>{share:.1f}%</td><td>{pct(p)}</td></tr>"
        for who, name, share, p in cands
    )
    extra = []
    if not pd.isna(r.p_runoff):
        extra.append(
            f"Chance no one passes 50% and Georgia holds a runoff: {pct(r.p_runoff)}."
        )
    polls = f"{r.n_polls} polls since 1 August" if r.n_polls else "No published polls"
    weights = (f"The blend weights polls {r.weight_polls:.0%}, AI surveys {r.weight_silicon:.0%} "
               f"and past results {r.weight_fundamentals:.0%}.")  # fmt: skip
    return (
        '<div class="race-detail">'
        '<table class="cands" data-quarto-disable-processing="true"><thead><tr><th>Candidate</th><th>Expected vote</th>'
        f"<th>Chance of winning</th></tr></thead><tbody>{rows}</tbody></table>"
        f'<p class="race-note">Projected margin {lean(r.margin_mean, 1)}, 80% range '
        f"{lean(r.margin_lo80, 1)} to {lean(r.margin_hi80, 1)}. {polls}. {weights} "
        + " ".join(extra)
        + "</p>"
        + _sources_svg(r, c_margin)
        + "</div>"
    )


def ratings_strip() -> HTML:
    """One row per race, closest first: a dot left of centre for a Democratic
    favourite, right for a Republican, further out the safer the race. Each row opens
    to show the candidates and what every source says."""
    f, cm = forecast(), method_c_margin()
    counts = f["rating"].value_counts()
    pills = [
        f'<button type="button" class="pill" data-filter="all" aria-pressed="true">All <span>{len(f)}</span></button>'
    ]
    for lab in RATING_ORDER:
        if counts.get(lab, 0):
            pills.append(f'<button type="button" class="pill" data-filter="{lab}" aria-pressed="false" '
                         f'style="--c:{RATING_TEXT[lab]}">{lab} <span>{counts[lab]}</span></button>')  # fmt: skip
    rows = []
    for r in f.itertuples():
        p = {"D": r.p_dem_win, "R": r.p_rep_win, "O": r.p_other_win}[r.favourite]
        colour = RATING_COLOUR.get(r.rating, RATING_COLOUR["Toss-up"])
        words = RATING_TEXT.get(r.rating, RATING_TEXT["Toss-up"])
        side = "left" if r.favourite == "D" else "right"
        side += " inside" if p >= 0.85 else ""  # far from the centre: label towards it
        sr = f"{PARTY_NAME[r.favourite]} candidate favoured, {pct(p)} chance; rated {r.rating}."
        rows.append(
            f'<li class="race" data-rating="{r.rating}" data-name="{r.race.lower()}">'
            f"<details><summary>"
            f'<span class="race-name">{r.race}</span>'
            f'<span class="race-track" aria-hidden="true"><span class="race-dot {side}" '
            f'style="--x:{_dot_x(r):.1f}%;--c:{colour};--t:{words}"><b>{pct(p)}</b></span></span>'
            f'<span class="race-rating" style="--c:{words}">{r.rating}</span>'
            f'<span class="visually-hidden">{sr}</span>'
            f"</summary>{_detail(r, cm.get(r.race, np.nan))}</details></li>"
        )
    return HTML(
        '<div class="ratings" id="ratings-strip">'
        '<div class="ratings-controls">'
        '<label class="visually-hidden" for="race-search">Find a state</label>'
        '<input id="race-search" type="search" placeholder="Find a state" autocomplete="off">'
        f'<div class="pills" role="group" aria-label="Show races by rating">{"".join(pills)}</div>'
        "</div>"
        '<div class="ratings-head" aria-hidden="true"><span>State</span>'
        "<span class='axis'><span>Democratic favoured</span><span>Republican favoured</span></span>"
        "<span>Rating</span></div>"
        f'<ol class="ratings-list">{"".join(rows)}</ol>'
        '<p class="ratings-empty" hidden>No state matches. Clear the search or choose All.</p>'
        "</div>"
        "<script>" + _STRIP_JS + "</script>"
    )


_STRIP_JS = """
(() => {
  const root = document.getElementById('ratings-strip');
  if (!root) return;
  const rows = [...root.querySelectorAll('.race')];
  const pills = [...root.querySelectorAll('.pill')];
  const search = root.querySelector('#race-search');
  const empty = root.querySelector('.ratings-empty');
  let rating = 'all';
  const apply = () => {
    const q = search.value.trim().toLowerCase();
    let shown = 0;
    rows.forEach(r => {
      const ok = (rating === 'all' || r.dataset.rating === rating) && r.dataset.name.includes(q);
      r.hidden = !ok;
      shown += ok;
    });
    empty.hidden = shown > 0;
  };
  pills.forEach(p => p.addEventListener('click', () => {
    rating = p.dataset.filter;
    pills.forEach(o => o.setAttribute('aria-pressed', String(o === p)));
    apply();
  }));
  search.addEventListener('input', apply);
})();
"""


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def tbl_forecast() -> GT:
    f = forecast().copy()
    df = pd.DataFrame({
        "State": f["race"],
        "Democrat": f["dem_candidate"],
        "Republican": f["rep_candidate"],
        "Rating": f["rating"],
        "Democratic win": f["p_dem_win"].map(pct),
        "Republican win": f["p_rep_win"].map(pct),
        "Margin": f["margin_mean"].map(lambda m: lean(m, 1)),
        "80% range": [f"{lean(a, 1)} to {lean(b, 1)}" for a, b in zip(f["margin_lo80"], f["margin_hi80"])],
        "Polls": f["n_polls"],
    })  # fmt: skip
    return table(df).cols_align(
        "right",
        columns=["Democratic win", "Republican win", "Margin", "80% range", "Polls"],
    )


def tbl_sources() -> GT:
    """What each source says, per race: past results, the two AI surveys, Method C,
    the polls and the final forecast (margins, Democrat minus Republican)."""
    f = forecast().merge(
        method_c_margin(), left_on="race", right_index=True, how="left"
    )
    cols = {"fund_margin": "Past results", "silicon_margin_A": "Method A", "silicon_margin_B": "Method B",
            "method_c": "Method C", "poll_margin": "Polls", "margin_mean": "Forecast"}  # fmt: skip
    df = f[["race", *cols]].rename(columns={"race": "State", **cols})
    for c in cols.values():
        df[c] = df[c].map(lambda m: lean(m, 1))
    return (
        table(df)
        .cols_align("right", columns=list(cols.values()))
        .tab_spanner("AI surveys (uncalibrated)", columns=["Method A", "Method B"])
    )


def tbl_models() -> GT:
    cfg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
    log = pd.read_csv(ANA / "run_log.csv")
    log = log[~log["simulated_inputs"].astype(bool)]
    rate = (
        log.assign(ok=log["calls"] * log["parse_rate"])
        .groupby("model_slug")[["ok", "calls"]]
        .sum()
    )
    rows = []
    for m in cfg["models"]:
        temp = m.get("temperature", cfg["temperature"])
        rows.append({
            "Model": MODEL_NAME.get(m["slug"], m["slug"]),
            "OpenRouter identifier": m["slug"],
            "Host": ", ".join(m["provider_order"]),
            "Reasoning": m.get("reasoning_effort") or "not offered",
            "Temperature": "provider default" if temp is None else f"{temp:g}",
            "Answers": int(rate.loc[m["slug"], "calls"]),
            "Usable": rate.loc[m["slug"], "ok"] / rate.loc[m["slug"], "calls"],
        })  # fmt: skip
    return (
        table(pd.DataFrame(rows))
        .fmt_integer("Answers")
        .fmt_percent("Usable", decimals=1)
        .cols_align("right", columns=["Answers", "Usable"])
    )


def tbl_response_distribution() -> GT:
    a = responses("A")
    cands = pd.read_parquet(ANA / "candidates.parquet")
    party = cands.set_index(["race", "candidate"])["party_code"].to_dict()
    label = {"D": "Democrat", "R": "Republican", "O": "Independent or third party"}
    a["answer"] = [
        "Unusable" if not ok else label.get(party.get((r, v)), v)
        for ok, r, v in zip(a["parsed"], a["race"], a["governor_vote"])
    ]
    a["Model"] = a["model_slug"].map(MODEL_NAME)
    cols = [
        "Democrat",
        "Republican",
        "Independent or third party",
        "Other",
        "Don't know",
        "Unusable",
    ]
    t = pd.crosstab(a["Model"], a["answer"]).reindex(columns=cols, fill_value=0)
    t.loc["All models"] = t.sum()
    t["Total"] = t.sum(axis=1)
    return table(t.reset_index(), rowname_col="Model").fmt_integer(
        columns=cols + ["Total"]
    )


def tbl_ticket_splitting() -> GT:
    t = pd.read_csv(ANA / "ticket_splitting.csv")
    df = pd.DataFrame(
        {
            "Model": t["model_slug"].map(MODEL_NAME),
            "Answers": t["answers"],
            "Split ticket": t["split_rate"],
        }
    )
    return table(df).fmt_integer("Answers").fmt_percent("Split ticket", decimals=1)


def tbl_validation() -> GT:
    v = pd.read_csv(FC / "validation_summary.csv")
    v = v[v["run_date"] == v["run_date"].min()]
    wide = v.pivot_table(index="check", columns="model_slug", values="changed_pct")
    wide.columns = [MODEL_NAME[c] for c in wide.columns]
    retest = v.groupby("model_slug")["retest_changed_pct"].first()
    base = pd.DataFrame(
        [retest.rename(index=MODEL_NAME)], index=["Asked again, nothing changed"]
    )
    order = ["Asked again, nothing changed"] + [CHECK_NAME[c] for c in CHECK_NAME]
    wide.index = wide.index.map(CHECK_NAME)
    df = (
        pd.concat([base, wide]).reindex(order).reset_index(names="Change to the prompt")
    )
    models = [c for c in df.columns if c != "Change to the prompt"]
    return (
        table(df, rowname_col="Change to the prompt")
        .fmt_number(columns=models, decimals=1, pattern="{x}%")
        .tab_spanner("Votes that changed", columns=models)
    )


def tbl_stability() -> GT:
    s = pd.read_csv(FC / "stability_summary.csv")
    s["Model"] = s["model_slug"].map(MODEL_NAME)
    bc = s[s["method"].isin(["B", "C"])]
    df = pd.DataFrame({
        "Method": bc["method"].map({"B": "B: group split, margin", "C": "C: forecaster, Democratic share"}),
        "Model": bc["Model"],
        "Days": bc["n_days"],
        "Spread within a day": bc["within_day_sd"],
        "Spread between days": bc["between_day_sd"],
    })  # fmt: skip
    return (
        table(df, groupname_col="Method", rowname_col="Model")
        .fmt_number(columns=["Spread within a day", "Spread between days"], decimals=1)
        .sub_missing(missing_text="one day so far")
    )


def tbl_costs() -> GT:
    c = pd.read_csv(FC / "cost_summary.csv")
    g = c.groupby(["method", "model_slug"], as_index=False).agg(
        calls=("calls", "sum"), prompt=("prompt_tokens", "sum"), completion=("completion_tokens", "sum"),
        cost=("cost_usd", "sum"), seconds=("mean_seconds_per_call", "mean"),
    )  # fmt: skip
    df = pd.DataFrame({
        "Method": g["method"].map(METHOD_NAME),
        "Model": g["model_slug"].map(MODEL_NAME),
        "Calls": g["calls"],
        "Input tokens": g["prompt"],
        "Output tokens": g["completion"],
        "Seconds per call": g["seconds"],
        "Cost (US$)": g["cost"],
    })  # fmt: skip
    total = {"Method": "All methods and models", "Model": "Total", "Calls": df["Calls"].sum(),
             "Input tokens": df["Input tokens"].sum(), "Output tokens": df["Output tokens"].sum(),
             "Seconds per call": np.nan, "Cost (US$)": df["Cost (US$)"].sum()}  # fmt: skip
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    return (
        table(df, groupname_col="Method", rowname_col="Model")
        .fmt_integer(columns=["Calls", "Input tokens", "Output tokens"])
        .fmt_number(columns="Seconds per call", decimals=1)
        .fmt(dollars, columns="Cost (US$)")
        .sub_missing(missing_text="")
    )


def tbl_cost_by_day() -> GT:
    c = pd.read_csv(FC / "cost_summary.csv")
    d = c.groupby("run_date", as_index=False).agg(
        calls=("calls", "sum"), cost=("cost_usd", "sum")
    )
    df = pd.DataFrame({
        "Day": [day(t) for t in pd.to_datetime(d["run_date"])],
        "Run": np.where(d.index == 0, "Full run on a laptop", "Daily GitHub Actions run"),
        "Calls": d["calls"],
        "Cost (US$)": d["cost"],
    })  # fmt: skip
    return table(df).fmt_integer("Calls").fmt(dollars, columns="Cost (US$)")


def tbl_frame_states() -> GT:
    s = pd.read_csv(ANA / "frame_states.csv")
    df = pd.DataFrame({
        "State": s["race"],
        "CES respondents": s["n_respondents"],
        "Effective sample": s["n_effective"],
        "Democrats": s["share_D"] / 100,
        "Independents": s["share_I"] / 100,
        "Republicans": s["share_R"] / 100,
        "Republican, 2024 survey": s["share_R_2024"] / 100,
        "Republican, 2025 survey": s["share_R_2025"] / 100,
    }).sort_values("Effective sample")  # fmt: skip
    pcts = [
        "Democrats",
        "Independents",
        "Republicans",
        "Republican, 2024 survey",
        "Republican, 2025 survey",
    ]
    return (
        table(df)
        .fmt_integer(["CES respondents", "Effective sample"])
        .fmt_percent(pcts, decimals=0)
        .tab_spanner("Weighted share of registered voters", columns=pcts[:3])
        .tab_spanner("By survey year", columns=pcts[3:])
    )


def tbl_composition() -> GT:
    """The 3,600 personas against the weighted frame, by party, age and degree."""
    p = pd.read_parquet(ANA / "personas.parquet")
    fr = pd.read_parquet(ANA / "frame.parquet")
    states = fr["race"].nunique()
    rows = []
    for var, title, labels in [
        ("party_group", "Party", {"D": "Democrats (incl. leaners)", "I": "Independents", "R": "Republicans (incl. leaners)"}),
        ("age_group", "Age", {a: a for a in ["18-29", "30-44", "45-64", "65+"]}),
        ("degree", "Education", {True: "Four-year degree", False: "No four-year degree"}),
    ]:  # fmt: skip
        pers = p[var].value_counts(normalize=True)
        frame = fr.groupby(var)["share"].sum() / states  # average over states
        for k, lab in labels.items():
            rows.append(
                {
                    "group": title,
                    "Category": lab,
                    "Personas": pers.get(k, 0),
                    "Frame": frame.get(k, 0),
                }
            )
    df = pd.DataFrame(rows)
    df["Difference"] = (df["Personas"] - df["Frame"]) * 100
    return (
        table(df, groupname_col="group", rowname_col="Category")
        .fmt_percent(["Personas", "Frame"], decimals=0)
        .fmt_number("Difference", decimals=0, force_sign=True, pattern="{x} pts")
        .cols_label(Frame="Weighted frame")
    )


def tbl_polls() -> GT:
    po = pd.read_csv(RAW / "governor_polls_2026.csv", parse_dates=["end_date"])
    po = po[po["end_date"] >= "2026-08-01"]
    f = forecast().set_index("race")
    g = po.groupby("race").agg(
        n=("pollster", "size"),
        latest=("end_date", "max"),
        lv=("population", lambda s: (s == "LV").mean()),
    )
    df = pd.DataFrame({
        "State": g.index,
        "Polls": g["n"].values,
        "Likely voters": g["lv"].values,
        "Latest fieldwork": [day(t, short=True) for t in g["latest"]],
        "Poll average": [lean(f.loc[s, "poll_margin"], 1) for s in g.index],
    }).sort_values("Polls", ascending=False)  # fmt: skip
    return (
        table(df)
        .fmt_percent("Likely voters", decimals=0)
        .cols_align("right", columns=["Poll average"])
    )


def tbl_fundamentals() -> GT:
    c = pd.read_csv(FC / "fundamentals_coefficients.csv")
    names = {"intercept": "Intercept", "pvi": "State lean (Cook PVI, per point)",
             "incumbent_running": "Incumbent running (+1 D, -1 R)", "president_democrat": "President's party (+1 D, -1 R)"}  # fmt: skip
    df = pd.DataFrame(
        {
            "Term": c["term"].map(names),
            "Estimate": c["estimate"],
            "Standard error": c["std_error"],
        }
    )
    return table(df).fmt_number(["Estimate", "Standard error"], decimals=2)


def tbl_config() -> GT:
    info = run_info()
    cfg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
    rows = [
        ("Personas per state", "100, drawn in proportion to CES weight"),
        ("Validation personas per state", "10"),
        ("Groups per state (Method B)", "24: party x age x degree"),
        ("Repeats in the full run", str(cfg["repeats"])),
        ("Temperature", f"{cfg['temperature']:g} (provider default for GPT-6 Luna)"),
        ("Simulated elections", f"{info['n_simulations']:,}"),
        ("Random seed (forecast)", str(info["seed"])),
        ("Shared national swing, SD", f"{info['national_sd']:g} points"),
        ("Extra polling error, SD", f"{info['poll_floor_sd']:g} points"),
        (
            "Polls used",
            f"{info['polls_used']:,}, fieldwork ending on or after {info['poll_start']}",
        ),
    ]
    return table(pd.DataFrame(rows, columns=["Setting", "Value"]))


# ---------------------------------------------------------------------------
# An example of exactly what a model was sent, and what it answered
# ---------------------------------------------------------------------------
def example_prompt(state: str = "Michigan", model: str = "openai/gpt-6-luna") -> HTML:
    """Rebuild one Method A prompt with the pipeline's own code (scripts/common.py),
    in the candidate order that call used, beside the model's reply."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from common import candidate_lines, fill, quoted_names, read_prompt

    p = pd.read_parquet(ANA / "personas.parquet")
    persona = p[(p["race"] == state) & p["validation"]].iloc[0]
    a = responses("A")
    row = a[
        (a["persona_id"] == persona["persona_id"])
        & (a["model_slug"] == model)
        & (a["repeat"] == 1)
    ].iloc[0]
    cands = pd.read_parquet(ANA / "candidates.parquet")
    cands = (
        cands[cands["race"] == state]
        .set_index("candidate")
        .loc[json.loads(row["candidate_order"])]
        .reset_index()
    )
    user = fill(read_prompt("method_a_vote.txt"), persona=persona["persona_text"], state=state,
                candidate_list=candidate_lines(cands), candidate_names=quoted_names(cands))  # fmt: skip
    esc = html.escape
    return HTML(
        f'<p class="prompt-label">System message</p><pre class="prompt">{esc(read_prompt("system.txt"))}</pre>'
        f'<p class="prompt-label">Question</p><pre class="prompt">{esc(user)}</pre>'
        f'<p class="prompt-label">{esc(MODEL_NAME[model])} replied</p><pre class="prompt">{esc(row["raw_text"].strip())}</pre>'
    )
