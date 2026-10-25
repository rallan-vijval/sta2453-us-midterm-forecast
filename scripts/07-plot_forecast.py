"""
Purpose : Draw the sketched charts from the forecast output. Every chart treats
          both parties the same way (margins read "D+5" or "R+12"):
          1. forecast_map.png      tile map: the favoured party in each race and its chance of winning
          2. race_intervals.png    winning margin per race, 80% and 95% intervals
          3. silicon_vs_polls.png  AI-survey margin against the poll-average margin, where polls exist
          4. seat_count.png        governorships each party holds, across all simulated elections
          5. sources.png           past results, AI surveys and polls for each race, beside the forecast
Author  : Allwin Steve Paul Justin Anantharaj (Member 3)
Date    : 2026-10-24
Inputs  : data/03-forecast/forecast.parquet, simulations.parquet   (05-forecast.py)
          with --llm-test: data/03-forecast-llm-test/forecast.parquet
Outputs : other/charts/*.png
          with --llm-test: data/03-forecast-llm-test/charts/*.png (practice, not committed)
Notes   : Win-probability bands follow the map sketch: Solid > 90%, Likely 75-90%,
          Lean 60-75%, Toss-up 40-60%. Each state is one equal-sized tile, so small
          states are as visible as large ones. Grey tiles have no governor race in 2026.
Run     : uv run python scripts/07-plot_forecast.py [--llm-test]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Same palette as the website (website/styles.scss).
PAPER, INK, MUTED, RULE = "#F7F4EE", "#1E1E1C", "#6B665E", "#D9D3C7"
DEM, REP, ACCENT = "#2C5F8A", "#B04A2F", "#B8893B"
BANDS = [  # (lowest Democratic win probability, label, colour)
    (0.90, "Solid D", "#1F4466"), (0.75, "Likely D", "#3F74A3"), (0.60, "Lean D", "#9CBDDB"),
    (0.40, "Toss-up", "#D8B24E"), (0.25, "Lean R", "#E7A893"), (0.10, "Likely R", "#C2593F"),
    (0.00, "Solid R", "#842C1B"),
]
NO_RACE = "#CFCAC0"
OTHER = "#7B6D8D"  # an independent or minor-party candidate is favoured

ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "Guam": "GU", "Northern Mariana Islands": "MP", "U.S. Virgin Islands": "VI",
}
# Tile positions (column, row), row 0 at the top.
TILES = {
    "AK": (0, 0), "ME": (11, 0), "VT": (10, 1), "NH": (11, 1),
    "WA": (1, 2), "ID": (2, 2), "MT": (3, 2), "ND": (4, 2), "MN": (5, 2), "IL": (6, 2), "WI": (7, 2),
    "MI": (8, 2), "NY": (9, 2), "RI": (10, 2), "MA": (11, 2),
    "OR": (1, 3), "NV": (2, 3), "WY": (3, 3), "SD": (4, 3), "IA": (5, 3), "IN": (6, 3), "OH": (7, 3),
    "PA": (8, 3), "NJ": (9, 3), "CT": (10, 3),
    "CA": (1, 4), "UT": (2, 4), "CO": (3, 4), "NE": (4, 4), "MO": (5, 4), "KY": (6, 4), "WV": (7, 4),
    "VA": (8, 4), "MD": (9, 4), "DE": (10, 4),
    "AZ": (2, 5), "NM": (3, 5), "KS": (4, 5), "AR": (5, 5), "TN": (6, 5), "NC": (7, 5), "SC": (8, 5),
    "OK": (4, 6), "LA": (5, 6), "MS": (6, 6), "AL": (7, 6), "GA": (8, 6),
    "HI": (0, 7), "TX": (4, 7), "FL": (9, 7),
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["IBM Plex Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10.5, "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "semibold", "axes.titlesize": 12, "axes.titlelocation": "left",
})


def band(p: float) -> tuple[str, str]:
    for lo, label, colour in BANDS:
        if p >= lo:
            return label, colour
    return BANDS[-1][1], BANDS[-1][2]


def lean(m) -> str:
    """Margin in points (Democrat minus Republican) as text: D+5, R+12, Even."""
    m = float(m)
    return "Even" if abs(m) < 0.5 else (f"D+{m:.0f}" if m > 0 else f"R+{-m:.0f}")


LEAN_TICKS = FuncFormatter(lambda m, _: lean(m))


def tile(p_d: float, p_r: float, p_o: float) -> tuple[str, str, str]:
    """(band label, colour, tile text) for the favoured party, which can be an independent."""
    party, p = max([("D", p_d), ("R", p_r), ("I", p_o)], key=lambda t: t[1])
    if party == "I" and p >= 0.6:
        return "Independent favoured", OTHER, f"I {p:.0%}"
    if p < 0.6:
        return "Toss-up", BANDS[3][2], f"{party} {p:.0%}"
    label, colour = band(p_d if party == "D" else 1 - p_r)
    return label, colour, f"{party} {p:.0%}"


def forecast_map(f: pd.DataFrame, out: Path, note: str):
    p_r = f["p_rep_win"] if "p_rep_win" in f else 1 - f["p_dem_win"]
    p_o = f["p_other_win"] if "p_other_win" in f else 0 * f["p_dem_win"]
    probs = {ABBR[race]: (d, r, o) for race, d, r, o in zip(f["race"], f["p_dem_win"], p_r, p_o)}
    fig, ax = plt.subplots(figsize=(8, 6.4))
    for code, (c, r) in TILES.items():
        if code in probs:
            label, colour, fav = tile(*probs[code])
        else:
            label, colour = "No race", NO_RACE
        ax.add_patch(Rectangle((c, -r), 0.92, 0.92, facecolor=colour, edgecolor=PAPER))
        dark = label in {"Solid D", "Likely D", "Likely R", "Solid R", "Independent favoured"}
        ax.text(c + 0.46, -r + 0.56, code, ha="center", va="center", fontsize=10, weight="semibold",
                color="white" if dark else INK)
        if code in probs:
            ax.text(c + 0.46, -r + 0.24, fav, ha="center", va="center", fontsize=6.8,
                    color="white" if dark else INK)
    ax.set_xlim(-0.2, 12.1)
    ax.set_ylim(-7.3, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [Patch(facecolor=c, label=lab) for _, lab, c in BANDS] + [Patch(facecolor=NO_RACE, label="No race")]
    if any(tile(*v)[0] == "Independent favoured" for v in probs.values()):
        handles.append(Patch(facecolor=OTHER, label="Independent favoured"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 0), ncol=4, frameon=False, fontsize=9,
              title="Each tile: the favoured candidate's party (D, R or I = independent) and its chance of winning.\n"
                    "Solid > 90%, Likely 75-90%, Lean 60-75%, Toss-up: no one above 60%. "
                    "Plurality wins; Georgia needs a majority or goes to a runoff.",
              title_fontsize=8, alignment="left")
    ax.set_title("Which party is favoured in each governor's race" + note)
    fig.savefig(out / "forecast_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def race_intervals(f: pd.DataFrame, out: Path, note: str):
    f = f.sort_values("margin_mean").reset_index(drop=True)
    y = np.arange(len(f))
    m = lambda col: 2 * f[col] - 100  # two-party share -> margin
    colour = np.where(f["margin_mean"] >= 0, DEM, REP)
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(f) + 1.6))
    ax.hlines(y, m("dem_share_lo95"), m("dem_share_hi95"), color=colour, linewidth=1, alpha=0.6)
    ax.hlines(y, m("dem_share_lo80"), m("dem_share_hi80"), color=colour, linewidth=5, alpha=0.85)
    ax.scatter(m("dem_share_mean"), y, s=18, color=PAPER, edgecolor=INK, linewidth=0.8, zorder=3)
    ax.axvline(0, color=INK, linestyle="--", linewidth=0.8)
    lim = np.ceil(max(abs(m("dem_share_lo95")).max(), abs(m("dem_share_hi95")).max()) / 10) * 10
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(LEAN_TICKS)
    ax.set_yticks(y, f["race"].map(ABBR))
    ax.set_xlabel("Forecast winning margin (points of the two-party vote)")
    ax.text(0.01, 1.005, "Republican ahead", transform=ax.transAxes, color=REP, fontsize=9, va="bottom")
    ax.text(0.99, 1.005, "Democrat ahead", transform=ax.transAxes, color=DEM, fontsize=9, va="bottom", ha="right")
    ax.set_title("Forecast margin in every race, with 80% and 95% intervals" + note, pad=18)
    ax.grid(axis="x", color=RULE, linewidth=0.6)
    ax.text(0.99, 0.01, "thick bar: 80% of simulations\nthin line: 95%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "race_intervals.png", dpi=200)
    plt.close(fig)


def silicon_vs_polls(f: pd.DataFrame, out: Path, note: str):
    p = f.dropna(subset=["poll_margin", "sil_margin"])
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    if p.empty:
        ax.text(0.5, 0.5, "No race has both polls and AI-survey estimates yet", ha="center", transform=ax.transAxes)
    else:
        x, y = p["poll_margin"].to_numpy(float), p["sil_margin"].to_numpy(float)
        lim = np.ceil(max(abs(x).max(), abs(y).max()) / 10) * 10 + 5
        ax.plot([-lim, lim], [-lim, lim], color=INK, linewidth=0.8)
        ax.axhline(0, color=RULE, linewidth=0.8)
        ax.axvline(0, color=RULE, linewidth=0.8)
        ax.scatter(x, y, s=36, color=ACCENT, zorder=3)
        for xi, yi, code in zip(x, y, p["race"].map(ABBR)):
            ax.annotate(code, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8.5)
        if CAL and CAL.get("calibration_slope"):
            a_, b_ = CAL["calibration_intercept"], CAL["calibration_slope"]
            xs = np.array([-lim, lim])
            ax.plot(xs, (xs - a_) / b_, color=ACCENT, linestyle="--", linewidth=1)
            ax.text(0.97, 0.03, f"dashed: fitted calibration, poll = {a_:+.1f} + {b_:.2f} x AI survey",
                    transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
        gap = float(np.mean(y - x))
        side = "more Democratic" if gap > 0 else "more Republican"
        ax.text(0.03, 0.97, f"On average the AI surveys are {abs(gap):.1f} points {side} than the polls.\n"
                "Above the line: AI more Democratic. Below: AI more Republican.",
                transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(LEAN_TICKS)
        ax.yaxis.set_major_formatter(LEAN_TICKS)
    ax.set_xlabel("Poll average: winning margin")
    ax.set_ylabel("AI surveys (Methods A and B): winning margin")
    ax.set_title("AI surveys against the polls" + note)
    ax.grid(color=RULE, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out / "silicon_vs_polls.png", dpi=200)
    plt.close(fig)


def sources(f: pd.DataFrame, out: Path, note: str):
    """What each source says about each race, next to the final forecast."""
    f = f.sort_values("margin_mean").reset_index(drop=True)
    y = np.arange(len(f))
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(f) + 1.8))
    ax.hlines(y, f[["fund_margin", "silicon_margin_corrected", "poll_margin", "margin_mean"]].min(axis=1),
              f[["fund_margin", "silicon_margin_corrected", "poll_margin", "margin_mean"]].max(axis=1),
              color=RULE, linewidth=1)
    ax.scatter(f["fund_margin"], y, s=22, marker="s", color=MUTED, label="Past results (fundamentals)", zorder=3)
    ax.scatter(f["silicon_margin_corrected"], y, s=26, marker="^", color=ACCENT,
               label="AI surveys (calibrated)", zorder=3)
    ax.scatter(f["poll_margin"], y, s=22, marker="D", facecolor="none", edgecolor=INK, label="Poll average", zorder=3)
    ax.scatter(f["margin_mean"], y, s=40, color=np.where(f["margin_mean"] >= 0, DEM, REP), label="Forecast", zorder=4)
    ax.axvline(0, color=INK, linestyle="--", linewidth=0.8)
    lim = np.ceil(np.nanmax(np.abs(f[["fund_margin", "silicon_margin_corrected", "poll_margin"]].to_numpy())) / 10) * 10
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(LEAN_TICKS)
    ax.set_yticks(y, f["race"].map(ABBR))
    ax.set_xlabel("Winning margin (points of the two-party vote)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("What each source says, and the forecast that combines them" + note)
    ax.grid(axis="x", color=RULE, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out / "sources.png", dpi=200)
    plt.close(fig)


def seat_count(sims: pd.DataFrame, out: Path, note: str):
    d, r = sims["dem_governors"].astype(int), sims["rep_governors"].astype(int)
    counts = d.value_counts().sort_index()
    # Blue: Democratic majority. Red: Republican majority. Gold: neither party has 26 (independents hold the balance).
    colour = [DEM if k >= 26 else REP if (r[d == k] >= 26).all() else ACCENT for k in counts.index]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.bar(counts.index, counts / len(sims), color=colour, width=0.8)
    ax.axvline(25.5, color=INK, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Democratic governors after the election (of 50, including the 14 seats not on the ballot)")
    ax.set_ylabel("Share of simulations")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.text(0.01, 0.97, f"Republicans hold a majority (26+): {(r >= 26).mean():.0%}", transform=ax.transAxes,
            va="top", color=REP, fontsize=9, weight="semibold")
    ax.text(0.99, 0.97, f"Democrats hold a majority (26+): {(d >= 26).mean():.0%}", transform=ax.transAxes,
            va="top", ha="right", color=DEM, fontsize=9, weight="semibold")
    ax.set_title("Governorships held by each party, across all simulations" + note, pad=10)
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out / "seat_count.png", dpi=200)
    plt.close(fig)


CAL: dict = {}


def main(llm_test: bool):
    src = ROOT / "data" / ("03-forecast-llm-test" if llm_test else "03-forecast")
    info = src / "run_info.json"
    if info.exists():
        CAL.update(json.loads(info.read_text()))
    out = src / "charts" if llm_test else ROOT / "other" / "charts"
    out.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(src / "forecast.parquet")
    sims = pd.read_parquet(src / "simulations.parquet")
    simulated = bool(f.get("is_simulated", pd.Series([False])).any())
    note = "\nMade from simulated data: a test of the pipeline, not a forecast" if simulated else ""
    forecast_map(f, out, note)
    race_intervals(f, out, note)
    silicon_vs_polls(f, out, note)
    seat_count(sims, out, note)
    if {"fund_margin", "silicon_margin_corrected"} <= set(f.columns):
        sources(f, out, note)
    print(f"Saved {len(list(out.glob('*.png')))} charts to {out}" + (" - made from simulated inputs, do not publish" if simulated else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-test", action="store_true", help="read data/03-forecast-llm-test instead")
    main(parser.parse_args().llm_test)
