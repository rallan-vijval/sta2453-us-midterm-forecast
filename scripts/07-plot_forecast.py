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
Inputs  : also data/01-raw_data/us_states.geojson   state outlines (US Census, public domain;
          downloaded once from PublicaMundi/MappingAPI on GitHub if missing)
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patheffects
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]

# Same palette as the website (website/styles.scss and website/page.py): a dark
# slate page, light text, and party colours that get stronger as a race gets safer.
PAPER, INK, MUTED, RULE = "#1A1F26", "#E6E8EB", "#9AA3AF", "#343C48"
DEM, REP, ACCENT = "#6A9BE0", "#E8776A", "#E3B341"
BANDS = [  # (lowest Democratic win probability, label, colour)
    (0.90, "Solid Democratic", "#2F6BC2"),
    (0.75, "Likely Democratic", "#5C8FD8"),
    (0.60, "Lean Democratic", "#A3C3EE"),
    (0.40, "Toss-up", "#E3B341"),
    (0.25, "Lean Republican", "#F2B1A8"),
    (0.10, "Likely Republican", "#E06E60"),
    (0.00, "Solid Republican", "#B9412F"),
]
NO_RACE = "#2E3540"
OTHER = "#A58BD0"  # an independent or minor-party candidate is favoured

# The website's typefaces ship with the repo (other/fonts, SIL Open Font License):
# Karla for words, Lekton for the numbers on the axes.
for _ttf in sorted((ROOT / "other" / "fonts").glob("*.ttf")):
    font_manager.fontManager.addfont(_ttf)

ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "U.S. Virgin Islands": "VI",
}
FULL_NAME = {code: name for name, code in ABBR.items()}
# Tile positions (column, row), row 0 at the top.
TILES = {
    "AK": (0, 0),
    "ME": (11, 0),
    "VT": (10, 1),
    "NH": (11, 1),
    "WA": (1, 2),
    "ID": (2, 2),
    "MT": (3, 2),
    "ND": (4, 2),
    "MN": (5, 2),
    "IL": (6, 2),
    "WI": (7, 2),
    "MI": (8, 2),
    "NY": (9, 2),
    "RI": (10, 2),
    "MA": (11, 2),
    "OR": (1, 3),
    "NV": (2, 3),
    "WY": (3, 3),
    "SD": (4, 3),
    "IA": (5, 3),
    "IN": (6, 3),
    "OH": (7, 3),
    "PA": (8, 3),
    "NJ": (9, 3),
    "CT": (10, 3),
    "CA": (1, 4),
    "UT": (2, 4),
    "CO": (3, 4),
    "NE": (4, 4),
    "MO": (5, 4),
    "KY": (6, 4),
    "WV": (7, 4),
    "VA": (8, 4),
    "MD": (9, 4),
    "DE": (10, 4),
    "AZ": (2, 5),
    "NM": (3, 5),
    "KS": (4, 5),
    "AR": (5, 5),
    "TN": (6, 5),
    "NC": (7, 5),
    "SC": (8, 5),
    "OK": (4, 6),
    "LA": (5, 6),
    "MS": (6, 6),
    "AL": (7, 6),
    "GA": (8, 6),
    "HI": (0, 7),
    "TX": (4, 7),
    "FL": (9, 7),
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Karla", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.monospace": ["Lekton", "DejaVu Sans Mono"],
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "font.size": 11.5,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
    }
)


def mono_ticks(ax, which: str = "x"):
    """Numbers on an axis in Lekton, as on the website; state names stay in Karla."""
    labels = (ax.get_xticklabels() if "x" in which else []) + (
        ax.get_yticklabels() if "y" in which else []
    )
    for label in labels:
        label.set_fontfamily("monospace")
        label.set_fontsize(12.5)


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
    party, p = max([("Dem", p_d), ("Rep", p_r), ("Ind", p_o)], key=lambda t: t[1])
    text = f"{party} {'>99%' if p >= 0.995 else f'{p:.0%}'}"  # 10,000 draws cannot show certainty
    if party == "Ind" and p >= 0.6:
        return "Independent favoured", OTHER, text
    if p < 0.6:
        return "Toss-up", BANDS[3][2], text
    label, colour = band(p_d if party == "Dem" else 1 - p_r)
    return label, colour, text


def text_on(colour: str) -> str:
    """Dark text on a light fill, light text on a dark one."""
    r, g, b = (int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return PAPER if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else INK


# ---------------------------------------------------------------------------
# Map with real state outlines
# ---------------------------------------------------------------------------
STATES_GEOJSON = ROOT / "data" / "01-raw_data" / "us_states.geojson"
# US Census state outlines (public domain), simplified, as published by PublicaMundi/MappingAPI.
STATES_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
# Small north-eastern states are labelled in a column to the right, with a leader line.
CALLOUTS = [
    "Maine",
    "Vermont",
    "New Hampshire",
    "Massachusetts",
    "Rhode Island",
    "Connecticut",
    "New Jersey",
    "Delaware",
    "Maryland",
]


def load_outlines() -> dict:
    """{state name: list of rings of (lon, lat)}; downloads the outlines once if they are missing."""
    if not STATES_GEOJSON.exists():
        import requests

        STATES_GEOJSON.write_bytes(requests.get(STATES_URL, timeout=60).content)
    shapes = {}
    for feat in json.loads(STATES_GEOJSON.read_text())["features"]:
        geom = feat["geometry"]
        polys = (
            geom["coordinates"]
            if geom["type"] == "MultiPolygon"
            else [geom["coordinates"]]
        )
        shapes[feat["properties"]["name"]] = [
            np.asarray(poly[0], float) for poly in polys
        ]
    return shapes


def albers(lon, lat, lon0=-96.0, lat0=37.5, p1=29.5, p2=45.5):
    """Albers equal-area conic projection (the usual one for US maps); returns x, y on a unit sphere."""
    lon, lat = np.radians(lon), np.radians(lat)
    lon0, lat0, p1, p2 = map(np.radians, (lon0, lat0, p1, p2))
    n = (np.sin(p1) + np.sin(p2)) / 2
    c = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho0 = np.sqrt(c - 2 * n * np.sin(lat0)) / n
    rho = np.sqrt(c - 2 * n * np.sin(lat)) / n
    theta = n * (lon - lon0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def project(name: str, ring: np.ndarray) -> np.ndarray:
    """Lower 48 in the standard projection; Alaska and Hawaii moved to insets below, as on most US maps."""
    lon, lat = ring[:, 0].copy(), ring[:, 1]
    if name == "Alaska":
        lon = np.where(lon > 0, lon - 360, lon)  # Aleutians that cross the date line
        x, y = albers(lon, lat, lon0=-154, lat0=50, p1=55, p2=65)
        return np.column_stack([0.35 * x - 0.30, 0.35 * y - 0.20])
    if name == "Hawaii":
        x, y = albers(lon, lat, lon0=-157, lat0=20, p1=8, p2=18)
        return np.column_stack([x - 0.10, y - 0.21])
    x, y = albers(lon, lat)
    return np.column_stack([x, y])


def label_point(rings: list) -> tuple[float, float]:
    """Area-weighted centre of a state's largest piece."""

    def area_centroid(r):
        x, y = r[:, 0], r[:, 1]
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        a = cross.sum() / 2
        if abs(a) < 1e-12:
            return 0.0, x.mean(), y.mean()
        return (
            abs(a),
            ((x[:-1] + x[1:]) * cross).sum() / (6 * a),
            ((y[:-1] + y[1:]) * cross).sum() / (6 * a),
        )

    _, cx, cy = max(area_centroid(r) for r in rings)
    return cx, cy


# Nudges (in map units) where the centre of a state is not the best place for its name.
LABEL_NUDGE = {
    "Florida": (0.022, -0.03),
    "Louisiana": (-0.006, 0.0),
    "Michigan": (0.0, -0.02),
    "Wisconsin": (0.003, -0.006),
    "Georgia": (0.006, -0.012),
    "North Carolina": (0.0, 0.006),
    "Idaho": (0.0, -0.02),
    "California": (-0.008, 0.0),
    "Kentucky": (0.006, 0.0),
    "West Virginia": (-0.004, -0.002),
    "Virginia": (0.008, -0.003),
    "Minnesota": (-0.008, -0.01),
    "Oklahoma": (0.01, -0.004),
    "Illinois": (-0.004, 0.0),
    "Indiana": (0.003, 0.012),
    "Mississippi": (-0.01, -0.012),
    "Alabama": (0.004, -0.006),
    "South Carolina": (0.008, -0.004),
    "Tennessee": (0.004, 0.0),
}
# States labelled to the left of their shape rather than inside it (Hawaii's islands are too small).
LABEL_BESIDE = {"Hawaii"}


def forecast_map(f: pd.DataFrame, out: Path, note: str):
    p_r = f["p_rep_win"] if "p_rep_win" in f else 1 - f["p_dem_win"]
    p_o = f["p_other_win"] if "p_other_win" in f else 0 * f["p_dem_win"]
    probs = {
        race: (d, r, o) for race, d, r, o in zip(f["race"], f["p_dem_win"], p_r, p_o)
    }
    shapes = {
        name: [project(name, r) for r in rings]
        for name, rings in load_outlines().items()
        if name in ABBR
        and name not in {"Guam", "Northern Mariana Islands", "U.S. Virgin Islands"}
    }

    fig, ax = plt.subplots(figsize=(12, 8.2))
    for name, rings in shapes.items():
        _, colour, fav = (
            tile(*probs[name]) if name in probs else ("No race", NO_RACE, "")
        )
        for ring in rings:
            ax.add_patch(
                Polygon(
                    ring, closed=True, facecolor=colour, edgecolor=PAPER, linewidth=0.8
                )
            )
        # Light text with a dark outline: readable on every fill, and on a dark
        # neighbouring state where a long name runs past the border.
        on_fill = INK if fav else MUTED
        halo = [patheffects.withStroke(linewidth=2.8, foreground=PAPER)]
        if name in CALLOUTS:
            continue
        x, y = label_point(rings)
        dx, dy = LABEL_NUDGE.get(name, (0, 0))
        if name in LABEL_BESIDE:
            left = min(r[:, 0].min() for r in rings)
            ax.text(
                left - 0.01,
                y,
                f"{name}\n{fav}" if fav else name,
                ha="right",
                va="center",
                fontsize=9,
                weight="bold",
                color=INK,
                zorder=5,
            )
            continue
        text = (
            name.replace(" ", "\n", 1)
            if name
            in {
                "North Dakota",
                "South Dakota",
                "North Carolina",
                "South Carolina",
                "New Mexico",
                "West Virginia",
            }
            else name
        )
        ax.text(
            x + dx,
            y + dy + (0.006 if fav else 0),
            text,
            ha="center",
            va="center",
            fontsize=8.5 if fav else 7,
            weight="bold",
            linespacing=0.9,
            color=on_fill,
            zorder=5,
            path_effects=halo,
        )  # above every fill
        if fav:
            ax.text(
                x + dx,
                y + dy - (0.017 if "\n" in text else 0.01),
                fav,
                ha="center",
                va="center",
                fontsize=8.5,
                color=on_fill,
                family="monospace",
                zorder=5,
                path_effects=halo,
            )  # Lekton

    # North-eastern states too small to hold a name: a labelled column on the right.
    xr = max(r[:, 0].max() for rings in shapes.values() for r in rings) + 0.035
    ys = sorted(
        ((label_point(shapes[n])[1], n) for n in CALLOUTS if n in shapes), reverse=True
    )
    step = 0.03
    top = ys[0][0] + 0.03
    for i, (_, name) in enumerate(ys):
        x0, y0 = label_point(shapes[name])
        ty = top - i * step
        _, colour, fav = (
            tile(*probs[name]) if name in probs else ("No race", NO_RACE, "")
        )
        ax.plot([x0, xr - 0.006], [y0, ty], color=MUTED, linewidth=0.7)
        ax.add_patch(
            Rectangle(
                (xr - 0.004, ty - 0.008),
                0.016,
                0.016,
                facecolor=colour,
                edgecolor=PAPER,
            )
        )
        ax.text(
            xr + 0.018,
            ty,
            f"{name}  {fav}" if fav else f"{name}  (no race)",
            va="center",
            fontsize=10,
            color=INK if fav else MUTED,
            weight="bold" if fav else "normal",
        )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    handles = [Patch(facecolor=c, label=lab) for _, lab, c in BANDS] + [
        Patch(facecolor=NO_RACE, label="No race")
    ]
    if any(tile(*v)[0] == "Independent favoured" for v in probs.values()):
        handles.append(Patch(facecolor=OTHER, label="Independent favoured"))
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, 0),
        ncol=4,
        frameon=False,
        fontsize=11,
        title="Each state: its name, then the favoured candidate's party (Dem = Democratic, Rep = Republican,\n"
        "Ind = independent) and its chance of winning. "
        "Solid > 90%, Likely 75-90%, Lean 60-75%, Toss-up: no one above 60%. "
        "Plurality wins; Georgia needs a majority or goes to a runoff.",
        title_fontsize=10,
        alignment="left",
    )
    ax.set_title("Which party is favoured in each governor's race" + note)
    fig.savefig(out / "forecast_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def race_intervals(f: pd.DataFrame, out: Path, note: str):
    f = f.sort_values("margin_mean").reset_index(drop=True)
    y = np.arange(len(f))
    m = lambda col: 2 * f[col] - 100  # two-party share -> margin
    colour = np.where(f["margin_mean"] >= 0, DEM, REP)
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(f) + 1.6))
    ax.hlines(
        y,
        m("dem_share_lo95"),
        m("dem_share_hi95"),
        color=colour,
        linewidth=1.8,
        alpha=0.75,
    )
    ax.hlines(
        y,
        m("dem_share_lo80"),
        m("dem_share_hi80"),
        color=colour,
        linewidth=6.5,
        alpha=0.95,
    )
    ax.scatter(
        m("dem_share_mean"),
        y,
        s=30,
        color=PAPER,
        edgecolor=INK,
        linewidth=1.2,
        zorder=3,
    )
    ax.axvline(0, color=INK, linestyle="--", linewidth=0.8)
    lim = (
        np.ceil(
            max(abs(m("dem_share_lo95")).max(), abs(m("dem_share_hi95")).max()) / 10
        )
        * 10
    )
    ax.set_xlim(lim, -lim)  # Democrats on the left, as in the website's ratings strip
    ax.xaxis.set_major_formatter(LEAN_TICKS)
    ax.set_yticks(y, f["race"])
    ax.set_xlabel("Forecast winning margin (points of the two-party vote)")
    ax.text(
        0.01,
        1.005,
        "Democrat ahead",
        transform=ax.transAxes,
        color=DEM,
        fontsize=11,
        va="bottom",
    )
    ax.text(
        0.99,
        1.005,
        "Republican ahead",
        transform=ax.transAxes,
        color=REP,
        fontsize=11,
        va="bottom",
        ha="right",
    )
    ax.set_title(
        "Forecast margin in every race, with 80% and 95% intervals" + note, pad=18
    )
    ax.grid(axis="x", color=RULE, linewidth=0.6)
    ax.text(
        0.01,
        0.01,
        "thick bar: 80% of simulations\nthin line: 95%",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=MUTED,
    )  # bottom left is empty: safe Republican races
    mono_ticks(ax, "x")
    fig.tight_layout()
    fig.savefig(out / "race_intervals.png", dpi=200)
    plt.close(fig)


def place_labels(ax, xs, ys, names, fontsize=7, avoid=()):
    """Write a full name beside each point without names overlapping each other, the points or
    the texts in `avoid`: try positions around the point, nearest first, and draw a thin leader
    line when a name has to sit further away."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    dots = [to_px((x, y)) for x, y in zip(xs, ys)]
    dot_boxes = [(px - 5, py - 5, px + 5, py + 5) for px, py in dots]
    placed = [tuple(t.get_window_extent(renderer).extents) for t in avoid]
    near = [
        (5, 3, "left", "bottom"),
        (5, -3, "left", "top"),
        (-5, 3, "right", "bottom"),
        (-5, -3, "right", "top"),
        (0, 7, "center", "bottom"),
        (0, -7, "center", "top"),
    ]
    far = [(dx * k, dy * k, ha, va) for k in (2.2, 3.4, 4.6) for dx, dy, ha, va in near]

    def overlaps(box):
        return any(
            not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
            for b in placed + dot_boxes
        )

    order = sorted(
        range(len(names)),
        key=lambda i: (
            -sum(
                abs(dots[i][0] - d[0]) < 60 and abs(dots[i][1] - d[1]) < 30
                for d in dots
            )
        ),
    )
    for i in order:
        for dx, dy, ha, va in near + far:
            t = ax.annotate(
                names[i],
                (xs[i], ys[i]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                ha=ha,
                va=va,
                color=INK,
            )
            bb = t.get_window_extent(renderer)
            box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            if not overlaps(box):
                if (dx, dy, ha, va) in far:
                    t.remove()
                    t = ax.annotate(
                        names[i],
                        (xs[i], ys[i]),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=fontsize,
                        ha=ha,
                        va=va,
                        color=INK,
                        arrowprops={
                            "arrowstyle": "-",
                            "color": MUTED,
                            "linewidth": 0.5,
                            "shrinkA": 0,
                            "shrinkB": 3,
                        },
                    )
                placed.append(box)
                break
            t.remove()
        else:  # nowhere free: keep the nearest position
            ax.annotate(
                names[i],
                (xs[i], ys[i]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=fontsize,
                color=INK,
            )


def silicon_vs_polls(f: pd.DataFrame, out: Path, note: str):
    p = f.dropna(subset=["poll_margin", "sil_margin"])
    fig, ax = plt.subplots(figsize=(8.4, 8.0))  # room for 33 full state names
    if p.empty:
        ax.text(
            0.5,
            0.5,
            "No race has both polls and AI-survey estimates yet",
            ha="center",
            transform=ax.transAxes,
        )
    else:
        x, y = p["poll_margin"].to_numpy(float), p["sil_margin"].to_numpy(float)
        lim = np.ceil(max(abs(x).max(), abs(y).max()) / 10) * 10 + 5
        ax.plot([-lim, lim], [-lim, lim], color=INK, linewidth=0.8)
        ax.axhline(0, color=RULE, linewidth=0.8)
        ax.axvline(0, color=RULE, linewidth=0.8)
        ax.scatter(x, y, s=38, color=ACCENT, edgecolor=PAPER, linewidth=0.8, zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        gap = float(np.mean(y - x))
        side = "more Democratic" if gap > 0 else "more Republican"
        key = (
            f"On average the AI surveys are {abs(gap):.1f} points {side} than the polls.\n"
            "Above the solid line: AI more Democratic. Below it: AI more Republican."
        )
        if CAL and CAL.get("calibration_slope"):
            a_, b_ = CAL["calibration_intercept"], CAL["calibration_slope"]
            xs = np.array([-lim, lim])
            ax.plot(xs, (xs - a_) / b_, color=ACCENT, linestyle="--", linewidth=1)
            key += f"\nDashed line: the fitted correction, poll = {a_:+.1f} + {b_:.2f} \u00d7 AI survey."
        # Under the chart, so it never covers a point or a name.
        ax.text(
            0,
            -0.15,
            key,
            transform=ax.transAxes,
            va="top",
            fontsize=9.5,
            color=MUTED,
            linespacing=1.4,
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(LEAN_TICKS)
        ax.yaxis.set_major_formatter(LEAN_TICKS)
    ax.set_xlabel("Poll average: winning margin")
    ax.set_ylabel("AI surveys (Methods A and B): winning margin")
    ax.set_title("AI surveys against the polls" + note)
    ax.grid(color=RULE, linewidth=0.6)
    mono_ticks(ax, "xy")
    fig.tight_layout()
    if not p.empty:  # names last, once the layout is fixed, so the overlap checks hold in the saved image
        place_labels(ax, x, y, list(p["race"]), fontsize=8.5)
        for t in ax.texts:  # a dark outline keeps names readable where they cross a line
            t.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground=PAPER)])
    fig.savefig(out / "silicon_vs_polls.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def sources(f: pd.DataFrame, out: Path, note: str):
    """What each source says about each race, next to the final forecast."""
    f = f.sort_values("margin_mean").reset_index(drop=True)
    y = np.arange(len(f))
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(f) + 1.8))
    ax.hlines(
        y,
        f[
            ["fund_margin", "silicon_margin_corrected", "poll_margin", "margin_mean"]
        ].min(axis=1),
        f[
            ["fund_margin", "silicon_margin_corrected", "poll_margin", "margin_mean"]
        ].max(axis=1),
        color=MUTED,
        linewidth=1.2,
        alpha=0.5,
    )
    ax.scatter(
        f["fund_margin"],
        y,
        s=40,
        marker="s",
        color=MUTED,
        edgecolor=PAPER,
        linewidth=0.8,
        label="Past results (fundamentals)",
        zorder=3,
    )
    ax.scatter(
        f["silicon_margin_corrected"],
        y,
        s=50,
        marker="^",
        color=ACCENT,
        edgecolor=PAPER,
        linewidth=0.8,
        label="AI surveys (calibrated)",
        zorder=3,
    )
    ax.scatter(
        f["poll_margin"],
        y,
        s=40,
        marker="D",
        facecolor="none",
        edgecolor=INK,
        linewidth=1.3,
        label="Poll average",
        zorder=3,
    )
    for ahead, colour, label in [
        (f["margin_mean"] >= 0, DEM, "Forecast: Democrat ahead"),
        (f["margin_mean"] < 0, REP, "Forecast: Republican ahead"),
    ]:
        ax.scatter(
            f.loc[ahead, "margin_mean"],
            y[ahead.to_numpy()],
            s=70,
            color=colour,
            edgecolor=PAPER,
            linewidth=1,
            label=label,
            zorder=4,
        )
    ax.axvline(0, color=INK, linestyle="--", linewidth=0.8)
    lim = (
        np.ceil(
            np.nanmax(
                np.abs(
                    f[
                        ["fund_margin", "silicon_margin_corrected", "poll_margin"]
                    ].to_numpy()
                )
            )
            / 10
        )
        * 10
    )
    ax.set_xlim(lim, -lim)  # Democrats on the left, as in the website's ratings strip
    ax.xaxis.set_major_formatter(LEAN_TICKS)
    ax.set_yticks(y, f["race"])
    ax.set_xlabel("Winning margin (points of the two-party vote)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    ax.set_title("What each source says, and the forecast that combines them" + note)
    ax.grid(axis="x", color=RULE, linewidth=0.6)
    mono_ticks(ax, "x")
    fig.tight_layout()
    fig.savefig(out / "sources.png", dpi=200)
    plt.close(fig)


def seat_count(sims: pd.DataFrame, out: Path, note: str):
    d, r = sims["dem_governors"].astype(int), sims["rep_governors"].astype(int)
    counts = d.value_counts().sort_index()
    # Blue: Democratic majority. Red: Republican majority. Gold: neither party has 26 (independents hold the balance).
    colour = [
        DEM if k >= 26 else REP if (r[d == k] >= 26).all() else ACCENT
        for k in counts.index
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.bar(counts.index, counts / len(sims), color=colour, width=0.8)
    ax.set_ylim(
        0, (counts / len(sims)).max() * 1.35
    )  # room for the labels above the bars
    ax.axvline(25.5, color=INK, linestyle="--", linewidth=0.8)
    ax.set_xlabel(
        "Democratic governors after the election (of 50, including the 14 seats not on the ballot)"
    )
    ax.set_ylabel("Share of simulations")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.text(
        0.01,
        0.97,
        f"Republicans hold a majority (26+): {(r >= 26).mean():.0%}",
        transform=ax.transAxes,
        va="top",
        color=REP,
        fontsize=11,
        weight="bold",
    )
    neither = ((d < 26) & (r < 26)).mean()
    if neither > 0:
        ax.text(
            0.01,
            0.86,
            f"Neither party has 26: {neither:.0%}",
            transform=ax.transAxes,
            va="top",
            color=ACCENT,
            fontsize=11,
            weight="bold",
        )
    ax.text(
        0.99,
        0.97,
        f"Democrats hold a majority (26+): {(d >= 26).mean():.0%}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        color=DEM,
        fontsize=11,
        weight="bold",
    )
    ax.set_title(
        "Governorships held by each party, across all simulations" + note, pad=10
    )
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    mono_ticks(ax, "xy")
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
    note = (
        "\nMade from simulated data: a test of the pipeline, not a forecast"
        if simulated
        else ""
    )
    forecast_map(f, out, note)
    race_intervals(f, out, note)
    silicon_vs_polls(f, out, note)
    seat_count(sims, out, note)
    if {"fund_margin", "silicon_margin_corrected"} <= set(f.columns):
        sources(f, out, note)
    print(
        f"Saved {len(list(out.glob('*.png')))} charts to {out}"
        + (" - made from simulated inputs, do not publish" if simulated else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-test", action="store_true", help="read data/03-forecast-llm-test instead"
    )
    main(parser.parse_args().llm_test)
