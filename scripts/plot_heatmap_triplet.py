\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Δ-anomaly heatmaps for foF2, hmF2, TEC (2011 & 2022) with eclipse overlays.

For each event date:
  - parse station .dat files
  - build quiet proxy from neighbors (minute-of-day median)
  - Δ = (D0 - Q)/Q * 100 (%)
  - resample to 1-minute grid (small-gap interpolation, light smoothing)
  - imshow heatmaps with Δ [%]
  - eclipse interval shown as a colored line per station

Outputs:
  output/figures/heatmap_triplet_<date>_nearest.(png|pdf)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from eclipse_iono.events import EVENTS_2011, EVENTS_2022, StationEvent
from eclipse_iono.iono_dat import parse_dat_one, three_day_frames, quiet_proxy_from_neighbors, delta_percent
from eclipse_iono.util import ensure_dir


VARS = ["fof2", "hmf2", "tec"]
CLIM = (-40, 40)              # Δ% color scale
CMAP = "RdBu_r"
STEP_MIN = 1
PAD_MIN = 20                  # padding around earliest start / latest end (per event)
GAP_MAX_MIN = 2
SMOOTH_MED = 3
SMOOTH_MEAN = 5

OVERLAY_ALPHA = 0.85
OVERLAY_WIDTH = 1.8

mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.grid": False,
    "axes.linewidth": 0.9,
})

LINESTYLES = [
    (0, (4.0, 2.0)),
    (0, (1.5, 1.5)),
    (0, (6.0, 2.0, 1.6, 2.0)),
    (0, (10.0, 2.0)),
    (0, (2.0, 2.0, 2.0, 2.0)),
    (0, (8.0, 3.0, 2.0, 3.0)),
    (0, (1.0, 1.0)),
    (0, (12.0, 4.0)),
]

OVERLAY_COLORS = [
    "k", "tab:orange", "tab:green", "tab:purple",
    "tab:red", "tab:blue", "tab:brown", "tab:pink",
]


def build_window(stations: List[StationEvent], pad_min: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return (
        min(e.t_start for e in stations) - pd.Timedelta(minutes=pad_min),
        max(e.t_end for e in stations) + pd.Timedelta(minutes=pad_min),
    )


def time_grid(t0: pd.Timestamp, t1: pd.Timestamp, step_min: int) -> pd.DatetimeIndex:
    return pd.date_range(t0, t1, freq=f"{step_min}min", tz="UTC")


def interp_small_gaps(s: pd.Series, max_gap_min: int) -> pd.Series:
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="1min", tz="UTC"))
    interp = full.interpolate(method="time", limit=max_gap_min, limit_direction="both")
    return interp


def temporal_smooth(s: pd.Series, w_med: int, w_mean: int) -> pd.Series:
    out = s.rolling(window=w_med, center=True, min_periods=1).median()
    out = out.rolling(window=w_mean, center=True, min_periods=1).mean()
    return out


def resample_to_grid(delta: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    interp = interp_small_gaps(delta, GAP_MAX_MIN)
    sm = temporal_smooth(interp, SMOOTH_MED, SMOOTH_MEAN)
    aligned = sm.reindex(grid, method="nearest", tolerance=pd.Timedelta(seconds=30))
    return aligned


def read_station_delta(folder: Path, event_date: pd.Timestamp, code: str, var: str) -> Optional[pd.Series]:
    p = folder / f"{code}.dat"
    if not p.exists():
        return None
    df = parse_dat_one(p, event_date)
    if df is None or df.empty:
        return None
    dm, d0, dp = three_day_frames(df, event_date)
    if var not in d0.columns:
        return None
    y0 = d0[var].astype(float)
    yq = quiet_proxy_from_neighbors(dm, dp, d0.index, var)
    return delta_percent(y0, yq)


def overlay_eclipse_lines(ax, tg: pd.DatetimeIndex, stations: List[StationEvent], add_legend: bool) -> None:
    handles = []
    for r, e in enumerate(stations):
        # draw line across indices where we're inside eclipse [start, end]
        xs = [i for i, T in enumerate(tg) if (T >= e.t_start and T <= e.t_end)]
        if not xs:
            continue
        ls = LINESTYLES[r % len(LINESTYLES)]
        color = OVERLAY_COLORS[r % len(OVERLAY_COLORS)]
        ln, = ax.plot(xs, [r + 0.5] * len(xs), color=color, lw=OVERLAY_WIDTH, alpha=OVERLAY_ALPHA, linestyle=ls)
        handles.append((ln, e.code))

    if handles and add_legend:
        leg = ax.legend([h[0] for h in handles], [h[1] for h in handles],
                        loc="upper right", frameon=False, fontsize=9,
                        title="Eclipse interval", title_fontsize=9)
        for lh in leg.legend_handles:
            lh.set_linewidth(OVERLAY_WIDTH)


def draw_time_ticks(ax, tg: pd.DatetimeIndex) -> None:
    nx = len(tg)
    want = 10
    stride = max(1, int(np.ceil(nx / want)))
    xt = np.arange(0, nx, stride)
    ax.set_xticks(xt)
    ax.set_xticklabels([tg[i].strftime("%H:%M") for i in xt])
    ax.set_xlabel("UT")


def build_matrix(event_date: pd.Timestamp, folder: Path, stations: List[StationEvent], var: str, t0: pd.Timestamp, t1: pd.Timestamp) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex]:
    tg = time_grid(t0, t1, step_min=STEP_MIN)
    rows, codes = [], []
    for e in stations:
        dlt = read_station_delta(folder, event_date, e.code, var)
        codes.append(e.code)
        if dlt is None or dlt.empty:
            rows.append(np.full(len(tg), np.nan))
            print(f"[MISS] {e.code}: {e.code}.dat or missing {var}")
            continue
        on_grid = resample_to_grid(dlt, tg)
        rows.append(on_grid.to_numpy())
        cov = np.isfinite(on_grid).sum() / len(tg) * 100.0
        print(f"[COV ] {e.code} {var}: {cov:.1f}%")
    M = np.vstack(rows) if rows else np.zeros((0, len(tg)))
    return M, codes, tg


def plot_triplet(event_date: pd.Timestamp, folder: Path, stations: List[StationEvent], outdir: Path, interpolation: str) -> None:
    t0, t1 = build_window(stations, pad_min=PAD_MIN)
    print(f"[TRIPLET] {event_date.date()} window={t0:%H:%M}–{t1:%H:%M} UTC | folder={folder}")

    matrices = []
    tg = None
    codes = None
    for var in VARS:
        M, codes_var, tg_var = build_matrix(event_date, folder, stations, var, t0, t1)
        matrices.append(M)
        tg = tg_var if tg is None else tg
        codes = codes_var if codes is None else codes

    ny, nx = matrices[0].shape

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(15.5, 8.0))
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.08, top=0.97, hspace=0.04)

    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("white")
    norm = Normalize(*CLIM)

    last_im = None
    titles = [r"$\Delta$ foF2 (%)", r"$\Delta$ hmF2 (%)", r"$\Delta$ TEC (%)"]

    for i, (M, ax) in enumerate(zip(matrices, axes)):
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", origin="lower",
                       interpolation=interpolation, cmap=cmap, norm=norm, extent=[0, nx, 0, ny])
        last_im = im

        ax.set_yticks(np.arange(ny) + 0.5)
        if i == 1:
            ax.set_yticklabels(codes)
            ax.set_ylabel("Station")
        else:
            ax.set_yticklabels([])

        ax.text(0.01, 0.93, titles[i], transform=ax.transAxes, ha="left", va="top", fontsize=11)

        if i == 2:
            draw_time_ticks(ax, tg)
        else:
            ax.set_xticklabels([])

        overlay_eclipse_lines(ax, tg, stations, add_legend=(i == 0))

    cax = fig.add_axes([0.88, 0.10, 0.018, 0.80])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label(r"$\Delta$ anomaly (%)")

    out_png = outdir / f"heatmap_triplet_{event_date.date()}_{interpolation}.png"
    out_pdf = outdir / f"heatmap_triplet_{event_date.date()}_{interpolation}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote: {out_png} / {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--outdir", type=Path, default=Path("output/figures"))
    ap.add_argument("--interpolation", type=str, default="nearest")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    jobs = [
        (pd.Timestamp("2011-01-04", tz="UTC"), args.data_root / "2011-01-04", EVENTS_2011),
        (pd.Timestamp("2022-10-25", tz="UTC"), args.data_root / "2022-10-25", EVENTS_2022),
    ]

    for event_date, folder, stations in jobs:
        if not folder.exists():
            print(f"[MISS] folder: {folder}")
            continue
        plot_triplet(event_date, folder, stations, outdir, interpolation=args.interpolation)


if __name__ == "__main__":
    main()
