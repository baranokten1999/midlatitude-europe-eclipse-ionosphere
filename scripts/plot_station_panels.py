\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-panel station plots from .dat files (three blocks: D-1, D0, D+1).

For each station/event:
  - D0 vs UT for foF2, hmF2, TEC
  - quiet proxy Q from D-1/D+1 (minute-of-day median)
  - Δ = (D0 - Q)/Q * 100 (%)
  - left column: full day
  - right column: zoom around station eclipse (start-30 min to end+30 min)
  - station-specific eclipse shading (pre / during / post)

Outputs:
  output/figures/<STA>_<YYYY-MM-DD>_multi.(png|pdf)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib import dates as mdates

from eclipse_iono.events import EVENT_META
from eclipse_iono.iono_dat import parse_dat_one, three_day_frames, quiet_proxy_from_neighbors, delta_percent
from eclipse_iono.util import ensure_dir


# ---- shading padding around station eclipse start/end
PRE_PAD_MIN = 20
POST_PAD_MIN = 20

# ---- plot style
COL_D0 = "tab:blue"
COL_Q = "0.15"
COL_D = "tab:red"

LAB_D0 = "D"
LAB_Q = "Q"
LAB_D = "Δ"

plt.rcParams.update({
    "figure.figsize": (16, 10),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "savefig.dpi": 220,
})


def shade_station_interval(ax, ev):
    if ev is None:
        return
    pre_L = ev.t_start - pd.Timedelta(minutes=PRE_PAD_MIN)
    pre_R = ev.t_start
    peak_L = ev.t_start
    peak_R = ev.t_end
    post_L = ev.t_end
    post_R = ev.t_end + pd.Timedelta(minutes=POST_PAD_MIN)

    for (L, R), alpha in [((pre_L, pre_R), 0.15), ((peak_L, peak_R), 0.30), ((post_L, post_R), 0.15)]:
        ax.axvspan(L, R, color="0.5", alpha=alpha, lw=0)


def compute_common_ylim(y0: pd.Series, yq: pd.Series, mask_zoom: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    def lim(series_list: List[pd.Series]) -> Tuple[float, float]:
        vals = []
        for s in series_list:
            v = s.to_numpy()
            vals.append(v[np.isfinite(v)])
        if not vals:
            return (0.0, 1.0)
        vv = np.concatenate(vals) if len(vals) > 1 else vals[0]
        if vv.size == 0:
            return (0.0, 1.0)
        lo, hi = np.nanmin(vv), np.nanmax(vv)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return (0.0, 1.0)
        if hi == lo:
            d = 0.1 * abs(hi) + 1.0
            return (lo - d, hi + d)
        pad = 0.06 * (hi - lo)
        return (lo - pad, hi + pad)

    full_ylim = lim([y0, yq])
    zoom_ylim = lim([y0[mask_zoom], yq[mask_zoom]])
    return full_ylim, zoom_ylim


def compute_delta_ylim(d: pd.Series, mask_zoom: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    def lim(s: pd.Series) -> Tuple[float, float]:
        v = s.to_numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (-10.0, 10.0)
        lo, hi = np.nanmin(v), np.nanmax(v)
        if hi == lo:
            d = 0.1 * abs(hi) + 5.0
            return (lo - d, hi + d)
        pad = 0.08 * (hi - lo)
        return (lo - pad, hi + pad)

    full = lim(d)
    zoom = lim(d[mask_zoom])
    return full, zoom


def plot_one_station(event_date: pd.Timestamp, station_code: str, dm: pd.DataFrame, d0: pd.DataFrame, dp: pd.DataFrame, outdir: Path) -> None:
    date_key = str(event_date.date())
    sta_key = station_code.upper()

    ev_meta = EVENT_META.get((date_key, sta_key), None)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        6, 2,
        height_ratios=[3, 1, 3, 1, 3, 1],
        width_ratios=[3.6, 2.3],
        hspace=0.10, wspace=0.14,
    )

    # Left column (full-day)
    axF = fig.add_subplot(gs[0, 0])
    axFd = fig.add_subplot(gs[1, 0], sharex=axF)
    axH = fig.add_subplot(gs[2, 0], sharex=axF)
    axHd = fig.add_subplot(gs[3, 0], sharex=axF)
    axT = fig.add_subplot(gs[4, 0], sharex=axF)
    axTd = fig.add_subplot(gs[5, 0], sharex=axF)

    # Right column (zoom)
    axFz = fig.add_subplot(gs[0, 1])
    axFdz = fig.add_subplot(gs[1, 1], sharex=axFz)
    axHz = fig.add_subplot(gs[2, 1], sharex=axFz)
    axHdz = fig.add_subplot(gs[3, 1], sharex=axFz)
    axTz = fig.add_subplot(gs[4, 1], sharex=axFz)
    axTdz = fig.add_subplot(gs[5, 1], sharex=axFz)

    # Full-day x-limits
    day_start = pd.Timestamp(f"{date_key} 00:00", tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)
    ticks4h = pd.date_range(day_start, day_end, freq="4h")
    hourly_minor = mdates.HourLocator(interval=1)

    for ax in (axF, axFd, axH, axHd, axT, axTd):
        ax.set_xlim(day_start, day_end)
        ax.set_xticks(ticks4h)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_locator(hourly_minor)

    # Zoom window: station-specific, start-30 min to end+30 min
    if ev_meta is not None:
        z0 = ev_meta.t_start - pd.Timedelta(minutes=30)
        z1 = ev_meta.t_end + pd.Timedelta(minutes=30)
    else:
        z0 = day_start + pd.Timedelta(hours=6)
        z1 = day_start + pd.Timedelta(hours=12)

    for ax in (axFz, axFdz, axHz, axHdz, axTz, axTdz):
        ax.set_xlim(z0, z1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    VARS = [
        ("fof2", r"$f_{o\mathrm{F2}}$", "MHz", (axF, axFd, axFz, axFdz)),
        ("hmf2", r"$h_{m\mathrm{F2}}$", "km",  (axH, axHd, axHz, axHdz)),
        ("tec",  "TEC",                 "TECU",(axT, axTd, axTz, axTdz)),
    ]

    legend_handles = None

    for var, label_tex, unit_tex, (a_main, a_delta, a_zoom, a_zdelta) in VARS:
        if var not in d0.columns:
            continue

        y0 = d0[var].astype(float)
        yq = quiet_proxy_from_neighbors(dm, dp, d0.index, var)
        dlt = delta_percent(y0, yq)

        mask_zoom = (y0.index >= z0) & (y0.index <= z1)
        full_ylim, zoom_ylim = compute_common_ylim(y0, yq, mask_zoom.to_numpy())
        full_dylim, zoom_dylim = compute_delta_ylim(dlt, mask_zoom.to_numpy())

        # Full-day D & Q
        a_main.plot(y0.index, y0, color=COL_D0, lw=1.8, label=LAB_D0)
        a_main.plot(yq.index, yq, color=COL_Q,  lw=1.5, label=LAB_Q)
        a_main.set_ylabel(f"{label_tex} [{unit_tex}]")
        a_main.set_ylim(*full_ylim)
        shade_station_interval(a_main, ev_meta)

        # Full-day Δ
        a_delta.plot(dlt.index, dlt, color=COL_D, lw=1.5, label=LAB_D)
        a_delta.set_ylim(*full_dylim)
        shade_station_interval(a_delta, ev_meta)

        # Zoom D & Q
        a_zoom.plot(y0.index, y0, color=COL_D0, lw=1.8)
        a_zoom.plot(yq.index, yq, color=COL_Q,  lw=1.5)
        a_zoom.set_ylim(*zoom_ylim)
        shade_station_interval(a_zoom, ev_meta)

        # Zoom Δ
        a_zdelta.plot(dlt.index, dlt, color=COL_D, lw=1.5)
        a_zdelta.set_ylim(*zoom_dylim)
        shade_station_interval(a_zdelta, ev_meta)

        if legend_handles is None:
            legend_handles = [
                Line2D([0], [0], color=COL_D0, lw=2.0, label=LAB_D0),
                Line2D([0], [0], color=COL_Q,  lw=1.6, label=LAB_Q),
                Line2D([0], [0], color=COL_D,  lw=1.6, label=LAB_D),
            ]

    for ax in (axF, axFz):
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_minor_locator(MaxNLocator(10))

    # Hide x tick labels except bottom row
    for ax in (axF, axFd, axH, axHd, axT, axFz, axFdz, axHz, axHdz, axTz):
        ax.tick_params(labelbottom=False)

    axTd.set_xlabel("UT")
    axTdz.set_xlabel("UT")

    fig.align_ylabels([axF, axH, axT])

    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.03),
                   ncol=3, frameon=False, fontsize=10, handlelength=2.8, columnspacing=1.8)

    fig.suptitle(f"{sta_key} — {date_key}", y=0.99, fontsize=14)

    out_base = outdir / f"{sta_key}_{date_key}_multi"
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {sta_key} {date_key} -> {out_base.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"), help="Root folder containing date subfolders.")
    ap.add_argument("--outdir", type=Path, default=Path("output/figures"))
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    jobs = [
        ("2011-01-04", args.data_root / "2011-01-04"),
        ("2022-10-25", args.data_root / "2022-10-25"),
    ]

    any_done = False
    for date_str, folder in jobs:
        base = pd.Timestamp(date_str, tz="UTC")
        if not folder.exists():
            print(f"[MISS] folder: {folder}")
            continue

        for p in sorted(folder.glob("*.dat")):
            sta = p.stem.upper()
            df = parse_dat_one(p, base)
            if df is None or df.empty:
                print(f"[WARN] {sta}: empty or unreadable")
                continue
            dm, d0, dp = three_day_frames(df, base)
            if d0.empty:
                print(f"[SKIP] {sta} {date_str}: D0 empty")
                continue
            plot_one_station(base, sta, dm, d0, dp, outdir)
            any_done = True

    if not any_done:
        print("[ERROR] Nothing plotted. Check data folders + filenames.")


if __name__ == "__main__":
    main()
