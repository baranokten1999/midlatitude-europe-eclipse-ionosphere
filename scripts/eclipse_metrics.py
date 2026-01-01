\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eclipse–ionosphere response metrics.

For each event × station × variable, compute:
  - most negative Δ within a station window (t_start-30min .. t_end+30min)
  - lag(t_min vs t_great)
  - eta = -Δ_min / (100 * peak_mag)
  - corr(-Δ, eclipse_shape) at zero lag, and best lag within ±60 min

Outputs:
  - prints summary to stdout
  - output/eta_vs_emax.pdf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import numpy.linalg as la
import matplotlib.pyplot as plt

from eclipse_iono.events import EVENTS_2011, EVENTS_2022, EVENT_META, StationEvent
from eclipse_iono.iono_dat import parse_dat_one, three_day_frames, quiet_proxy_from_neighbors, delta_percent
from eclipse_iono.util import ensure_dir


def triangular_eclipse_fraction(t: pd.Timestamp, ev: StationEvent) -> float:
    """0 at t_start, 1 at t_great, 0 at t_end, scaled by peak_mag."""
    if t.tz is None:
        t = t.tz_localize("UTC")
    if t < ev.t_start or t > ev.t_end:
        return 0.0
    if ev.t_great <= ev.t_start or ev.t_end <= ev.t_great:
        return float(ev.peak_mag)

    if t <= ev.t_great:
        frac = (t - ev.t_start).total_seconds() / (ev.t_great - ev.t_start).total_seconds()
    else:
        frac = (ev.t_end - t).total_seconds() / (ev.t_end - ev.t_great).total_seconds()

    frac = max(0.0, min(1.0, frac))
    return float(frac * ev.peak_mag)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(xm * ym) / denom)


def lagged_corr(delta_arr: np.ndarray, shape_arr: np.ndarray, dt_minutes: float, max_lag_minutes: float = 60.0) -> Tuple[float, float, float]:
    """corr(-Δ, E(t+τ)) over τ in [-max_lag, +max_lag] (steps of dt)."""
    n = delta_arr.size
    if n < 4:
        return float("nan"), float("nan"), float("nan")

    rho0 = pearson_corr(-delta_arr, shape_arr)
    max_lag_steps = int(round(max_lag_minutes / dt_minutes))

    best_rho = -1.0
    best_tau = 0.0

    for lag_steps in range(-max_lag_steps, max_lag_steps + 1):
        if lag_steps == 0:
            rho = rho0
            tau = 0.0
        elif lag_steps > 0:
            # correlate Δ(t) with shape(t+tau): shift shape forward by lag_steps
            s_shift = shape_arr[lag_steps:]
            d_shift = delta_arr[:-lag_steps]
            tau = lag_steps * dt_minutes
            rho = pearson_corr(-d_shift, s_shift)
        else:
            k = -lag_steps
            s_shift = shape_arr[:-k]
            d_shift = delta_arr[k:]
            tau = -k * dt_minutes
            rho = pearson_corr(-d_shift, s_shift)

        if np.isfinite(rho) and rho > best_rho:
            best_rho = float(rho)
            best_tau = float(tau)

    return float(rho0), float(best_rho), float(best_tau)


def load_all(data_root: Path) -> Dict[Tuple[str, str], Dict[str, pd.DataFrame]]:
    all_data: Dict[Tuple[str, str], Dict[str, pd.DataFrame]] = {}

    for date_str, folder in [("2011-01-04", data_root / "2011-01-04"), ("2022-10-25", data_root / "2022-10-25")]:
        base = pd.Timestamp(date_str, tz="UTC")
        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue
        for p in sorted(folder.glob("*.dat")):
            sta = p.stem.upper()
            df = parse_dat_one(p, base)
            if df is None or df.empty:
                print(f"[WARN] {sta}: empty or unreadable")
                continue
            dm, d0, dp = three_day_frames(df, base)
            all_data[(date_str, sta)] = {"dm": dm, "d0": d0, "dp": dp}

    return all_data


def global_summary_and_regression(metrics: List[Dict[str, Any]]) -> None:
    vars_order = [("hmf2", "hmF2"), ("tec", "TEC"), ("fof2", "foF2")]

    print("\nSUMMARY (per variable)\n")
    for key, label in vars_order:
        rows = [m for m in metrics if m["var_key"] == key and np.isfinite(m["eta"])]
        if not rows:
            continue
        eta_vals = np.array([m["eta"] for m in rows], dtype=float)
        chi_vals = np.array([m["chi_min"] for m in rows if np.isfinite(m["chi_min"])], dtype=float)
        lam_vals = np.array([abs(m["lambda_best"]) for m in rows if np.isfinite(m["lambda_best"])], dtype=float)
        print(f"Variable: {label}")
        print(f"  N rows                = {len(rows)}")
        print(f"  <eta>                 = {np.nanmean(eta_vals):6.3f}")
        print(f"  median(eta)           = {np.nanmedian(eta_vals):6.3f}")
        print(f"  <chi_min>             = {np.nanmean(chi_vals):6.3f}")
        print(f"  median(|lambda_best|) = {np.nanmedian(lam_vals):6.3f}\n")

    print("MULTIVARIATE REGRESSION (eta vs peak_mag, lat_deg, geom_ratio, |lambda_best|)\n")
    for key, label in vars_order:
        rows = [m for m in metrics
                if m["var_key"] == key
                and np.isfinite(m["eta"])
                and np.isfinite(m["peak_mag"])
                and np.isfinite(m["lat_deg"])
                and np.isfinite(m["geom_ratio"])
                and np.isfinite(m["lambda_best"])]
        if len(rows) < 5:
            continue

        eta_vals = np.array([m["eta"] for m in rows], dtype=float)
        peak_mag = np.array([m["peak_mag"] for m in rows], dtype=float)
        lat_deg  = np.array([m["lat_deg"] for m in rows], dtype=float)
        geom_rat = np.array([m["geom_ratio"] for m in rows], dtype=float)
        lam_abs  = np.array([abs(m["lambda_best"]) for m in rows], dtype=float)

        X = np.column_stack([np.ones_like(eta_vals), peak_mag, lat_deg, geom_rat, lam_abs])
        beta, *_ = la.lstsq(X, eta_vals, rcond=None)
        eta_pred = X @ beta

        ss_res = np.sum((eta_vals - eta_pred) ** 2)
        ss_tot = np.sum((eta_vals - eta_vals.mean()) ** 2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        print(f"Variable: {label}")
        print(f"  N points      = {len(rows)}")
        print(f"  R^2           = {R2:6.3f}")
        print("  Coefficients (eta = b0 + b1*peak_mag + b2*lat_deg + b3*geom_ratio + b4*|lambda_best|):")
        print(f"    b0 (offset)     = {beta[0]:8.3f}")
        print(f"    b1 (peak_mag)   = {beta[1]:8.3f}")
        print(f"    b2 (lat_deg)    = {beta[2]:8.3f}")
        print(f"    b3 (geom_ratio) = {beta[3]:8.3f}")
        print(f"    b4 (|lambda|)   = {beta[4]:8.3f}\n")


def plot_eta_vs_emax(metrics: List[Dict[str, Any]], outdir: Path) -> None:
    vars_order = [("tec", "TEC"), ("hmf2", "hmF2"), ("fof2", "foF2")]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 8))

    event_markers = {
        "2011-01-04": ("o", "2011-01-04"),
        "2022-10-25": ("^", "2022-10-25"),
    }

    for ax, (key, label) in zip(axes, vars_order):
        rows = [m for m in metrics if m["var_key"] == key and np.isfinite(m["eta"]) and np.isfinite(m["peak_mag"])]
        for m in rows:
            evt = m["event_str"]
            marker, _ = event_markers.get(evt, ("o", evt))
            ax.scatter(m["peak_mag"], m["eta"], marker=marker, edgecolor="k", facecolor="none", zorder=3)
            ax.text(m["peak_mag"] + 0.005, m["eta"], m["station"], fontsize=8)
        ax.set_ylabel(rf"$\eta_{{\mathrm{{{label}}}}}$")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(r"Maximum obscuration $E_{\max}$ at station")

    handles = []
    for evt, (marker, label_evt) in event_markers.items():
        h = axes[0].scatter([], [], marker=marker, edgecolor="k", facecolor="none", label=label_evt)
        handles.append(h)
    axes[0].legend(handles=handles, title="Event", loc="best", fontsize=8)

    fig.tight_layout()
    outpath = outdir / "eta_vs_emax.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote: {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--outdir", type=Path, default=Path("output"))
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    figdir = ensure_dir(outdir)

    all_data = load_all(args.data_root)
    if not all_data:
        print("[ERROR] No data loaded.")
        return

    # Event-level maximum peak magnitude across stations
    peak_by_event: Dict[str, float] = {}
    for (event_str, sta), _parts in all_data.items():
        ev = EVENT_META.get((event_str, sta.upper()))
        if ev is None:
            continue
        peak_by_event[event_str] = max(peak_by_event.get(event_str, 0.0), float(ev.peak_mag))

    VARS = [("fof2", "foF2"), ("hmf2", "hmF2"), ("tec", "TEC")]
    metrics: List[Dict[str, Any]] = []

    for (event_str, sta), parts in sorted(all_data.items()):
        sta_key = sta.upper()
        ev = EVENT_META.get((event_str, sta_key))
        if ev is None:
            continue
        dm, d0, dp = parts["dm"], parts["d0"], parts["dp"]
        if d0.empty:
            continue

        peak_evt = peak_by_event.get(event_str, float("nan"))
        peak_sta = float(ev.peak_mag)
        geom_ratio = (peak_sta / peak_evt) if np.isfinite(peak_evt) and peak_evt > 0 else float("nan")

        t0 = ev.t_start - pd.Timedelta(minutes=30)
        t1 = ev.t_end + pd.Timedelta(minutes=30)

        idx_d0 = d0.index
        mask_win = (idx_d0 >= t0) & (idx_d0 <= t1)

        eclipse_shape = pd.Series([triangular_eclipse_fraction(t, ev) for t in idx_d0], index=idx_d0, dtype=float)
        ecl_duration_min = (ev.t_end - ev.t_start).total_seconds() / 60.0

        for var_key, var_label in VARS:
            if var_key not in d0.columns:
                continue

            y0 = d0[var_key].astype(float)
            yq = quiet_proxy_from_neighbors(dm, dp, d0.index, var_key)
            dlt = delta_percent(y0, yq)

            d_win = dlt[mask_win]
            s_win = eclipse_shape[mask_win]
            mask_valid = np.isfinite(d_win.values) & np.isfinite(s_win.values) & (s_win.values > 0)

            if not mask_valid.any():
                continue

            d_arr = d_win.values[mask_valid]
            s_arr = s_win.values[mask_valid]
            t_arr = d_win.index[mask_valid]

            dt_minutes = (t_arr[1] - t_arr[0]).total_seconds() / 60.0 if len(t_arr) > 1 else 15.0

            i_min = int(np.argmin(d_arr))
            delta_min = float(d_arr[i_min])
            t_min = t_arr[i_min]
            ecl_at_min = float(s_arr[i_min])

            lag_min_vs_great = (t_min - ev.t_great).total_seconds() / 60.0

            chi_min = float(ecl_at_min / peak_sta) if peak_sta > 0 else float("nan")
            chi_min = max(0.0, min(1.0, chi_min)) if np.isfinite(chi_min) else float("nan")

            eta = (-delta_min / (100.0 * peak_sta)) if peak_sta > 0 else float("nan")

            rho0, rho_max, tau_best = lagged_corr(d_arr, s_arr, dt_minutes)

            metrics.append(dict(
                event_str=event_str,
                station=sta_key,
                var_key=var_key,
                var_label=var_label,
                peak_mag=peak_sta,
                lat_deg=float(ev.lat_deg),
                geom_ratio=float(geom_ratio),
                delta_min=delta_min,
                chi_min=chi_min,
                eta=eta,
                rho0=rho0,
                rho_max=rho_max,
                tau_best_minutes=tau_best,
                lambda_best=(tau_best / ecl_duration_min if ecl_duration_min > 0 else np.nan),
                lag_min_vs_great=lag_min_vs_great,
            ))

    # Print a compact table
    print("\nCollected metrics rows:", len(metrics))
    for m in metrics:
        print(f"{m['event_str']} {m['station']:>5} {m['var_label']:>4}  "
              f"eta={m['eta']:+.3f}  Δmin={m['delta_min']:+.1f}%  "
              f"lag_min={m['lag_min_vs_great']:+.1f}min  rho0={m['rho0']:+.2f}  best={m['rho_max']:+.2f}@{m['tau_best_minutes']:+.0f}min")

    global_summary_and_regression(metrics)
    plot_eta_vs_emax(metrics, outdir=figdir)


if __name__ == "__main__":
    main()
