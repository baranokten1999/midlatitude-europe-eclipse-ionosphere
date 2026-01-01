\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eclipse obscuration map (2011-01-04 & 2022-10-25) with contour lines and ionosonde stations.

- Computes max obscuration over a lat/lon grid using PyEphem (no external data).
- Uses a compressed cache for the expensive grid sampling.
- Saves PNG/PDF to output/.

Outputs:
  output/eclipse_overlay_lines_with_stations.png
  output/eclipse_overlay_lines_with_stations.pdf
  output/eclipse_cache/grid_<tag>_<hash>.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ephem

from eclipse_iono.util import ensure_dir


# --------------------------- STATIONS ---------------------------
STATIONS_2011 = [
    {"code": "AT138", "lat": 38.00, "lon_e": 23.50},    # Athens
    {"code": "DB049", "lat": 50.10, "lon_e": 4.60},     # Dourbes
    {"code": "RL052", "lat": 51.50, "lon_e": 359.40},   # Chilton (359.40°E = -0.60°)
    {"code": "EB040", "lat": 40.80, "lon_e": 0.50},     # Roquetes
    {"code": "MO155", "lat": 55.76, "lon_e": 38.28},    # Elektrogli/Moscow
]

STATIONS_2022 = [
    {"code": "AT138", "lat": 38.00, "lon_e": 23.50},    # Athens
    {"code": "FF051", "lat": 51.70, "lon_e": 358.50},   # Fairford
    {"code": "VT139", "lat": 40.60, "lon_e": 17.80},    # San Vito
    {"code": "RO041", "lat": 41.90, "lon_e": 12.50},    # Rome
]


def lon_from_e(lon_e: float) -> float:
    """Convert [0,360]E to [-180,180] longitude."""
    return lon_e if lon_e <= 180.0 else lon_e - 360.0


def split_stations(st2011, st2022):
    """Separate AT138 (common) from 2011-only and 2022-only."""
    at138 = None
    s11, s22 = [], []
    for s in st2011:
        if s["code"] == "AT138":
            at138 = s
        else:
            s11.append(s)
    for s in st2022:
        if s["code"] == "AT138":
            at138 = s if at138 is None else at138
        else:
            s22.append(s)
    return at138, s11, s22


AT138, STATIONS_2011_ONLY, STATIONS_2022_ONLY = split_stations(STATIONS_2011, STATIONS_2022)

# --------------------------- CONFIG ---------------------------
EV_2011 = ("2011-01-04", datetime(2011, 1, 4, 6, 0, tzinfo=timezone.utc), datetime(2011, 1, 4, 14, 0, tzinfo=timezone.utc))
EV_2022 = ("2022-10-25", datetime(2022, 10, 25, 7, 0, tzinfo=timezone.utc), datetime(2022, 10, 25, 14, 0, tzinfo=timezone.utc))

LEVELS = np.array([0.20, 0.40, 0.60, 0.80], dtype=float)

COLOR_2011, LS_2011, LW_2011 = "#1f77b4", ":", 1.6
COLOR_2022, LS_2022, LW_2022 = "#b40426", "--", 1.8

LABEL_FONTSIZE = 8
PAINT_ALPHA = 0.05

# Station markers
MK_2011, EC_2011, FC_2011, SZ_2011 = "^", "#2ca02c", "#2ca02c", 45
MK_2022, EC_2022, FC_2022, SZ_2022 = "s", "#7e57c2", "#7e57c2", 45
MK_AT,   EC_AT,   FC_AT,   SZ_AT   = "*", "#b8860b", "#ffd54d", 110
TEXT_FS = 7

# --------------------- GEOMETRY / PHYSICS ---------------------
def circle_overlap_area(R: float, r: float, d: float) -> float:
    if d >= R + r:
        return 0.0
    if d <= abs(R - r):
        return math.pi * min(R, r) ** 2
    R2, r2, d2 = R * R, r * r, d * d
    alpha = math.acos(max(-1.0, min(1.0, (d2 + R2 - r2) / (2 * d * R))))
    beta = math.acos(max(-1.0, min(1.0, (d2 + r2 - R2) / (2 * d * r))))
    return R2 * alpha + r2 * beta - 0.5 * math.sqrt(
        max(0.0, (-d + R + r) * (d + R - r) * (d - R + r) * (d + R + r))
    )


def obscuration_fraction(sun_radius_rad: float, moon_radius_rad: float, sep_rad: float) -> float:
    return circle_overlap_area(sun_radius_rad, moon_radius_rad, sep_rad) / (math.pi * sun_radius_rad ** 2)


# --------------------- OBSCURATION CORE -----------------------
def max_obscuration_for_point(lat_deg: float, lon_deg: float, t0_utc, t1_utc, dt_minutes: int) -> float:
    """Scan max obscuration for a single (lat,lon) between t0_utc..t1_utc."""
    obs = ephem.Observer()
    obs.lat = f"{lat_deg:.8f}"
    obs.lon = f"{lon_deg:.8f}"
    obs.elevation = 0
    obs.pressure = 0
    obs.temperature = 15.0

    sun, moon = ephem.Sun(), ephem.Moon()
    dt = timedelta(minutes=dt_minutes)
    t = t0_utc
    max_obs = 0.0

    while t <= t1_utc:
        obs.date = ephem.Date(t)
        sun.compute(obs)
        moon.compute(obs)

        # Sun above horizon
        if sun.alt > 0:
            sun_rad = (sun.size / 2.0) * (math.pi / (180.0 * 3600.0))
            moon_rad = (moon.size / 2.0) * (math.pi / (180.0 * 3600.0))
            sep_rad = float(ephem.separation((sun.az, sun.alt), (moon.az, moon.alt)))
            if sep_rad < (sun_rad + moon_rad):
                f = obscuration_fraction(sun_rad, moon_rad, sep_rad)
                if f > max_obs:
                    max_obs = f

        t += dt

    return float(max_obs)


# --------------------- GRID & CACHING -------------------------
def cache_key(cache_dir: Path, extent, dlon, dlat, t0, t1, dt_minutes, tag) -> Path:
    payload = dict(
        extent=extent,
        dlon=dlon,
        dlat=dlat,
        t0=t0.isoformat(),
        t1=t1.isoformat(),
        dt=dt_minutes,
        tag=tag,
    )
    s = json.dumps(payload, sort_keys=True).encode()
    h = hashlib.md5(s).hexdigest()[:12]
    return cache_dir / f"grid_{tag}_{h}.npz"


def compute_obscuration_grid_threaded(
    extent: Tuple[float, float, float, float],
    dlon: float,
    dlat: float,
    t0,
    t1,
    dt_minutes: int,
    tag: str,
    cache_dir: Path,
    recompute: bool,
    max_workers: int,
    batch: int,
):
    """Thread-parallel grid with caching."""
    cache_path = cache_key(cache_dir, extent, dlon, dlat, t0, t1, dt_minutes, tag)
    if (not recompute) and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return data["lons"], data["lats"], data["grid"]

    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.arange(lon_min, lon_max + 1e-9, dlon)
    lats = np.arange(lat_min, lat_max + 1e-9, dlat)
    grid = np.zeros((lats.size, lons.size), dtype=np.float32)

    tasks = [(i, j, float(lat), float(lon)) for i, lat in enumerate(lats) for j, lon in enumerate(lons)]

    def worker(i: int, j: int, lat: float, lon: float):
        v = max_obscuration_for_point(lat, lon, t0, t1, dt_minutes)
        return i, j, v

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for k in range(0, len(tasks), batch):
            futs = [ex.submit(worker, *t) for t in tasks[k : k + batch]]
            for fut in as_completed(futs):
                i, j, v = fut.result()
                grid[i, j] = v

    np.savez_compressed(cache_path, lons=lons, lats=lats, grid=grid)
    return lons, lats, grid


def smooth_grid(grid: np.ndarray) -> np.ndarray:
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float)
    k /= k.sum()
    g = np.pad(grid, ((1, 1), (1, 1)), mode="edge")
    out = (
        k[0, 0] * g[:-2, :-2]
        + k[0, 1] * g[:-2, 1:-1]
        + k[0, 2] * g[:-2, 2:]
        + k[1, 0] * g[1:-1, :-2]
        + k[1, 1] * g[1:-1, 1:-1]
        + k[1, 2] * g[1:-1, 2:]
        + k[2, 0] * g[2:, :-2]
        + k[2, 1] * g[2:, 1:-1]
        + k[2, 2] * g[2:, 2:]
    )
    return out.astype(grid.dtype)


def place_labels(ax, pts, codes, proj, custom_bias=None):
    """Small no-overlap label placer in lon/lat degrees."""
    dx, dy = 0.55, 0.33
    placed = []
    base_up = [(0, 0.88), (0.28, 0.88), (-0.28, 0.88), (0, 0), (0.28, 0), (-0.28, 0), (0, -0.88)]
    base_down = [(0, -0.88), (0.28, -0.88), (-0.28, -0.88), (0, 0), (0.28, 0), (-0.28, 0), (0, 0.88)]

    for (lon, lat), text in zip(pts, codes):
        pref = "up"
        if custom_bias and text in custom_bias:
            pref = custom_bias[text]
        offsets = base_up if pref == "up" else base_down

        chosen = None
        for ox, oy in offsets:
            xmin, xmax = lon + ox - dx / 2, lon + ox + dx / 2
            ymin, ymax = lat + oy - dy / 2, lat + oy + dy / 2
            rect = (xmin, xmax, ymin, ymax)
            if not any((xmin < Xmax and xmax > Xmin and ymin < Ymax and ymax > YMin) for (Xmin, Xmax, YMin, YMax) in placed):
                chosen = (ox, oy, rect)
                break

        if chosen is None:
            ox, oy = 0, 0.88 if pref == "up" else -0.88
            rect = (lon - dx / 2, lon + dx / 2, lat - dy / 2, lat + dy / 2)
        else:
            ox, oy, rect = chosen

        placed.append(rect)
        ax.text(
            lon + ox,
            lat + oy,
            text,
            fontsize=TEXT_FS,
            ha="center",
            va="center",
            transform=proj,
            color="k",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="w", alpha=0.9)],
        )


def plot_overlay_contours_with_stations(lons, lats, grid11, grid22, extent):
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9.2, 7.4), dpi=150)
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj)

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f1ec", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e6eef7", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.4, x_inline=False, y_inline=False)
    try:
        gl.top_labels = False
        gl.right_labels = False
    except Exception:
        pass

    # Underpaint
    if PAINT_ALPHA > 0:
        ax.contourf(lons, lats, grid11, levels=LEVELS, colors=[COLOR_2011], alpha=PAINT_ALPHA, transform=proj, zorder=1)
        ax.contourf(lons, lats, grid22, levels=LEVELS, colors=[COLOR_2022], alpha=PAINT_ALPHA, transform=proj, zorder=1)

    cs11 = ax.contour(lons, lats, grid11, levels=LEVELS, colors=COLOR_2011, linewidths=LW_2011, linestyles=LS_2011, transform=proj, zorder=2)
    ax.clabel(cs11, levels=LEVELS, fmt=lambda v: f"{int(v*100)}%", inline=True, fontsize=LABEL_FONTSIZE)

    cs22 = ax.contour(lons, lats, grid22, levels=LEVELS, colors=COLOR_2022, linewidths=LW_2022, linestyles=LS_2022, transform=proj, zorder=3)
    ax.clabel(cs22, levels=LEVELS, fmt=lambda v: f"{int(v*100)}%", inline=True, fontsize=LABEL_FONTSIZE)

    # AT138
    if AT138:
        lon = lon_from_e(AT138["lon_e"])
        lat = AT138["lat"]
        ax.scatter([lon], [lat], marker=MK_AT, s=SZ_AT, facecolor=FC_AT, edgecolor=EC_AT, linewidths=1.0, transform=proj, zorder=4)
        place_labels(ax, [(lon, lat)], [AT138["code"]], proj)

    # 2011-only
    pts_2011, codes_2011 = [], []
    for s in STATIONS_2011_ONLY:
        lon = lon_from_e(s["lon_e"])
        lat = s["lat"]
        ax.scatter([lon], [lat], marker=MK_2011, s=SZ_2011, facecolor=FC_2011, edgecolor=EC_2011, linewidths=1.0, transform=proj, zorder=4)
        pts_2011.append((lon, lat))
        codes_2011.append(s["code"])
    place_labels(ax, pts_2011, codes_2011, proj, custom_bias={"RL052": "down"})

    # 2022-only
    pts_2022, codes_2022 = [], []
    for s in STATIONS_2022_ONLY:
        lon = lon_from_e(s["lon_e"])
        lat = s["lat"]
        ax.scatter([lon], [lat], marker=MK_2022, s=SZ_2022, facecolor=FC_2022, edgecolor=EC_2022, linewidths=1.0, transform=proj, zorder=4)
        pts_2022.append((lon, lat))
        codes_2022.append(s["code"])
    place_labels(ax, pts_2022, codes_2022, proj)

    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_2011, lw=LW_2011, ls=LS_2011, label="2011-01-04"),
            Line2D([0], [0], color=COLOR_2022, lw=LW_2022, ls=LS_2022, label="2022-10-25"),
            Line2D([0], [0], marker=MK_2011, color=EC_2011, markerfacecolor=FC_2011, lw=0, label="2011 stations"),
            Line2D([0], [0], marker=MK_2022, color=EC_2022, markerfacecolor=FC_2022, lw=0, label="2022 stations"),
            Line2D([0], [0], marker=MK_AT, color=EC_AT, markerfacecolor=FC_AT, lw=0, label="AT138 (both)"),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.95,
    )

    plt.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extent", type=float, nargs=4, default=[-15, 45, 30, 72], help="lon_min lon_max lat_min lat_max")
    ap.add_argument("--dlon", type=float, default=0.5)
    ap.add_argument("--dlat", type=float, default=0.5)
    ap.add_argument("--dt-min", type=int, default=1, dest="dt_min", help="time step in minutes")
    ap.add_argument("--recompute", action="store_true", help="ignore cache and recompute grids")
    ap.add_argument("--max-workers", type=int, default=min(32, (os.cpu_count() or 4) * 2))
    ap.add_argument("--batch", type=int, default=600)
    ap.add_argument("--outdir", type=Path, default=Path("output"))
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    cache_dir = ensure_dir(outdir / "eclipse_cache")

    extent = tuple(args.extent)
    tag11, t011, t111 = EV_2011
    tag22, t022, t122 = EV_2022

    print("[2011] grid (cache+threads)…")
    lons, lats, grid11 = compute_obscuration_grid_threaded(
        extent, args.dlon, args.dlat, t011, t111, args.dt_min, tag11,
        cache_dir=cache_dir, recompute=args.recompute, max_workers=args.max_workers, batch=args.batch
    )

    print("[2022] grid (cache+threads)…")
    _, _, grid22 = compute_obscuration_grid_threaded(
        extent, args.dlon, args.dlat, t022, t122, args.dt_min, tag22,
        cache_dir=cache_dir, recompute=args.recompute, max_workers=args.max_workers, batch=args.batch
    )

    grid11 = smooth_grid(grid11)
    grid22 = smooth_grid(grid22)

    print("[plot] lines + stations…")
    fig = plot_overlay_contours_with_stations(lons, lats, grid11, grid22, extent)

    fig.savefig(outdir / "eclipse_overlay_lines_with_stations.png", dpi=240)
    fig.savefig(outdir / "eclipse_overlay_lines_with_stations.pdf")
    plt.close(fig)
    print("[OK] wrote outputs to:", outdir)


if __name__ == "__main__":
    main()
