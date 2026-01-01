\
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")


def sniff_header_and_sep(path: Path) -> Tuple[Optional[List[str]], str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    first = txt.splitlines()[0].strip() if txt else ""
    sep = "\t" if ("\t" in first) else r"\s+"
    if re.search(r"[A-Za-z]", first):
        hdr = re.split(r"\t| +", first)
        hdr = [h.strip() for h in hdr if h.strip()]
        return hdr, sep
    return None, sep


def canonicalize_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        k = c.strip()
        kl = k.lower().replace(" ", "")
        out.append(
            {
                "time": "time",
                "fof2": "fof2",
                "hmf2": "hmf2",
                "tec": "tec",
                "nmf2": "NmF2",
            }.get(kl, k)
        )
    return out


def to_minutes(hhmm: str) -> Optional[int]:
    if not isinstance(hhmm, str):
        return None
    m = _TIME_RE.match(hhmm)
    if not m:
        return None
    h = int(m.group(1))
    mi = int(m.group(2))
    ss = int(m.group(3) or 0)
    return h * 60 + mi + (1 if ss >= 30 else 0)


def split_three_blocks(mins: np.ndarray) -> List[int]:
    mins = np.array([m if m is not None else np.nan for m in mins], dtype="float64")
    cut = [0]
    for i in range(1, len(mins)):
        if np.isnan(mins[i]) or np.isnan(mins[i - 1]):
            cut.append(i)
        elif mins[i] < mins[i - 1] - 30:
            cut.append(i)
    cut = sorted(set([c for c in cut if 0 <= c <= len(mins)]))
    cut.append(len(mins))
    if len(cut) == 4:
        return cut
    if len(cut) > 4:
        spans = [(cut[j], cut[j + 1]) for j in range(len(cut) - 1)]
        spans = sorted(spans, key=lambda ab: (ab[1] - ab[0]), reverse=True)[:3]
        pts = sorted([s[0] for s in spans] + [spans[-1][1]])
        if pts[0] != 0:
            pts = [0] + pts
        if pts[-1] != len(mins):
            pts[-1] = len(mins)
        return [pts[0], pts[1], pts[2], pts[3]]
    n = len(mins)
    return [0, n // 3, 2 * n // 3, n]


def assign_utc_index(base_date: pd.Timestamp, times: pd.Series) -> pd.DatetimeIndex:
    mins = times.astype(str).map(to_minutes).to_numpy()
    i0, i1, i2, i3 = split_three_blocks(mins)
    anchors = [
        (base_date - pd.Timedelta(days=1)).tz_convert("UTC"),
        base_date.tz_convert("UTC"),
        (base_date + pd.Timedelta(days=1)).tz_convert("UTC"),
    ]
    parts: List[pd.DatetimeIndex] = []
    for k, (s, e) in enumerate([(i0, i1), (i1, i2), (i2, i3)]):
        hhmm = times.iloc[s:e].astype(str)
        ts = pd.to_datetime(
            anchors[k].strftime("%Y-%m-%d") + " " + hhmm,
            utc=True,
            errors="coerce",
        )
        parts.append(ts)
    idx = pd.DatetimeIndex(list(parts[0]) + list(parts[1]) + list(parts[2]))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx


def parse_dat_one(path: Path, base_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    names, sep = sniff_header_and_sep(path)
    try:
        df = pd.read_csv(
            path,
            sep=sep,
            engine="python",
            header=0 if names is not None else None,
            names=None if names is not None else ["time", "fof2", "hmf2", "tec"],
            comment="#",
            on_bad_lines="skip",
            dtype=str,
        )
    except Exception as e:
        print(f"[WARN] {path.name}: read error: {e}")
        return None

    df.columns = canonicalize_cols(list(df.columns))
    if "time" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time"})

    keep = ["time"] + [c for c in ["fof2", "hmf2", "tec", "NmF2"] if c in df.columns]
    df = df[keep].copy()

    for c in ["fof2", "hmf2", "tec", "NmF2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure both fof2 and NmF2 exist if either does (NmF2 [m^-3] from foF2 [MHz])
    if "fof2" in df.columns and "NmF2" not in df.columns:
        df["NmF2"] = 1.24e10 * (df["fof2"] ** 2)
    if "NmF2" in df.columns and "fof2" not in df.columns:
        df["fof2"] = np.sqrt(df["NmF2"] / 1.24e10)

    idx = assign_utc_index(base_date, df["time"])
    df = df.assign(time=idx).dropna(subset=["time"]).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def three_day_frames(df_raw: pd.DataFrame, base: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d_1_start = (base - pd.Timedelta(days=1)).tz_convert("UTC")
    d0_start = base.tz_convert("UTC")
    d1_start = (base + pd.Timedelta(days=1)).tz_convert("UTC")
    d2_start = (base + pd.Timedelta(days=2)).tz_convert("UTC")

    df = df_raw.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    m_minus = (df.index >= d_1_start) & (df.index < d0_start)
    m_d0 = (df.index >= d0_start) & (df.index < d1_start)
    m_plus = (df.index >= d1_start) & (df.index < d2_start)

    return df[m_minus], df[m_d0], df[m_plus]


def minute_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.hour * 60 + idx.minute).astype(int)


def quiet_proxy_from_neighbors(dminus: pd.DataFrame, dplus: pd.DataFrame, d0_index: pd.DatetimeIndex, var: str) -> pd.Series:
    """Quiet proxy = median of D-1 and D+1 at the same minute-of-day."""
    mo = minute_of_day(d0_index)
    res = pd.Series(index=d0_index, dtype=float)

    def build_map(df: pd.DataFrame) -> Dict[int, float]:
        if var not in df.columns:
            return {}
        s = df[var].copy()
        s = s[~s.index.duplicated(keep="first")]
        return dict(zip(minute_of_day(s.index), s.values))

    m1 = build_map(dminus)
    m2 = build_map(dplus)

    vals = []
    for m in mo:
        vs = []
        if m in m1 and np.isfinite(m1[m]):
            vs.append(float(m1[m]))
        if m in m2 and np.isfinite(m2[m]):
            vs.append(float(m2[m]))
        vals.append(float(np.median(vs)) if vs else np.nan)

    res[:] = vals
    return res


def delta_percent(y: pd.Series, yq: pd.Series) -> pd.Series:
    out = (y - yq) / yq * 100.0
    out[(~np.isfinite(yq)) | (yq == 0)] = np.nan
    return out
