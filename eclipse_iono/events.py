\
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class StationEvent:
    code: str
    name: str
    lat_deg: float
    lon_deg_east: float
    t_start: pd.Timestamp
    t_great: pd.Timestamp
    t_end: pd.Timestamp
    peak_mag: float


def _ts(date_str: str, time_str: str) -> pd.Timestamp:
    return pd.to_datetime(f"{date_str} {time_str}", format="%Y-%m-%d %H:%M:%S", utc=True)


EVENTS_2011: List[StationEvent] = [
    StationEvent("AT138", "ATHENS (Greece)", 38.00, 23.50,
                 _ts("2011-01-04", "06:57:30"), _ts("2011-01-04", "08:23:33"), _ts("2011-01-04", "09:58:16"), 0.6747),
    StationEvent("DB049", "DOURBES (Belgium)", 50.10, 4.60,
                 _ts("2011-01-04", "07:44:38"), _ts("2011-01-04", "08:14:35"), _ts("2011-01-04", "09:36:26"), 0.7627),
    StationEvent("RL052", "CHILTON (England)", 51.50, 359.40,
                 _ts("2011-01-04", "08:05:42"), _ts("2011-01-04", "08:11:45"), _ts("2011-01-04", "09:30:58"), 0.7480),
    StationEvent("EB040", "ROQUETES (Spain)", 40.80, 0.50,
                 _ts("2011-01-04", "07:17:34"), _ts("2011-01-04", "07:57:48"), _ts("2011-01-04", "09:17:36"), 0.6331),
    StationEvent("MO155", "ELEKTROUGLI (Russia)", 55.76, 38.28,
                 _ts("2011-01-04", "07:38:08"), _ts("2011-01-04", "09:03:48"), _ts("2011-01-04", "10:29:53"), 0.8124),
]

EVENTS_2022: List[StationEvent] = [
    StationEvent("AT138", "ATHENS (Greece)", 38.00, 23.50,
                 _ts("2022-10-25", "09:36:24"), _ts("2022-10-25", "10:43:58"), _ts("2022-10-25", "11:51:09"), 0.3778),
    StationEvent("FF051", "FAIRFORD (England)", 51.70, 358.50,
                 _ts("2022-10-25", "09:08:40"), _ts("2022-10-25", "09:59:12"), _ts("2022-10-25", "10:51:27"), 0.2593),
    StationEvent("VT139", "SAN VITO (Italy)", 40.60, 17.80,
                 _ts("2022-10-25", "09:27:53"), _ts("2022-10-25", "10:33:06"), _ts("2022-10-25", "11:38:00"), 0.3673),
    StationEvent("RO041", "ROME (Italy)", 41.90, 12.50,
                 _ts("2022-10-25", "09:25:34"), _ts("2022-10-25", "10:21:44"), _ts("2022-10-25", "11:19:05"), 0.2650),
]

EVENT_META: Dict[Tuple[str, str], StationEvent] = {}
for e in EVENTS_2011:
    EVENT_META[("2011-01-04", e.code.upper())] = e
for e in EVENTS_2022:
    EVENT_META[("2022-10-25", e.code.upper())] = e
