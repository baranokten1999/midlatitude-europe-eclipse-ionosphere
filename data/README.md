# Input tables in `data/`

This repository includes per-station `.dat` tables for two event dates:

- `data/2011-01-04/` (stations: AT138, DB049, EB040, MO155, RL052)
- `data/2022-10-25/` (stations: AT138, FF051, RO041, VT139)

## File structure

Each `<station>.dat` file is **tab-separated** with columns:

- `foF2` (MHz)
- `hmF2` (km)
- `TEC` (TECU)

The rows contain **three consecutive 24-hour blocks** (D−1, D0, D+1), with no explicit separators.
Blocks are concatenated back-to-back. Each block begins at `00:00`.

Cadence depends on station (typically 10 or 15 minutes):
- 15-min cadence: 96 rows per day → 288 rows total (+ header)
- 10-min cadence: 144 rows per day → 432 rows total (+ header)

Missing values may appear as blanks or NaNs and are handled by the analysis code.

## Provenance / redistribution note

The underlying ionosonde parameters (foF2, hmF2, ionosonde-derived TEC) originate from the
GIRO Digisonde network. These tables are lightweight, analysis-ready representations used
to reproduce the figures/metrics in the associated manuscript.

If you replace these files with your own downloads, keep the same column order and the same
three-block (D−1, D0, D+1) convention.
