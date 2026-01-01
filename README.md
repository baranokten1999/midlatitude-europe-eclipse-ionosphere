# Mid-Latitude European Ionospheric Response to Two Partial Solar Eclipses (2011 & 2022)

Reproducibility code + lightweight input tables for the manuscript:
**“Mid-Latitude European Ionospheric Response to Two Partial Solar Eclipses in Different Solar Cycles”**
(Advances in Space Research).

## What’s here
- `scripts/` — runnable scripts to reproduce the eclipse geometry plots, station multi-panels, heatmaps, and response metrics.
- `eclipse_iono/` — small helper package used by the scripts.
- `data/` — per-station `.dat` tables for each event date (three consecutive 24-hour blocks: D−1, D0, D+1).
- `output/` — generated figures and cache (created when you run the scripts).

## Data format (`data/<event-date>/<station>.dat`)
Each file is a **tab-separated** table with columns:

- `foF2` in **MHz**
- `hmF2` in **km**
- `TEC` in **TECU**

The file contains **three consecutive 24-hour blocks** (day-before, eclipse day, day-after),
with **no explicit separators**. Each block starts at `00:00`. For example, `00:00` appears
three times in the file.

See `data/README.md` for details and provenance notes.

## Quick start (pip)
From the repo root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run (recommended outdir so all figures land in one place):

```bash
python scripts/plot_obscuration_contours.py --outdir output/figures
python scripts/plot_station_panels.py --data-root data --outdir output/figures
python scripts/plot_heatmap_triplet.py --data-root data --outdir output/figures
python scripts/eclipse_metrics.py --data-root data --outdir output/figures
```

## Conda environment (optional)
```bash
conda env create -f environment.yml
conda activate eclipse_iono
```

## Outputs (created under `output/`)
- Eclipse obscuration overlay:
  - `output/figures/eclipse_overlay_lines_with_stations.(png|pdf)`
- Station multi-panel figures:
  - `output/figures/<STA>_<YYYY-MM-DD>_multi.(png|pdf)`
- Heatmaps (triplet):
  - `output/figures/heatmap_triplet_<YYYY-MM-DD>_nearest.(png|pdf)`
- Metrics figure:
  - `output/figures/eta_vs_emax.pdf`

## Citation
After you mint a DOI (Zenodo), cite the **version DOI** corresponding to the release used
to generate the results. Add it here:

- Zenodo DOI: `10.5281/zenodo.XXXXXXX` (placeholder)

A `CITATION.cff` file is included for GitHub/Zenodo citation metadata.

## License
- Code: MIT (see `LICENSE`)
- Figures/tables in `data/` and any generated outputs: treat as **CC BY 4.0** unless you
  replace them or specify otherwise in your publication workflow.
