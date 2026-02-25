# Demo data (MIMIC-IV demo in MEDS format)

This directory holds the MEDS events and labels used by the quickstart demo (`demo.py`). All data is from the **MIMIC-IV demo** (a reduced, publicly available subset for teaching and development), not the full MIMIC-IV database. The script fetches these files from the GitHub repo at runtime and loads them into memory; they are also committed here for versioning and attribution.

## Files

- **mimic_iv_demo_meds_events.parquet** — MEDS event stream: `subject_id`, `time`, `code`, `table`, `value`. One row per clinical event (100 subjects total, from the MIMIC-IV demo).
- **mimic_iv_demo_meds_labels.parquet** — One row per subject with **artificially generated** task labels (from the prep script; hardcoded in the prep script; generated once from model embeddings): `subject_id`, `prediction_time` (single as-of time for all, from events), `readmission_risk`, `phenotype_class`, `overall_survival_months`, `event_observed`. The demo uses `prediction_time` from this file to define the cutoff for embedding extraction (MEDS-style: labels file is source of truth). Used by the demo’s four task heads; labels are for demonstration only and have no clinical meaning.

## Source and license

- **Source:** [MIMIC-IV Clinical Database **Demo**](https://physionet.org/content/mimic-iv-demo/) (100 de-identified patients; reduced subset for teaching/dev), converted to the [Medical Event Data Standard (MEDS)](https://github.com/Medical-Event-Data-Standard/meds) and published as [MIMIC-IV demo data in MEDS](https://physionet.org/content/mimic-iv-demo-meds/) (van de Water et al., PhysioNet, 2025). Full citation: van de Water et al. (2025). [MIMIC-IV demo data in the Medical Event Data Standard (MEDS) (version 0.0.1)](https://doi.org/10.13026/t2y8-ea41). PhysioNet. RRID:SCR_007345.
- **License:** The data in this directory (events and labels) is offered under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1.0/) in compliance with the upstream dataset’s license. You may share and adapt it with attribution and share-alike (any derivative you publish must also be under ODbL).
- **Attribution:** When using this data, cite the MIMIC-IV demo and the MEDS conversion as above, include the ODbL notice, and do not apply technical measures that restrict reuse.

## Reproducing the data

From the quickstart repo root, run the prep script. It downloads the MIMIC-IV demo MEDS release to a temporary directory via wget, builds the events table and labels, writes both parquets to `data/`, then removes the temp directory:

```bash
uv run scripts/prep_mimic_demo_data.py
```

For reference, the script runs the following to fetch the data (so you know what’s happening under the hood):

```bash
wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/
```

Labels are looked up from a hardcoded table (no model run). Output is written to `data/` by default; use `--output-dir` to override.
