# Demo data (MIMIC-IV in MEDS format)

This directory holds the MEDS events and labels used by the quickstart demo (`demo.py`). The script fetches these files from the GitHub repo at runtime and loads them into memory; they are also committed here for versioning and attribution.

## Files

- **mimic_iv_demo_meds_events.parquet** — MEDS event stream: `subject_id`, `time`, `code`, `table`, `value`. One row per clinical event (100 subjects total, from the MIMIC-IV demo).
- **mimic_iv_demo_meds_labels.parquet** — One row per subject with **artificially generated** task labels (from the prep script; hardcoded in the prep script; generated once from model embeddings): `readmission_risk`, `phenotype_class`, `overall_survival_months`, `event_observed`. Used by the demo’s four task heads; labels are for demonstration only and have no clinical meaning.

## Source and license

- **Source:** [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/) (100 de-identified patients), converted to the [Medical Event Data Standard (MEDS)](https://github.com/Medical-Event-Data-Standard/meds) and published as [MIMIC-IV demo data in MEDS](https://physionet.org/content/mimic-iv-demo-meds/) (van de Water et al., PhysioNet, 2025).
- **License:** [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1.0/). You may share and adapt the database with attribution and share-alike.
- **Attribution:** When using this data, please cite the MIMIC-IV demo and the MEDS conversion as above and include the ODbL notice.

## Reproducing the data

To regenerate the two parquets from the original PhysioNet release:

1. Download the MIMIC-IV demo MEDS mirror:
   ```bash
   wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/
   ```
2. From the quickstart repo root, run the prep script with the path to the `0.0.1` directory (the one that contains `data/` and `metadata/`):
   ```bash
   uv run scripts/prep_mimic_demo_data.py /path/to/physionet.org/files/mimic-iv-demo-meds/0.0.1
   ```
   Events are built from the wget mirror; labels are looked up from a hardcoded table (no model run). Output is written to `data/` by default.
