# Standard Model Quickstart

Set up your environment, download smb-v1-1.7b, extract patient embeddings from synthetic data, and train classifiers.

Requires **Python 3.11+**.

## One-liner setup

```bash
bash -c "$(curl -fsSL https://docs.standardmodel.bio/quickstart.sh)"
```

This clones the repo, installs all locked dependencies via `uv sync`, and gets you ready to run.

## Manual setup

```bash
git clone https://github.com/standardmodelbio/quickstart.git
cd quickstart
uv sync
```

## Run the demo

```bash
cd quickstart
uv run demo.py
```

## Regenerate demo data (optional)

The demo fetches events and labels from GitHub (MIMIC-IV **demo** data—a reduced, publicly available subset, not the full MIMIC-IV database). To rebuild them locally, run the prep script. It downloads the [MIMIC-IV demo MEDS](https://physionet.org/content/mimic-iv-demo-meds/) to a temp dir (via wget), builds both parquets in `data/`, then removes the temp dir:

```bash
cd quickstart
uv run scripts/prep_mimic_demo_data.py
```

For reference, the script uses wget to fetch the data:

```bash
wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/
```

## Tests

```bash
uv run pytest tests/ -v
```

## Learn more

- [Synthetic data example](https://docs.standardmodel.bio/example)
- [Use your own data](https://docs.standardmodel.bio/your-own-data)
