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

## Tests

```bash
uv run pytest tests/ -v
```

## Learn more

- [Synthetic data example](https://docs.standardmodel.bio/example)
- [Use your own data](https://docs.standardmodel.bio/your-own-data)
