# Standard Model Quickstart

Set up your environment, download SMB-v1-1.7B, extract patient embeddings from dummy data, and train classifiers.

## Setup

From the repo root, run:

```bash
./quickstart/quickstart.sh
```

Then activate and run the demo:

```bash
source standard_model/bin/activate
cd quickstart && python demo.py
```

Alternatively, with [uv](https://docs.astral.sh/uv/):

```bash
cd quickstart
uv sync --extra dev
uv pip install "git+https://github.com/standardmodelbio/smb-utils.git"
python demo.py
```

## Tests

```bash
cd quickstart && uv run pytest tests/ -v
```
