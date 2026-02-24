#!/usr/bin/env python3
"""
Prepare MIMIC-IV demo MEDS data for the quickstart demo.

Reads the raw output of a wget mirror of PhysioNet's MIMIC-IV demo in MEDS format,
concatenates the train/tuning/held_out shards, aligns columns to what demo.py and
smb_utils.process_ehr_info expect, builds synthetic labels (for the four demo task
heads), and writes two parquet files into the repo's data/ directory.

Usage:
    # First, download the MIMIC-IV demo MEDS data (run from any directory):
    #   wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/
    #
    # Then run this script with the path to the 0.0.1 root (the directory that
    # contains "data/" and "metadata/"):
    uv run python scripts/prep_mimic_demo_data.py /path/to/physionet.org/files/mimic-iv-demo-meds/0.0.1
    #
    # Or with explicit output directory:
    uv run python scripts/prep_mimic_demo_data.py /path/to/0.0.1 --output-dir ./data

Output files (written to --output-dir, default repo data/):
    - mimic_iv_demo_meds_events.parquet   (MEDS events: subject_id, time, code, table, value)
    - mimic_iv_demo_meds_labels.parquet    (one row per subject: task labels for demo heads)

Data source: MIMIC-IV Clinical Database Demo, converted to MEDS. See data/README.md
and PhysioNet: https://physionet.org/content/mimic-iv-demo-meds/
License: ODbL (Open Database License). Attribution required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Code prefix -> modality (table) for MEDS serialization.
# MIMIC-IV MEDS parquets do not include a "table" column; we infer from code.
# -----------------------------------------------------------------------------
CODE_PREFIX_TO_TABLE = {
    "ICD10:": "condition",
    "ICD9:": "condition",
    "RxNorm:": "medication",
    "LOINC:": "lab",
    "CPT:": "procedure",
    "HCPCS:": "procedure",
    "NDC:": "medication",
    "SNOMED:": "condition",
}


def _infer_table_from_code(code: str) -> str:
    """
    Infer MEDS modality (table) from the code string.

    PhysioNet MEDS uses prefixes like ICD10:, RxNorm:, LOINC:. We map these
    to the table names expected by smb_utils (condition, medication, lab, procedure).
    Unknown prefixes default to "condition".
    """
    if not isinstance(code, str):
        return "condition"
    for prefix, table in CODE_PREFIX_TO_TABLE.items():
        if code.startswith(prefix):
            return table
    return "condition"


def build_events_table(wget_root: Path) -> pd.DataFrame:
    """
    Build a single MEDS events DataFrame from the wget mirror.

    Reads data/train/0.parquet, data/tuning/0.parquet, and data/held_out/0.parquet,
    concatenates them, and aligns columns to demo.py / process_ehr_info expectations:
    subject_id, time, code, table, value.

    Parameters
    ----------
    wget_root : Path
        Path to the 0.0.1 root (directory containing "data/" and "metadata/").

    Returns
    -------
    pd.DataFrame
        Sorted by subject_id, time. Columns: subject_id, time, code, table, value.
    """
    data_dir = wget_root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Expected directory: {data_dir}")

    shards = []
    for split in ("train", "tuning", "held_out"):
        path = data_dir / split / "0.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Expected parquet: {path}")
        shards.append(pd.read_parquet(path))

    # Concatenate all shards
    df = pd.concat(shards, ignore_index=True)

    # Required MEDS columns from PhysioNet
    if not {"subject_id", "time", "code"}.issubset(df.columns):
        raise ValueError(
            f"Expected columns subject_id, time, code. Got: {list(df.columns)}"
        )

    # Map numeric_value -> value (demo.py and smb_utils use "value")
    if "numeric_value" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"numeric_value": "value"})
    elif "value" not in df.columns:
        df["value"] = None

    # Add table (modality) inferred from code prefix
    df["table"] = df["code"].map(_infer_table_from_code)

    # Keep only columns needed for the demo
    df = df[["subject_id", "time", "code", "table", "value"]].copy()
    df = df.sort_values(["subject_id", "time"]).reset_index(drop=True)

    return df


def build_labels_table(df_events: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Build labels for the four demo task heads by deriving them from the event data.

    MIMIC-IV demo MEDS does not include outcome labels. We derive labels from
    each subject's event stream (event count, length of stay) so there is a
    learnable signal: the model's embeddings summarize the same events, so they
    can predict these derived outcomes. Labels are deterministic from the data
    (seed only used for optional small noise). For demonstration only; not
    clinically validated outcomes.

    Parameters
    ----------
    df_events : pd.DataFrame
        MEDS events table (must have subject_id and time).
    seed : int
        Random seed for optional noise (e.g. in survival months).

    Returns
    -------
    pd.DataFrame
        Columns: subject_id, readmission_risk, phenotype_class,
        overall_survival_months, event_observed.
    """
    # Per-subject summaries from the event stream (signal the model can learn)
    agg = df_events.groupby("subject_id").agg(
        event_count=("time", "count"),
        los_days=("time", lambda s: (s.max() - s.min()).total_seconds() / 86400),
    ).reset_index()

    rng = np.random.default_rng(seed)

    # readmission_risk: 1 if above median event count (more events = higher risk)
    median_events = agg["event_count"].median()
    agg["readmission_risk"] = (agg["event_count"] >= median_events).astype(int)

    # phenotype_class: quartile of event count (0–3), ordinal signal
    agg["phenotype_class"] = pd.qcut(
        agg["event_count"], q=4, labels=[0, 1, 2, 3], duplicates="drop"
    ).astype(int)

    # overall_survival_months: inverse to event burden (more events → shorter survival)
    # Scale to a plausible range and add small noise so it's not perfectly deterministic
    event_pct = (agg["event_count"] - agg["event_count"].min()) / (
        agg["event_count"].max() - agg["event_count"].min() + 1e-9
    )
    base_survival = 60 - event_pct * 45  # roughly 15–60 months
    noise = rng.normal(0, 2, size=len(agg))
    agg["overall_survival_months"] = np.clip(base_survival + noise, 1.0, None)

    # event_observed: 1 for all (no censoring in this demo)
    agg["event_observed"] = 1

    return agg[
        [
            "subject_id",
            "readmission_risk",
            "phenotype_class",
            "overall_survival_months",
            "event_observed",
        ]
    ].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MIMIC-IV demo MEDS events and labels for the quickstart demo.",
        epilog="Run wget first: wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/",
    )
    parser.add_argument(
        "wget_root",
        type=Path,
        help="Path to the 0.0.1 root (directory containing data/ and metadata/ from wget).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write output parquets. Default: repo data/ next to this script.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic labels (default: 42).",
    )
    args = parser.parse_args()

    wget_root = args.wget_root.resolve()
    if not wget_root.is_dir():
        parser.error(f"Not a directory: {wget_root}")

    # Default output: quickstart/data/ (repo root is parent of scripts/)
    if args.output_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        output_dir = repo_root / "data"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build events table from the three shards
    print("Building events table from train/tuning/held_out shards...")
    df_events = build_events_table(wget_root)
    n_subjects = df_events["subject_id"].nunique()
    n_events = len(df_events)
    print(f"  -> {n_events} events, {n_subjects} subjects")

    # Build labels derived from event data (so the model has a learnable signal)
    print("Building labels from event data (for demo task heads)...")
    df_labels = build_labels_table(df_events, seed=args.seed)
    print(f"  -> {len(df_labels)} label rows")

    # Write parquets
    events_path = output_dir / "mimic_iv_demo_meds_events.parquet"
    labels_path = output_dir / "mimic_iv_demo_meds_labels.parquet"
    df_events.to_parquet(events_path, index=False)
    df_labels.to_parquet(labels_path, index=False)
    print(f"Wrote {events_path}")
    print(f"Wrote {labels_path}")


if __name__ == "__main__":
    main()
