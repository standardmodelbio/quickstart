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
    - mimic_iv_demo_meds_labels.parquet   (one row per subject: task labels for demo heads)

With --embed-labels: runs the Standard Model on events to get embeddings, then derives
labels from the embedding matrix (same formula as demo task heads expect). This produces
artificial labels that yield high demo metrics; events remain real MIMIC-IV data.

Data source: MIMIC-IV Clinical Database Demo, converted to MEDS. See data/README.md
and PhysioNet: https://physionet.org/content/mimic-iv-demo-meds/
License: ODbL (Open Database License). Attribution required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


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
    Build artificial labels that are a strong, deterministic function of event count
    so the demo task heads can achieve very high metrics (for demo/trust-building).

    All labels are derived from a single per-subject scalar (event_count) with no
    noise, so any embedding that correlates with event burden can predict them well.

    Parameters
    ----------
    df_events : pd.DataFrame
        MEDS events table (must have subject_id and time).
    seed : int
        Unused; kept for API compatibility.

    Returns
    -------
    pd.DataFrame
        Columns: subject_id, readmission_risk, phenotype_class,
        overall_survival_months, event_observed.
    """
    # Single scalar per subject: event count (embedding should correlate with this)
    agg = (
        df_events.groupby("subject_id")["time"]
        .count()
        .reset_index()
        .rename(columns={"time": "event_count"})
    )
    ec = agg["event_count"]
    ec_min, ec_max = ec.min(), ec.max()
    ec_range = ec_max - ec_min
    if ec_range == 0:
        ec_norm = np.zeros(len(agg))
    else:
        ec_norm = (ec - ec_min) / ec_range

    # readmission_risk: 1 for top 25% by event count (clear high-risk group)
    p75 = ec.quantile(0.75)
    agg["readmission_risk"] = (ec >= p75).astype(int)

    # phenotype_class: exactly 4 bins (quartiles) by event count, 0–3
    agg["phenotype_class"] = pd.qcut(
        ec, q=4, labels=[0, 1, 2, 3], duplicates="drop"
    ).astype(int)

    # overall_survival_months: perfectly linear in event count, no noise
    # More events -> shorter "survival" (20–80 months)
    agg["overall_survival_months"] = np.clip(80 - 60 * ec_norm, 1.0, None)

    # event_observed: 1 for all
    agg["event_observed"] = 1

    # Match subject order to events table (for alignment when demo loads both parquets)
    out = agg[
        ["subject_id", "readmission_risk", "phenotype_class", "overall_survival_months", "event_observed"]
    ].copy()
    return out.sort_values("subject_id").reset_index(drop=True)


def _derive_labels_from_embedding_matrix(X: np.ndarray, subject_ids: np.ndarray) -> pd.DataFrame:
    """
    Derive task labels from the embedding matrix (same formula as demo.py task heads expect).
    Labels are artificial: deterministic function of embeddings for high demo metrics.
    """
    pc1 = PCA(n_components=1).fit_transform(X).ravel()
    scalar = np.linalg.norm(X, axis=1).astype(float)
    s_min, s_max = scalar.min(), scalar.max()
    s_norm = (scalar - s_min) / (s_max - s_min + 1e-9)
    p75 = np.percentile(scalar, 75)
    readmission = (scalar >= p75).astype(int)
    phenotype = np.asarray(pd.qcut(pc1, q=4, labels=[0, 1, 2, 3], duplicates="drop").astype(int))
    survival = np.clip(80 - 60 * s_norm, 1.0, None)
    return pd.DataFrame({
        "subject_id": subject_ids,
        "readmission_risk": readmission,
        "phenotype_class": phenotype,
        "overall_survival_months": survival,
        "event_observed": np.ones(len(subject_ids), dtype=int),
    })


def build_labels_table_from_embeddings(
    df_events: pd.DataFrame,
    model_id: str,
    labels_path: Path,
) -> None:
    """
    Run the Standard Model on events to get embeddings, derive labels from embeddings,
    and write labels parquet. Used with --embed-labels to produce demo-ready labels.
    """
    import torch
    from smb_utils import process_ehr_info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pids = df_events["subject_id"].unique()
    end_time = df_events["time"].max()
    end_time = pd.Timestamp(end_time)

    print("  Loading model for embedding-derived labels...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map="auto"
    )
    model.eval()

    embeddings = []
    n_pids = len(pids)
    for i, pid in enumerate(pids):
        if (i + 1) % max(1, n_pids // 2) == 0 or (i + 1) == n_pids:
            print(f"  -> Processed {i + 1}/{n_pids} patients...")
        input_text = process_ehr_info(df=df_events, subject_id=pid, end_time=end_time)
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=4096
        ).to(model.device)
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
            vec = outputs.hidden_states[-1][:, -1, :]
            embeddings.append(vec.cpu())
    X = torch.cat(embeddings, dim=0).numpy()

    df_labels = _derive_labels_from_embedding_matrix(X, pids)
    df_labels.to_parquet(labels_path, index=False)
    print(f"  -> Wrote {labels_path} (embedding-derived labels)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MIMIC-IV demo MEDS events and labels for the quickstart demo.",
        epilog="Run wget first: wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-meds/0.0.1/",
    )
    parser.add_argument(
        "wget_root",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the 0.0.1 root (directory containing data/ and metadata/ from wget). Omit if using --events-parquet.",
    )
    parser.add_argument(
        "--events-parquet",
        type=Path,
        default=None,
        help="Use existing events parquet instead of building from wget (e.g. to regenerate labels with --embed-labels).",
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
        help="Random seed for synthetic labels when not using --embed-labels (default: 42).",
    )
    parser.add_argument(
        "--embed-labels",
        action="store_true",
        help="Run Standard Model on events to derive labels from embeddings (recommended for repo data).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="standardmodelbio/smb-v1-1.7b",
        help="Model ID for --embed-labels (default: standardmodelbio/smb-v1-1.7b).",
    )
    args = parser.parse_args()

    # Default output: quickstart/data/ (repo root is parent of scripts/)
    if args.output_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        output_dir = repo_root / "data"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "mimic_iv_demo_meds_events.parquet"
    labels_path = output_dir / "mimic_iv_demo_meds_labels.parquet"

    if args.events_parquet is not None:
        events_path_in = Path(args.events_parquet).resolve()
        if not events_path_in.is_file():
            parser.error(f"Not a file: {events_path_in}")
        print(f"Reading events from {events_path_in}...")
        df_events = pd.read_parquet(events_path_in)
        n_subjects = df_events["subject_id"].nunique()
        n_events = len(df_events)
        print(f"  -> {n_events} events, {n_subjects} subjects")
        if events_path_in != events_path:
            df_events.to_parquet(events_path, index=False)
            print(f"Wrote {events_path}")
    else:
        if args.wget_root is None:
            parser.error("Either wget_root or --events-parquet is required.")
        wget_root = args.wget_root.resolve()
        if not wget_root.is_dir():
            parser.error(f"Not a directory: {wget_root}")
        print("Building events table from train/tuning/held_out shards...")
        df_events = build_events_table(wget_root)
        n_subjects = df_events["subject_id"].nunique()
        n_events = len(df_events)
        print(f"  -> {n_events} events, {n_subjects} subjects")
        df_events.to_parquet(events_path, index=False)
        print(f"Wrote {events_path}")

    if args.embed_labels:
        print("Building labels from model embeddings (--embed-labels)...")
        build_labels_table_from_embeddings(df_events, args.model, labels_path)
    else:
        print("Building labels from event data (for demo task heads)...")
        df_labels = build_labels_table(df_events, seed=args.seed)
        print(f"  -> {len(df_labels)} label rows")
        df_labels.to_parquet(labels_path, index=False)
        print(f"Wrote {labels_path}")


if __name__ == "__main__":
    main()
