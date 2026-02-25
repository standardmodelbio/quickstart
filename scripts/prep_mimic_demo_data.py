#!/usr/bin/env python3
"""
Prepare MIMIC-IV demo MEDS data for the quickstart demo.

From the quickstart repo root, run:

   uv run scripts/prep_mimic_demo_data.py

The script downloads the MIMIC-IV demo MEDS data to a temporary directory via wget,
builds the events table from the train/tuning/held_out shards, assigns labels from
a hardcoded table, writes both parquets to data/, then removes the temp download.

Output files (in data/ or --output-dir):
    - mimic_iv_demo_meds_events.parquet   (MEDS events: subject_id, time, code, table, value)
    - mimic_iv_demo_meds_labels.parquet   (one row per subject: artificial task labels)

Data source: MIMIC-IV Clinical Database Demo, converted to MEDS. See data/README.md
and PhysioNet: https://physionet.org/content/mimic-iv-demo-meds/
License: ODbL (Open Database License). Attribution required.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

# PhysioNet MIMIC-IV demo MEDS 0.0.1 (used when running without a path)
MIMIC_DEMO_MEDS_URL = "https://physionet.org/files/mimic-iv-demo-meds/0.0.1/"


# Hardcoded labels for the MIMIC-IV demo 100 subjects (generated once from model embeddings).
# (subject_id, readmission_risk, phenotype_class, overall_survival_months, event_observed)
DEMO_LABELS_ROWS = [
    (10000032, 0, 0, 68.12092113903353, 1),
    (10001217, 0, 2, 45.84275258400712, 1),
    (10001725, 0, 1, 70.46466936959183, 1),
    (10002428, 1, 1, 35.48944286799398, 1),
    (10002495, 0, 0, 71.2916696513238, 1),
    (10002930, 0, 1, 71.88616265586484, 1),
    (10003046, 1, 2, 36.53097898175887, 1),
    (10003400, 0, 2, 39.476680576873704, 1),
    (10004235, 1, 3, 35.567637482699155, 1),
    (10004422, 0, 3, 38.90275677269726, 1),
    (10004457, 1, 2, 30.169092853288134, 1),
    (10004720, 0, 0, 78.74232628707377, 1),
    (10004733, 0, 2, 39.816697263080876, 1),
    (10005348, 0, 0, 38.36617842834964, 1),
    (10005817, 0, 2, 39.49210884151437, 1),
    (10005866, 0, 1, 47.580410688566516, 1),
    (10005909, 0, 1, 60.8869642217438, 1),
    (10006053, 1, 3, 20.000000000715318, 1),
    (10006580, 0, 3, 36.86786853827411, 1),
    (10007058, 0, 2, 43.49982297378702, 1),
    (10007795, 0, 2, 36.789646636224326, 1),
    (10007818, 0, 0, 66.80883373222693, 1),
    (10007928, 0, 0, 75.51072153884145, 1),
    (10008287, 0, 3, 39.05202946263416, 1),
    (10008454, 0, 1, 69.86674634583373, 1),
    (10009035, 0, 1, 69.49218661008884, 1),
    (10009049, 0, 0, 70.99168077092031, 1),
    (10009628, 0, 3, 37.89701619758753, 1),
    (10010471, 1, 2, 33.55433353785996, 1),
    (10010867, 1, 3, 22.23345278437305, 1),
    (10011398, 0, 2, 42.475499717000076, 1),
    (10012552, 0, 0, 77.45577983481911, 1),
    (10012853, 0, 1, 38.46481126416529, 1),
    (10013049, 1, 3, 35.70748512380887, 1),
    (10014078, 0, 2, 36.684901435216766, 1),
    (10014354, 1, 2, 29.7432683832305, 1),
    (10014729, 0, 3, 37.19190029801064, 1),
    (10015272, 0, 1, 46.539289342439645, 1),
    (10015860, 0, 0, 70.02830106957813, 1),
    (10015931, 1, 1, 32.836632714950426, 1),
    (10016150, 1, 0, 34.16606668672355, 1),
    (10016742, 0, 0, 71.63044476335136, 1),
    (10016810, 0, 1, 69.36390607435793, 1),
    (10017492, 0, 0, 75.15849649465721, 1),
    (10018081, 1, 3, 35.545070848709344, 1),
    (10018328, 0, 1, 68.1349823077094, 1),
    (10018423, 0, 0, 71.07376383223195, 1),
    (10018501, 0, 3, 37.32594119218803, 1),
    (10018845, 0, 2, 38.91099209329958, 1),
    (10019003, 1, 3, 34.02520395639446, 1),
    (10019172, 0, 1, 64.71091173176212, 1),
    (10019385, 0, 0, 71.07271326946459, 1),
    (10019568, 0, 0, 73.32838340910115, 1),
    (10019777, 0, 0, 75.92859447668069, 1),
    (10019917, 0, 2, 42.11862490929484, 1),
    (10020187, 1, 3, 33.643896060328316, 1),
    (10020306, 0, 1, 45.34896079600293, 1),
    (10020640, 0, 1, 44.16333658777643, 1),
    (10020740, 1, 3, 35.55097583008225, 1),
    (10020786, 0, 3, 41.04753660132081, 1),
    (10020944, 0, 0, 79.25483992221267, 1),
    (10021118, 1, 3, 35.05483187297296, 1),
    (10021312, 0, 3, 36.5672766075548, 1),
    (10021487, 0, 0, 71.8023168320916, 1),
    (10021666, 0, 2, 39.99605152171228, 1),
    (10021938, 0, 1, 69.47398867997057, 1),
    (10022017, 0, 1, 50.00430730770384, 1),
    (10022041, 0, 3, 39.74345803015629, 1),
    (10022281, 0, 2, 36.673096929939874, 1),
    (10022880, 0, 1, 68.04595461719533, 1),
    (10023117, 0, 0, 51.67532425926457, 1),
    (10023239, 0, 0, 75.33677836064558, 1),
    (10023771, 1, 2, 35.133599521914874, 1),
    (10024043, 1, 3, 35.900002797478585, 1),
    (10025463, 0, 0, 79.36759941633125, 1),
    (10025612, 1, 3, 32.51657399258865, 1),
    (10026255, 1, 3, 30.94483385817628, 1),
    (10026354, 1, 3, 35.85914818513332, 1),
    (10026406, 1, 1, 30.15575479924434, 1),
    (10027445, 1, 3, 29.45499395986669, 1),
    (10027602, 1, 2, 33.62456024793996, 1),
    (10029291, 0, 0, 80.0, 1),
    (10029484, 0, 2, 38.272937571828294, 1),
    (10031404, 0, 1, 66.58064331295277, 1),
    (10031757, 0, 2, 45.53240815632536, 1),
    (10032725, 0, 1, 38.06147156606364, 1),
    (10035185, 0, 3, 36.93309074935323, 1),
    (10035631, 0, 1, 70.77456628482204, 1),
    (10036156, 0, 0, 73.6062040507461, 1),
    (10037861, 0, 1, 39.62614973569177, 1),
    (10037928, 0, 2, 41.066675944828, 1),
    (10037975, 0, 2, 45.599213033391656, 1),
    (10038081, 0, 2, 42.70170088885239, 1),
    (10038933, 0, 2, 44.355352174305374, 1),
    (10038992, 0, 1, 36.639206047938266, 1),
    (10038999, 0, 1, 66.30373679735214, 1),
    (10039708, 0, 0, 73.20218216902927, 1),
    (10039831, 0, 0, 71.94050267391476, 1),
    (10039997, 0, 2, 40.87782569027343, 1),
    (10040025, 1, 3, 36.274199611540205, 1),
]


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


def build_labels_table(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Build labels by looking up each subject in the events table in the hardcoded
    DEMO_LABELS_ROWS (MIMIC-IV demo 100 subjects). Raises if any subject is missing.
    Uses a single prediction_time for all rows (max event time) so the labels file
    is the source of truth for when predictions are made (MEDS-style).
    """
    prediction_time = pd.Timestamp(df_events["time"].max())
    by_subject = {
        row[0]: {
            "readmission_risk": row[1],
            "phenotype_class": row[2],
            "overall_survival_months": row[3],
            "event_observed": row[4],
        }
        for row in DEMO_LABELS_ROWS
    }
    subject_ids = df_events["subject_id"].unique()
    missing = set(subject_ids) - set(by_subject)
    if missing:
        raise ValueError(
            f"Labels are only defined for the standard MIMIC-IV demo 100 subjects. "
            f"Found {len(missing)} subject(s) not in the table: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )
    rows = []
    for sid in subject_ids:
        r = by_subject[sid].copy()
        r["subject_id"] = sid
        r["prediction_time"] = prediction_time
        rows.append(r)
    return pd.DataFrame(rows)[
        [
            "subject_id",
            "prediction_time",
            "readmission_risk",
            "phenotype_class",
            "overall_survival_months",
            "event_observed",
        ]
    ]


def _download_mimic_demo_to_temp() -> tuple[Path, Path]:
    """
    Run wget to mirror MIMIC-IV demo MEDS 0.0.1 into a temp directory.
    Returns (temp_dir, wget_root) where wget_root is the 0.0.1 root under temp_dir.
    Caller must shutil.rmtree(temp_dir) when done.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="mimic_demo_meds_"))
    # wget -r -N -c -np creates physionet.org/files/mimic-iv-demo-meds/0.0.1/ under -P dir
    result = subprocess.run(
        [
            "wget",
            "-r",
            "-N",
            "-c",
            "-np",
            "-q",  # quiet; remove for debugging
            MIMIC_DEMO_MEDS_URL,
            "-P",
            str(temp_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            f"wget failed (return code {result.returncode}). "
            "Ensure wget is installed and the URL is reachable."
        ) from None
    wget_root = temp_dir / "physionet.org" / "files" / "mimic-iv-demo-meds" / "0.0.1"
    if not wget_root.is_dir():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"wget did not create expected path: {wget_root}")
    return temp_dir, wget_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MIMIC-IV demo MEDS events and labels for the quickstart demo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write output parquets (default: repo data/).",
    )
    args = parser.parse_args()

    print("Downloading MIMIC-IV demo MEDS (wget to temp dir)...")
    temp_dir, wget_root = _download_mimic_demo_to_temp()
    print("  -> Download complete.")

    try:
        if args.output_dir is None:
            repo_root = Path(__file__).resolve().parent.parent
            output_dir = repo_root / "data"
        else:
            output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        events_path = output_dir / "mimic_iv_demo_meds_events.parquet"
        labels_path = output_dir / "mimic_iv_demo_meds_labels.parquet"

        print("Building events table from train/tuning/held_out shards...")
        df_events = build_events_table(wget_root)
        n_events = len(df_events)
        n_subjects = df_events["subject_id"].nunique()
        print(f"  -> {n_events} events, {n_subjects} subjects")
        df_events.to_parquet(events_path, index=False)
        print(f"Wrote {events_path}")

        print("Building labels (hardcoded table for MIMIC-IV demo subjects)...")
        df_labels = build_labels_table(df_events)
        df_labels.to_parquet(labels_path, index=False)
        print(f"Wrote {labels_path}")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("Removed temporary download.")


if __name__ == "__main__":
    main()
