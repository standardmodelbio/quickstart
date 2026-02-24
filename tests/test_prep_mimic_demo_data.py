"""Unit tests for scripts/prep_mimic_demo_data.py (build_labels_table, build_events_table)."""
import pandas as pd
import pytest

# prep_mimic_demo_data is importable via conftest (scripts/ on path)
import prep_mimic_demo_data as prep


def test_build_labels_table_returns_correct_columns_and_order():
    """build_labels_table returns a DataFrame with expected columns and one row per subject."""
    subject_ids = [10000032, 10001217, 10002428]  # subset of DEMO_LABELS_ROWS
    df_events = pd.DataFrame({
        "subject_id": subject_ids * 2,  # two events per subject
        "time": pd.to_datetime(["2022-01-01", "2022-01-02"] * 3),
        "code": ["ICD10:I10"] * 6,
        "table": ["condition"] * 6,
        "value": [None] * 6,
    })
    out = prep.build_labels_table(df_events)
    assert list(out.columns) == [
        "subject_id",
        "readmission_risk",
        "phenotype_class",
        "overall_survival_months",
        "event_observed",
    ]
    assert len(out) == 3
    assert out["subject_id"].tolist() == subject_ids
    # Spot-check one row (10000032 -> 0, 0, 68.12..., 1)
    row = out[out["subject_id"] == 10000032].iloc[0]
    assert row["readmission_risk"] == 0
    assert row["phenotype_class"] == 0
    assert row["event_observed"] == 1


def test_build_labels_table_raises_on_unknown_subject():
    """build_labels_table raises ValueError if events contain a subject not in DEMO_LABELS_ROWS."""
    df_events = pd.DataFrame({
        "subject_id": [10000032, 99999999],  # 99999999 not in hardcoded table
        "time": pd.to_datetime(["2022-01-01", "2022-01-01"]),
        "code": ["ICD10:I10", "ICD10:I10"],
        "table": ["condition", "condition"],
        "value": [None, None],
    })
    with pytest.raises(ValueError, match="Labels are only defined for the standard MIMIC-IV demo"):
        prep.build_labels_table(df_events)


def test_build_events_table_produces_expected_schema(tmp_path):
    """build_events_table reads wget shards and returns DataFrame with subject_id, time, code, table, value."""
    # Create minimal wget layout: data/train/0.parquet, data/tuning/0.parquet, data/held_out/0.parquet
    for split in ("train", "tuning", "held_out"):
        (tmp_path / "data" / split).mkdir(parents=True)
    minimal = pd.DataFrame({
        "subject_id": [1, 1, 2],
        "time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01"]),
        "code": ["ICD10:I10", "RxNorm:861004", "LOINC:2093-3"],
        "numeric_value": [None, None, 140.0],
    })
    for split in ("train", "tuning", "held_out"):
        minimal.to_parquet(tmp_path / "data" / split / "0.parquet", index=False)

    out = prep.build_events_table(tmp_path)
    assert list(out.columns) == ["subject_id", "time", "code", "table", "value"]
    assert len(out) == 9  # 3 rows * 3 shards
    assert set(out["table"].unique()) == {"condition", "medication", "lab"}
    assert out["code"].str.startswith(("ICD10:", "RxNorm:", "LOINC:")).all()
