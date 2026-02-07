"""Tests for create_meds_cohort_with_labels in demo.py."""
import random

import numpy as np
import pandas as pd
import pytest

from demo import create_meds_cohort_with_labels


def test_create_meds_cohort_with_labels_shape_and_columns():
    """DataFrames have expected shapes and required columns."""
    random.seed(42)
    np.random.seed(42)
    df_meds, df_labels = create_meds_cohort_with_labels(n_patients=50)

    assert isinstance(df_meds, pd.DataFrame)
    assert isinstance(df_labels, pd.DataFrame)
    assert len(df_labels) == 50
    assert len(df_meds) >= 50

    # MEDS events: subject_id, time, code, table, value
    for col in ["subject_id", "time", "code", "table"]:
        assert col in df_meds.columns, f"df_meds missing column {col}"

    # Labels: subject_id, readmission_risk, phenotype_class, overall_survival_months, event_observed
    for col in ["subject_id", "readmission_risk", "phenotype_class", "overall_survival_months", "event_observed"]:
        assert col in df_labels.columns, f"df_labels missing column {col}"


def test_create_meds_cohort_with_labels_value_ranges():
    """Label columns are in valid ranges."""
    random.seed(123)
    np.random.seed(123)
    _, df_labels = create_meds_cohort_with_labels(n_patients=30)

    assert df_labels["readmission_risk"].isin([0, 1]).all()
    assert df_labels["phenotype_class"].between(0, 3).all()
    assert (df_labels["overall_survival_months"] >= 1.0).all()
    assert df_labels["event_observed"].isin([0, 1]).all()


def test_create_meds_cohort_with_labels_one_per_patient():
    """Exactly one label row per patient."""
    random.seed(0)
    np.random.seed(0)
    df_meds, df_labels = create_meds_cohort_with_labels(n_patients=20)

    pids_meds = set(df_meds["subject_id"].unique())
    pids_labels = set(df_labels["subject_id"].unique())
    assert pids_meds == pids_labels
    assert len(df_labels) == len(pids_labels)


def test_create_meds_cohort_with_labels_chronological():
    """Events are sorted by subject_id and time."""
    random.seed(7)
    np.random.seed(7)
    df_meds, _ = create_meds_cohort_with_labels(n_patients=25)

    df_sorted = df_meds.sort_values(["subject_id", "time"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_meds, df_sorted)
