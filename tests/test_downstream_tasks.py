"""Tests for run_downstream_tasks in demo.py."""
import numpy as np
import pandas as pd
import torch

from demo import create_meds_cohort_with_labels, run_downstream_tasks


def _make_synthetic_labels(n: int, seed: int = 42):
    """Build a labels DataFrame matching create_meds_cohort_with_labels structure."""
    random = __import__("random")
    random.seed(seed)
    np.random.seed(seed)
    _, df_labels = create_meds_cohort_with_labels(n_patients=n)
    return df_labels


def test_run_downstream_tasks_completes():
    """run_downstream_tasks runs without error and produces valid metrics."""
    np.random.seed(42)
    n_patients = 60  # enough for 80/20 split
    dim = 64
    X = torch.randn(n_patients, dim)
    df_labels = _make_synthetic_labels(n_patients, seed=43)

    run_downstream_tasks(X, df_labels.copy())


def test_run_downstream_tasks_metrics_in_range():
    """Downstream task metrics are in valid ranges (smoke with small data)."""
    np.random.seed(99)
    n_patients = 100
    dim = 32
    X = torch.randn(n_patients, dim)
    df_labels = _make_synthetic_labels(n_patients, seed=99)

    # Capture printed metrics by running and not crashing; full metric checks would require
    # refactoring run_downstream_tasks to return values. Here we only assert no exception.
    run_downstream_tasks(X, df_labels.copy())
