"""Tests for run_downstream_tasks in demo.py."""
import numpy as np
import pandas as pd
import torch

from demo import create_meds_cohort_with_labels, run_downstream_tasks


def _make_inputs(n_patients, dim, seed=42):
    """Build synthetic embeddings and labels for downstream task tests."""
    np.random.seed(seed)
    X = torch.randn(n_patients, dim)
    _, df_labels = create_meds_cohort_with_labels(n_patients=n_patients)
    return X, df_labels


def test_run_downstream_tasks_returns_metrics():
    """run_downstream_tasks returns a dict with the expected metric keys."""
    X, df_labels = _make_inputs(n_patients=60, dim=64, seed=42)
    metrics = run_downstream_tasks(X, df_labels)

    assert isinstance(metrics, dict)
    for key in ("auc", "accuracy", "mae", "c_index"):
        assert key in metrics, f"Missing metric key: {key}"


def test_run_downstream_tasks_metrics_in_range():
    """Downstream task metrics are within valid ranges."""
    X, df_labels = _make_inputs(n_patients=100, dim=32, seed=99)
    metrics = run_downstream_tasks(X, df_labels)

    assert 0.0 <= metrics["auc"] <= 1.0, f"AUC out of range: {metrics['auc']}"
    assert 0.0 <= metrics["accuracy"] <= 1.0, f"Accuracy out of range: {metrics['accuracy']}"
    assert metrics["mae"] >= 0.0, f"MAE is negative: {metrics['mae']}"
    assert 0.0 <= metrics["c_index"] <= 1.0, f"C-Index out of range: {metrics['c_index']}"
