"""Tests for extract_embeddings in demo.py using mocks."""
from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from demo import extract_embeddings


def test_extract_embeddings_output_shape_with_mock():
    """extract_embeddings returns a tensor of shape (n_patients, hidden_size)."""
    n_patients = 3
    hidden_size = 64
    df = pd.DataFrame({
        "subject_id": [f"{i:04d}" for i in range(n_patients)],
        "time": pd.to_datetime(["2023-01-01"] * n_patients),
        "code": ["CPT:99213"] * n_patients,
        "table": ["procedure"] * n_patients,
        "value": [None] * n_patients,
    })

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    # hidden_states: list of tensors; last one shape (batch, seq_len, hidden_size)
    mock_model.return_value.hidden_states = [
        None,
        torch.randn(1, 10, hidden_size),
    ]

    mock_tokenizer = MagicMock()
    mock_encoding = MagicMock()
    mock_encoding.input_ids = torch.randint(0, 100, (1, 5))
    mock_encoding.to = lambda device: mock_encoding
    mock_tokenizer.return_value = mock_encoding

    with patch("demo.process_ehr_info", return_value="dummy text"):
        result = extract_embeddings(df, mock_model, mock_tokenizer)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (n_patients, hidden_size)
