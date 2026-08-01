import json

import numpy as np
import pandas as pd
import pytest
from mcp_proxyml.challenger import run_challenger


def _classification_csv(tmp_path, n=70, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + x2 > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    path = tmp_path / "classification.csv"
    df.to_csv(path, index=False)
    return str(path)


def _regression_csv(tmp_path, n=70, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2 * x1 - x2 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return str(path)


def _kwargs(**overrides):
    base = dict(
        complexity="moderate",
        task="auto",
        test_size=0.2,
        immutable_cols=None,
        feature_names=None,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Task auto-inference
# ---------------------------------------------------------------------------

def test_auto_resolves_classification(tmp_path):
    csv_path = _classification_csv(tmp_path)
    out = run_challenger(csv_path, "y", **_kwargs())
    assert out["task"] == "classification"
    assert set(out["challenger_metrics"]) == {"f1", "accuracy"}


def test_auto_resolves_regression(tmp_path):
    csv_path = _regression_csv(tmp_path)
    out = run_challenger(csv_path, "y", **_kwargs())
    assert out["task"] == "regression"
    assert set(out["challenger_metrics"]) == {"r2"}


# ---------------------------------------------------------------------------
# Complexity ladder
# ---------------------------------------------------------------------------

def test_complexity_changes_hyperparameters(tmp_path):
    csv_path = _classification_csv(tmp_path)
    simple = run_challenger(csv_path, "y", **_kwargs(complexity="simple"))
    flexible = run_challenger(csv_path, "y", **_kwargs(complexity="flexible"))
    assert simple["complexity"] == "simple"
    assert flexible["complexity"] == "flexible"
    simple_cs = simple["upload_payload"]["export"]["hyperparameters"]["Cs"]
    flexible_cs = flexible["upload_payload"]["export"]["hyperparameters"]["Cs"]
    assert simple_cs != flexible_cs


def test_invalid_complexity_raises(tmp_path):
    csv_path = _classification_csv(tmp_path)
    with pytest.raises(ValueError, match="complexity"):
        run_challenger(csv_path, "y", **_kwargs(complexity="bogus"))


def test_invalid_task_raises(tmp_path):
    csv_path = _classification_csv(tmp_path)
    with pytest.raises(ValueError, match="task"):
        run_challenger(csv_path, "y", **_kwargs(task="bogus"))


# ---------------------------------------------------------------------------
# Champion metrics
# ---------------------------------------------------------------------------

def test_champion_metrics_included_when_provided(tmp_path):
    csv_path = _classification_csv(tmp_path)
    df = pd.read_csv(csv_path)
    out = run_challenger(
        csv_path,
        "y",
        **_kwargs(
            champion_labels=df["y"].tolist(),
            champion_predictions=df["y"].tolist(),
        ),
    )
    assert out["champion_metrics_included"] is True
    assert "champion_metrics" in out["upload_payload"]
    assert set(out["upload_payload"]["champion_metrics"]) == set(out["challenger_metrics"])


def test_champion_metrics_omitted_when_not_provided(tmp_path):
    csv_path = _classification_csv(tmp_path)
    out = run_challenger(csv_path, "y", **_kwargs())
    assert out["champion_metrics_included"] is False
    assert "champion_metrics" not in out["upload_payload"]


def test_mismatched_champion_lengths_raises(tmp_path):
    csv_path = _classification_csv(tmp_path)
    with pytest.raises(ValueError, match="same length"):
        run_challenger(
            csv_path,
            "y",
            **_kwargs(champion_labels=[0, 1], champion_predictions=[0, 1, 1]),
        )


def test_only_champion_labels_raises(tmp_path):
    csv_path = _classification_csv(tmp_path)
    with pytest.raises(ValueError, match="both"):
        run_challenger(csv_path, "y", **_kwargs(champion_labels=[0, 1]))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_missing_target_col_raises(tmp_path):
    csv_path = _classification_csv(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        run_challenger(csv_path, "does_not_exist", **_kwargs())


# ---------------------------------------------------------------------------
# JSON-serializability
# ---------------------------------------------------------------------------

def test_upload_payload_complexity_is_plain_string(tmp_path):
    csv_path = _classification_csv(tmp_path)
    out = run_challenger(csv_path, "y", **_kwargs())
    assert isinstance(out["upload_payload"]["complexity"], str)


def test_upload_payload_is_json_serializable(tmp_path):
    csv_path = _classification_csv(tmp_path)
    out = run_challenger(
        csv_path,
        "y",
        **_kwargs(
            champion_labels=[0, 1] * 35,
            champion_predictions=[0, 1] * 35,
        ),
    )
    json.dumps(out)
