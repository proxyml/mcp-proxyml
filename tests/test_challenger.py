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


def _classification_csv_with_nan_targets(tmp_path, n=70, n_nan=10, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + x2 > 0).astype(float)
    y[:n_nan] = np.nan
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    path = tmp_path / "classification_nan.csv"
    df.to_csv(path, index=False)
    return str(path), df


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
        **_kwargs(champion_predictions=df["y"].tolist()),
    )
    assert out["champion_metrics_included"] is True
    assert "champion_metrics" in out["upload_payload"]
    assert set(out["upload_payload"]["champion_metrics"]) == set(out["challenger_metrics"])


def test_champion_metrics_omitted_when_not_provided(tmp_path):
    csv_path = _classification_csv(tmp_path)
    out = run_challenger(csv_path, "y", **_kwargs())
    assert out["champion_metrics_included"] is False
    assert "champion_metrics" not in out["upload_payload"]


def test_champion_predictions_wrong_length_raises(tmp_path):
    csv_path = _classification_csv(tmp_path)  # 70 rows
    with pytest.raises(ValueError, match="one entry per row"):
        run_challenger(csv_path, "y", **_kwargs(champion_predictions=[0, 1, 0]))


# ---------------------------------------------------------------------------
# Missing (NaN) target rows
# ---------------------------------------------------------------------------

def test_nan_targets_are_dropped_and_counted(tmp_path):
    csv_path, df = _classification_csv_with_nan_targets(tmp_path, n=70, n_nan=10)
    out = run_challenger(csv_path, "y", **_kwargs())
    assert out["n_samples_total"] == 70
    assert out["n_samples_dropped_unlabeled"] == 10
    assert out["upload_payload"]["n_samples"] == 60
    assert "10" in out["population_note"]


def test_no_nan_targets_reports_zero_dropped(tmp_path):
    csv_path = _classification_csv(tmp_path, n=70)
    out = run_challenger(csv_path, "y", **_kwargs())
    assert out["n_samples_total"] == 70
    assert out["n_samples_dropped_unlabeled"] == 0


def test_all_nan_targets_raises(tmp_path):
    csv_path, _ = _classification_csv_with_nan_targets(tmp_path, n=20, n_nan=20)
    with pytest.raises(ValueError, match="missing"):
        run_challenger(csv_path, "y", **_kwargs())


def test_champion_predictions_dropped_from_same_rows_as_challenger(tmp_path):
    # Champion predictions mirror the (possibly-NaN) target itself, except on
    # rows that get dropped as unlabeled, where they're deliberately wrong.
    # If those rows leaked into scoring, champion accuracy would come in
    # under 1.0 instead of exactly 1.0 — proving the shared-drop guarantee.
    csv_path, df = _classification_csv_with_nan_targets(tmp_path, n=70, n_nan=10)
    champion_predictions = [0.0 if pd.isna(v) else v for v in df["y"]]

    out = run_challenger(csv_path, "y", **_kwargs(champion_predictions=champion_predictions))

    assert out["upload_payload"]["champion_metrics"]["accuracy"] == 1.0


def test_population_note_in_upload_payload(tmp_path):
    csv_path, _ = _classification_csv_with_nan_targets(tmp_path, n=70, n_nan=10)
    out = run_challenger(csv_path, "y", **_kwargs())
    assert out["upload_payload"]["population_note"] == out["population_note"]
    assert out["upload_payload"]["n_samples_total"] == 70
    assert out["upload_payload"]["n_samples_dropped_unlabeled"] == 10


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
        **_kwargs(champion_predictions=[0, 1] * 35),
    )
    json.dumps(out)
