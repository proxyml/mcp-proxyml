import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def run_challenger(
    csv_path: str,
    target_col: str,
    *,
    complexity: str,
    task: str,
    test_size: float,
    immutable_cols: list[str] | None,
    feature_names: list[str] | None,
    champion_predictions: list | None = None,
) -> dict[str, Any]:
    """Train a local challenger model and assemble an upload-ready payload.

    Rows with a missing (NaN) target_col value are dropped before training,
    the CV split, and champion scoring — never silently included; the drop
    count and a scope-limitation note are recorded on the returned dict and
    in upload_payload. If champion_predictions is given, it must have one
    entry per row of the CSV (same order, pre-drop) — champion labels are
    always the target column itself, so the identical rows are dropped from
    both sides and the two are always evaluated on the same population.

    Pure glue around proxyml.local — no FastMCP/HTTP dependency. Imports
    proxyml.local lazily so this module (and mcp_proxyml.server) stay
    importable without the optional 'challenger' extra installed.
    """
    from proxyml.local import Complexity, to_challenger_upload, train_auto_challenger

    try:
        complexity_enum = Complexity(complexity)
    except ValueError:
        valid = [c.value for c in Complexity]
        raise ValueError(f"Invalid complexity {complexity!r}; must be one of {valid}") from None

    if task not in ("classification", "regression", "auto"):
        raise ValueError(f"Invalid task {task!r}; must be 'classification', 'regression', or 'auto'")

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} not found in CSV columns: {list(df.columns)}")

    result = train_auto_challenger(
        df,
        target_col,
        immutable_cols=immutable_cols,
        complexity=complexity_enum,
        feature_names=feature_names,
        task=task,
        test_size=test_size,
        champion_predictions=champion_predictions,
    )

    upload_payload = to_challenger_upload(result)

    return {
        "upload_payload": upload_payload,
        "challenger_metrics": result.metrics,
        "task": result.task,
        "complexity": result.complexity.value,
        "champion_metrics_included": result.champion_metrics is not None,
        "n_samples_total": result.n_samples_total,
        "n_samples_dropped_unlabeled": result.n_samples_dropped_unlabeled,
        "population_note": result.population_note,
    }
