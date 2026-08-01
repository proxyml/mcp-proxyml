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
    champion_labels: list | None = None,
    champion_predictions: list | None = None,
) -> dict[str, Any]:
    """Train a local challenger model and assemble an upload-ready payload.

    Pure glue around proxyml.local — no FastMCP/HTTP dependency. Imports
    proxyml.local lazily so this module (and mcp_proxyml.server) stay
    importable without the optional 'challenger' extra installed.
    """
    from proxyml.local import Complexity, score_champion, to_challenger_upload, train_auto_challenger

    if (champion_labels is None) != (champion_predictions is None):
        raise ValueError(
            "champion_labels and champion_predictions must both be provided, or both omitted"
        )
    if champion_labels is not None and len(champion_labels) != len(champion_predictions):
        raise ValueError(
            f"champion_labels and champion_predictions must be the same length "
            f"(got {len(champion_labels)} and {len(champion_predictions)})"
        )

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
    )

    champion_metrics = None
    if champion_labels is not None:
        champion_metrics = score_champion(champion_labels, champion_predictions, task=result.task)

    upload_payload = to_challenger_upload(
        result,
        n_samples=len(df),
        champion_metrics=champion_metrics,
    )

    return {
        "upload_payload": upload_payload,
        "challenger_metrics": result.metrics,
        "task": result.task,
        "complexity": result.complexity.value,
        "champion_metrics_included": champion_metrics is not None,
    }
