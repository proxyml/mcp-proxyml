import json
import logging
import os

import httpx
import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_proxyml.challenger import run_challenger
from mcp_proxyml.drift import interpret_diff
from mcp_proxyml.schema import infer_schema

logger = logging.getLogger(__name__)

mcp = FastMCP("ProxyML")

_DEFAULT_BASE_URL = "https://api.proxyml.ai/api/v1"
_TIMEOUT = 120.0


def _client() -> httpx.AsyncClient:
    api_key = os.environ.get("PROXYML_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "PROXYML_API_KEY is not set. "
            "Add it to the MCP server env config and restart."
        )
    return httpx.AsyncClient(
        base_url=os.environ.get("PROXYML_BASE_URL", _DEFAULT_BASE_URL),
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        timeout=_TIMEOUT,
    )


def _result(r: httpx.Response) -> dict:
    """Return the JSON body on success, or a structured error dict on failure."""
    if r.is_success:
        return r.json()
    try:
        detail = r.json().get("detail", r.text)
    except Exception:
        detail = r.text
    return {"error": True, "status_code": r.status_code, "detail": detail}


# ---------------------------------------------------------------------------
# Local tools (no API call)
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_infer_schema(
    csv_path: str,
    immutable_cols: list[str] | None = None,
) -> dict:
    """Infer a ProxyML feature schema from a local CSV file.

    Reads the file locally — no data is sent to the ProxyML server.
    Returns a schema dict ready to pass to proxyml_put_schema.
    Integer columns default to 'count'; consider 'categorical_ordinal' for
    ordered categories like ratings or education level.
    """
    if not os.path.isfile(csv_path):
        return {"error": True, "detail": f"File not found or not a file: {csv_path}"}
    df = pd.read_csv(csv_path)
    return infer_schema(df, immutable_cols)


# ---------------------------------------------------------------------------
# Challenger tools (local, no API call)
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_train_challenger(
    csv_path: str,
    target_col: str,
    complexity: str = "moderate",
    task: str = "auto",
    test_size: float = 0.2,
    immutable_cols: list[str] | None = None,
    feature_names: list[str] | None = None,
    champion_labels: list | None = None,
    champion_predictions: list | None = None,
) -> dict:
    """Train a local challenger model on a CSV file and assemble an upload-ready payload.

    Trains entirely locally — no data leaves this process. Requires the
    'challenger' extra: pip install 'mcp-proxyml[challenger]'.
    complexity is one of 'simple', 'moderate', 'flexible'; task is
    'classification', 'regression', or 'auto' to infer from target_col.
    Pass champion_labels and champion_predictions together to also score a
    champion model and include champion_metrics in the returned payload —
    the upload endpoint requires champion_metrics, so omit them only if you
    plan to fill them in later (see proxyml_score_champion).
    Returns upload_payload: save it and upload via the ProxyML dashboard's
    "Upload challenger" button, or POST it yourself to
    /app/projects/{project_id}/challenger.
    """
    if not os.path.isfile(csv_path):
        return {"error": True, "detail": f"File not found or not a file: {csv_path}"}
    try:
        return run_challenger(
            csv_path,
            target_col,
            complexity=complexity,
            task=task,
            test_size=test_size,
            immutable_cols=immutable_cols,
            feature_names=feature_names,
            champion_labels=champion_labels,
            champion_predictions=champion_predictions,
        )
    except ImportError:
        return {
            "error": True,
            "detail": (
                "Local challenger training requires the 'challenger' extra: "
                "pip install 'mcp-proxyml[challenger]'"
            ),
        }
    except ValueError as exc:
        return {"error": True, "detail": str(exc)}
    except Exception as exc:
        logger.exception("proxyml_train_challenger failed")
        return {"error": True, "detail": str(exc)}


@mcp.tool()
async def proxyml_score_champion(
    labels: list,
    predictions: list,
    task: str,
) -> dict:
    """Score a champion model's predictions against real labels, locally.

    task must be 'classification' or 'regression' (no 'auto' — pass the same
    task a paired challenger resolved to, so the two stay comparable).
    Returns {"f1":..., "accuracy":...} for classification or {"r2":...} for
    regression. Requires the 'challenger' extra:
    pip install 'mcp-proxyml[challenger]'.
    """
    if task not in ("classification", "regression"):
        return {
            "error": True,
            "detail": f"Invalid task {task!r}; must be 'classification' or 'regression'",
        }
    if len(labels) != len(predictions):
        return {
            "error": True,
            "detail": (
                f"labels and predictions must be the same length "
                f"(got {len(labels)} and {len(predictions)})"
            ),
        }
    try:
        from proxyml.local import score_champion
        return score_champion(labels, predictions, task=task)
    except ImportError:
        return {
            "error": True,
            "detail": (
                "Local challenger training requires the 'challenger' extra: "
                "pip install 'mcp-proxyml[challenger]'"
            ),
        }
    except Exception as exc:
        logger.exception("proxyml_score_champion failed")
        return {"error": True, "detail": str(exc)}


# ---------------------------------------------------------------------------
# Schema tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_get_schema(schema_name: str) -> dict:
    """Retrieve a stored feature schema by name."""
    async with _client() as c:
        r = await c.get(f"/schema/{schema_name}")
        return _result(r)


@mcp.tool()
async def proxyml_put_schema(schema: dict, schema_name: str) -> dict:
    """Upload or replace a feature schema."""
    async with _client() as c:
        r = await c.put(f"/schema/{schema_name}", content=json.dumps(schema))
        return _result(r)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_synthesize_data(
    schema_name: str,
    num_points: int = 100,
    instance: list | None = None,
) -> dict:
    """Generate synthetic samples from the stored feature schema.

    If instance is provided, blends schema samples with perturbations around
    that point (useful for local explanations and counterfactual search).
    Returns samples, feature_names, and feature_types.
    """
    async with _client() as c:
        if instance is None:
            payload = {"n": num_points, "schema_name": schema_name}
            r = await c.post("/synthesize/neighbors", content=json.dumps(payload))
        else:
            payload = {"n": num_points, "instance": instance, "schema_name": schema_name}
            r = await c.post("/synthesize/blended", content=json.dumps(payload))
        return _result(r)


# ---------------------------------------------------------------------------
# Surrogate model tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_train_surrogate(
    samples: list[list],
    predictions: list,
    schema_name: str,
    task: str = "auto",
    test_size: float = 0.2,
    feature_names: list[str] | None = None,
    name: str | None = None,
    comments: str | None = None,
) -> dict:
    """Train a linear surrogate model on samples scored by the user's model.

    samples: rows returned by proxyml_synthesize_data.
    predictions: your model's output for each sample, in the same order.
    task: 'classification', 'regression', or 'auto' (default).
    Returns version ID, fidelity metrics, and any training warnings.
    """
    payload: dict = {
        "samples": samples,
        "predictions": predictions,
        "schema_name": schema_name,
        "task": task,
        "test_size": test_size,
    }
    if feature_names is not None:
        payload["feature_names"] = feature_names
    if name is not None:
        payload["name"] = name
    if comments is not None:
        payload["comments"] = comments
    async with _client() as c:
        r = await c.post("/surrogate/train", content=json.dumps(payload))
        return _result(r)


@mcp.tool()
async def proxyml_list_surrogates(limit: int = 50, offset: int = 0) -> dict:
    """List trained surrogate models, newest first.

    Returns model metadata (version ID, task, metrics, name, trained_at).
    Use version IDs from here with other tools.
    """
    async with _client() as c:
        r = await c.get("/surrogate/models", params={"limit": limit, "offset": offset})
        return _result(r)


# ---------------------------------------------------------------------------
# Explanation tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_get_summary(version: str | None = None) -> dict:
    """Get feature importances and model summary for a surrogate.

    Omit version to use the most recently trained model.
    """
    async with _client() as c:
        params = {"version": version} if version is not None else {}
        r = await c.get("/explain/summary", params=params)
        return _result(r)


@mcp.tool()
async def proxyml_explain_local(
    instance: list,
    version: str | None = None,
) -> dict:
    """Per-feature contribution breakdown for a single instance.

    Returns prediction, feature_contributions (sorted by abs contribution),
    intercept, and optionally probabilities and per-class contributions.
    """
    payload: dict = {"instance": instance}
    if version is not None:
        payload["version"] = version
    async with _client() as c:
        r = await c.post("/explain/local", content=json.dumps(payload))
        return _result(r)


@mcp.tool()
async def proxyml_find_counterfactual(
    instance: list,
    target_label: float | str,
    n_neighbors: int = 10000,
    perturbation_scale: float = 0.1,
    version: str | None = None,
) -> dict:
    """Find a counterfactual: the nearest point that flips the prediction to target_label.

    A counterfactual is not guaranteed — check the 'counterfactual' field in the
    response (None means none was found within the search budget).
    """
    payload: dict = {
        "instance": instance,
        "target_label": target_label,
        "n_neighbors": n_neighbors,
        "perturbation_scale": perturbation_scale,
    }
    if version is not None:
        payload["version"] = version
    async with _client() as c:
        r = await c.post("/explain/counterfactual", content=json.dumps(payload))
        return _result(r)


@mcp.tool()
async def proxyml_diff_models(version_a: str, version_b: str) -> dict:
    """Compare feature importances between two surrogate versions.

    Both surrogates must share the same task type and at least one feature.
    """
    async with _client() as c:
        r = await c.get("/explain/diff", params={"version_a": version_a, "version_b": version_b})
        return _result(r)


@mcp.tool()
async def proxyml_detect_drift(
    version_a: str,
    version_b: str,
    coefficient_threshold: float = 0.1,
    fidelity_threshold: float = 0.05,
) -> dict:
    """Detect behavioural drift between two surrogate versions.

    Calls diff_models and applies thresholds to produce a structured pass/fail
    suitable for use in CI/CD pipelines.

    coefficient_threshold: flag features whose absolute coefficient delta exceeds this (default 0.1).
    fidelity_threshold: flag metrics that drop by more than this amount (default 0.05).

    Returns:
        passed: False if any feature or metric breach a threshold, or if features were added/removed.
        flagged_features: features whose |delta| exceeded coefficient_threshold, sorted by |delta| desc.
        metric_changes: per-metric deltas with a 'flagged' key where the drop exceeded fidelity_threshold.
        features_added / features_removed: schema changes between versions.
        summary: human-readable explanation of the result.
    """
    async with _client() as c:
        r = await c.get("/explain/diff", params={"version_a": version_a, "version_b": version_b})
        r.raise_for_status()
        diff = r.json()

    return interpret_diff(diff, version_a, version_b, coefficient_threshold, fidelity_threshold)


# ---------------------------------------------------------------------------
# Surrogate predictions
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_predict_batch(
    instances: list[list],
    version: str | None = None,
) -> dict:
    """Get surrogate model predictions for a list of instances.

    Returns predictions in the same order as instances, along with the
    model version used. Useful for comparing surrogate predictions against
    a dev model's output on the same synthetic samples.
    """
    payload: dict = {"inputs": instances}
    if version is not None:
        payload["version"] = version
    async with _client() as c:
        r = await c.post("/surrogate/predict/batch", content=json.dumps(payload))
        return _result(r)


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_get_usage() -> dict:
    """Return current tier, request count, and quota for the authenticated account.

    Useful as a pre-flight check before expensive operations like training or
    large synthesis runs.
    """
    async with _client() as c:
        r = await c.get("/account/usage")
        return _result(r)


# ---------------------------------------------------------------------------
# Surrogate export and batch explanation
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_export_surrogate(version: str) -> dict:
    """Export the full surrogate model: coefficients, intercepts, scaler params, and classes.

    Returns everything needed to reconstruct the surrogate outside ProxyML.
    Richer than proxyml_get_summary — use this for audit trails and governance reports.
    """
    async with _client() as c:
        r = await c.get(f"/surrogate/models/{version}/export")
        return _result(r)


@mcp.tool()
async def proxyml_explain_local_batch(
    instances: list[list],
    version: str | None = None,
) -> dict:
    """Per-feature contribution breakdown for multiple instances in one call.

    Equivalent to calling proxyml_explain_local for each instance, but more
    efficient. Returns a 'results' list in the same order as instances.
    """
    payload: dict = {"instances": instances}
    if version is not None:
        payload["version"] = version
    async with _client() as c:
        r = await c.post("/explain/local/batch", content=json.dumps(payload))
        return _result(r)


# ---------------------------------------------------------------------------
# Additional schema and model management tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_list_schemas() -> dict:
    """List all named feature schemas for the authenticated account."""
    async with _client() as c:
        r = await c.get("/schemas")
        return _result(r)


@mcp.tool()
async def proxyml_delete_schema(schema_name: str) -> dict:
    """Delete a named feature schema. Returns success or a structured error."""
    async with _client() as c:
        r = await c.delete(f"/schema/{schema_name}")
        if r.status_code == 204:
            return {"deleted": True, "schema_name": schema_name}
        return _result(r)


@mcp.tool()
async def proxyml_get_model_schema(version: str) -> dict:
    """Retrieve the feature schema that was snapshotted when a surrogate was trained.

    Useful for auditing which schema version a model was trained against.
    """
    async with _client() as c:
        r = await c.get(f"/surrogate/models/{version}/schema")
        return _result(r)


@mcp.tool()
async def proxyml_update_surrogate(
    version: str,
    name: str | None = None,
    comments: str | None = None,
) -> dict:
    """Update the name or comments of a surrogate without retraining.

    Omit a parameter to leave that field unchanged.
    Pass null/None to clear a field.
    """
    payload: dict = {}
    if name is not None:
        payload["name"] = name
    if comments is not None:
        payload["comments"] = comments
    if not payload:
        return {"error": True, "detail": "Provide at least one of: name, comments"}
    async with _client() as c:
        r = await c.patch(f"/surrogate/models/{version}", content=json.dumps(payload))
        return _result(r)


@mcp.tool()
async def proxyml_delete_surrogate(model_id: str) -> dict:
    """Delete a surrogate model by its UUID. Returns success or a structured error."""
    async with _client() as c:
        r = await c.delete(f"/surrogate/models/{model_id}")
        if r.status_code == 204:
            return {"deleted": True, "model_id": model_id}
        return _result(r)


@mcp.tool()
async def proxyml_predict(
    instance: list,
    version: str | None = None,
) -> dict:
    """Get a surrogate model prediction for a single instance.

    Returns prediction, and optionally probability (classification) or
    a regression value. Use proxyml_predict_batch for multiple instances.
    """
    payload: dict = {"inputs": instance}
    if version is not None:
        payload["version"] = version
    async with _client() as c:
        r = await c.post("/surrogate/predict", content=json.dumps(payload))
        return _result(r)


@mcp.tool()
async def proxyml_find_counterfactuals(
    instances: list[list],
    target_label: float | str,
    n_neighbors: int = 10000,
    perturbation_scale: float = 0.1,
    version: str | None = None,
) -> dict:
    """Find counterfactuals for multiple instances in one call.

    Equivalent to calling proxyml_find_counterfactual for each instance.
    Each result has a 'counterfactual' field (None if none was found).
    """
    payload: dict = {
        "instances": instances,
        "target_label": target_label,
        "n_neighbors": n_neighbors,
        "perturbation_scale": perturbation_scale,
    }
    if version is not None:
        payload["version"] = version
    async with _client() as c:
        r = await c.post("/explain/counterfactual/batch", content=json.dumps(payload))
        return _result(r)


@mcp.tool()
async def proxyml_health_check() -> dict:
    """Check API connectivity. Does not require authentication and does not count against usage quota.

    Returns status, model_loaded, and API version. Useful as a pre-flight
    check before training or synthesis operations.
    """
    base_url = os.environ.get("PROXYML_BASE_URL", _DEFAULT_BASE_URL)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.get(f"{base_url}/health")
            return _result(r)
    except httpx.RequestError as exc:
        return {"error": True, "detail": f"Network error: {exc}"}


# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
