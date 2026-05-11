import json
import os

import httpx
import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_proxyml.schema import infer_schema

mcp = FastMCP("ProxyML")

_DEFAULT_BASE_URL = "https://api.proxyml.ai/api/v1"


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
        timeout=60.0,
    )


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
    df = pd.read_csv(csv_path)
    return infer_schema(df, immutable_cols)


# ---------------------------------------------------------------------------
# Schema tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_get_schema(schema_name: str = "default") -> dict:
    """Retrieve a stored feature schema by name."""
    async with _client() as c:
        r = await c.get(f"/schema/{schema_name}")
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def proxyml_put_schema(schema: dict, schema_name: str = "default") -> dict:
    """Upload or replace a feature schema."""
    async with _client() as c:
        r = await c.put(f"/schema/{schema_name}", content=json.dumps(schema))
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_synthesize_data(
    num_points: int = 100,
    schema_name: str = "default",
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
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Surrogate model tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_train_surrogate(
    samples: list[list],
    predictions: list,
    schema_name: str = "default",
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
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def proxyml_list_surrogates(limit: int = 50, offset: int = 0) -> dict:
    """List trained surrogate models, newest first.

    Returns model metadata (version ID, task, metrics, name, trained_at).
    Use version IDs from here with other tools.
    """
    async with _client() as c:
        r = await c.get("/surrogate/models", params={"limit": limit, "offset": offset})
        r.raise_for_status()
        return r.json()


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
        r.raise_for_status()
        return r.json()


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
        r.raise_for_status()
        return r.json()


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
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def proxyml_diff_models(version_a: str, version_b: str) -> dict:
    """Compare feature importances between two surrogate versions.

    Both surrogates must share the same task type and at least one feature.
    """
    async with _client() as c:
        r = await c.get("/explain/diff", params={"version_a": version_a, "version_b": version_b})
        r.raise_for_status()
        return r.json()


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

    flagged_features = [
        entry for entry in diff["coefficient_diff"]
        if abs(entry["delta"]) > coefficient_threshold
    ]

    metric_changes = {
        metric: {**entry, "flagged": entry["delta"] < -fidelity_threshold}
        for metric, entry in diff["metric_diff"].items()
    }

    features_added = diff.get("features_added", [])
    features_removed = diff.get("features_removed", [])

    passed = not flagged_features and not any(
        v["flagged"] for v in metric_changes.values()
    ) and not features_added and not features_removed

    reasons = []
    if flagged_features:
        top = ", ".join(
            f"{e['feature']} (Δ={e['delta']:+.3f})" for e in flagged_features[:3]
        )
        reasons.append(
            f"{len(flagged_features)} feature(s) exceeded coefficient threshold "
            f"({coefficient_threshold}): {top}"
            + (" and more" if len(flagged_features) > 3 else "")
        )
    for metric, entry in metric_changes.items():
        if entry["flagged"]:
            reasons.append(
                f"{metric} dropped by {abs(entry['delta']):.3f} "
                f"(threshold: {fidelity_threshold})"
            )
    if features_added:
        reasons.append(f"Features added: {', '.join(features_added)}")
    if features_removed:
        reasons.append(f"Features removed: {', '.join(features_removed)}")

    if passed:
        largest = max(
            (abs(e["delta"]) for e in diff["coefficient_diff"]), default=0.0
        )
        summary = (
            f"PASSED: No significant drift detected. "
            f"Largest coefficient shift: {largest:.3f} (threshold: {coefficient_threshold})."
        )
    else:
        summary = "FAILED: " + ". ".join(reasons) + "."

    return {
        "passed": passed,
        "version_a": version_a,
        "version_b": version_b,
        "summary": summary,
        "flagged_features": flagged_features,
        "metric_changes": metric_changes,
        "features_added": features_added,
        "features_removed": features_removed,
    }


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
        r.raise_for_status()
        return r.json()


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
        r.raise_for_status()
        return r.json()


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
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
