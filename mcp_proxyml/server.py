import json
import logging
import os

import httpx
import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_proxyml.challenger import run_challenger
from mcp_proxyml.drift import interpret_diff
from mcp_proxyml.schema import infer_schema, infer_schema_like

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


def _ci_client(ci_token: str | None = None) -> httpx.AsyncClient:
    """Client authenticated with a project-scoped CI token (X-CI-Token).

    Deliberately does NOT send the account-wide X-API-KEY: a CI token only
    authorizes writes to the one project it was minted for.
    """
    token = ci_token or os.environ.get("PROXYML_CI_TOKEN", "")
    if not token:
        raise RuntimeError(
            "No CI token. Pass ci_token or set PROXYML_CI_TOKEN "
            "(mint one from the project's page in the ProxyML dashboard)."
        )
    return httpx.AsyncClient(
        base_url=os.environ.get("PROXYML_BASE_URL", _DEFAULT_BASE_URL),
        headers={"X-CI-Token": token, "Content-Type": "application/json"},
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
    Returns a schema dict ready to pass to proxyml_put_schema. It includes
    n_samples (the CSV's row count); keep it when uploading so data-drift checks
    can allow for small training sets. If you upload the schema unedited, also add
    "source": "data"; if you changed types or statistics by hand, "hand_written".
    It is reported as baseline_source on drift checks (unknown if left out).
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
    champion_predictions: list | None = None,
) -> dict:
    """Train a local challenger model on a CSV file and assemble an upload-ready payload.

    Trains entirely locally — no data leaves this process. Requires the
    'challenger' extra: pip install 'mcp-proxyml[challenger]'.
    complexity is one of 'simple', 'moderate', 'flexible'; task is
    'classification', 'regression', or 'auto' to infer from target_col.

    Rows with a missing target_col value are dropped before training, the
    CV split, and champion scoring — never silently included. The response
    reports n_samples_total, n_samples_dropped_unlabeled, and a
    population_note that is also embedded in upload_payload, so the
    dropped-row count and scope limitation travel with the upload for an
    auditor to see.

    Pass champion_predictions — one prediction per row of the CSV, same
    order — to also score a champion model against the same target_col and
    include champion_metrics in the returned payload; the identical rows
    dropped for a missing target are dropped from champion_predictions too,
    so the champion and challenger are always evaluated on the same
    population, and upload_payload's challenger_data_fingerprint and
    champion_data_fingerprint are set to matching values automatically. The
    upload endpoint requires champion_metrics, so omit champion_predictions
    only if you plan to fill it in later via proxyml_score_champion — in
    that case, copy its data_fingerprint into upload_payload as
    champion_data_fingerprint before uploading, so the endpoint can still
    catch the two having been scored on different data.

    Returns upload_payload: upload it with proxyml_upload_challenger (needs a
    project CI token), via the ProxyML dashboard's "Upload challenger"
    button, or POST it yourself to /app/projects/{project_id}/challenger.
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
    Returns {"f1":..., "accuracy":..., "data_fingerprint":...} for
    classification or {"r2":..., "data_fingerprint":...} for regression.
    Requires the 'challenger' extra: pip install 'mcp-proxyml[challenger]'.

    data_fingerprint is a hash of `labels` — since this tool call has no
    connection to any proxyml_train_challenger() call that ran separately,
    there's nothing here to guarantee the two were scored on the same data.
    If you're merging this result's metrics into a proxyml_train_challenger
    upload_payload by hand as champion_metrics, also copy this
    data_fingerprint in as champion_data_fingerprint — the upload endpoint
    rejects the upload if it doesn't match the challenger's own fingerprint,
    catching an accidental mismatched file (e.g. the wrong CSV) instead of
    silently producing a misleading comparison.
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
        from proxyml.local import fingerprint_labels, score_champion
        metrics = score_champion(labels, predictions, task=task)
        return {**metrics, "data_fingerprint": fingerprint_labels(labels)}
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
# CI challenger tools (project-scoped CI token, not the account API key)
# ---------------------------------------------------------------------------

@mcp.tool()
async def proxyml_upload_challenger(
    upload_payload: dict,
    ci_token: str | None = None,
) -> dict:
    """Upload a challenger comparison to a ProxyML project using a CI token.

    upload_payload is the upload_payload from proxyml_train_challenger (with
    champion_metrics filled in). Authenticates with a project-scoped CI token
    — passed as ci_token or read from PROXYML_CI_TOKEN — instead of
    PROXYML_API_KEY; the token authorizes writes to exactly one project.
    Mint it from that project's page in the ProxyML dashboard. Returns the
    new challenger_version. Requires a ProxyML server with /ci/challenger
    (alposit 0.2.0+).
    """
    try:
        client = _ci_client(ci_token)
    except RuntimeError as exc:
        return {"error": True, "detail": str(exc)}
    async with client as c:
        r = await c.post("/ci/challenger", content=json.dumps(upload_payload))
        return _result(r)


@mcp.tool()
async def proxyml_get_challenger_summary(ci_token: str | None = None) -> dict:
    """Fetch a CI-ready summary of a project's latest challenger comparison.

    Returns the winner, primary metric and delta, and a ready-to-post
    markdown snippet. The project is determined by the CI token (ci_token or
    PROXYML_CI_TOKEN). Errors if nothing has been uploaded to the project yet.
    """
    try:
        client = _ci_client(ci_token)
    except RuntimeError as exc:
        return {"error": True, "detail": str(exc)}
    async with client as c:
        r = await c.get("/ci/challenger/summary")
        return _result(r)


# ---------------------------------------------------------------------------
# Data drift (recent data vs. a model's training data)
# ---------------------------------------------------------------------------

async def _baseline_features(model_id: str) -> list[dict] | None:
    """The model's schema features, or None (quietly) if they cannot be fetched."""
    try:
        async with _client() as c:
            r = await c.get(f"/surrogate/models/{model_id}/schema")
        features = r.json().get("features") if r.is_success else None
        return features if isinstance(features, list) else None
    except Exception:  # a failed lookup must never fail the drift check itself
        return None


@mcp.tool()
async def proxyml_check_data_drift(
    model_id: str,
    csv_path: str,
    label: str | None = None,
    thresholds: dict | None = None,
    baseline_n_samples: int | None = None,
    align_to_model: bool = True,
) -> dict:
    """Check whether recent data has drifted from the data a model was trained on.

    Not the same as proxyml_detect_drift, which compares two model versions.
    This compares a local CSV of recent data against the model's stored
    training schema. The CSV is summarized locally; only statistics (means,
    standard deviations, category names with frequencies, count rates) and the
    row count are sent, never rows. Features are compared one at a time;
    changes in relationships between features are not detected. The verdict is
    stored (last 100 per model).

    csv_path: recent data with the same columns as the training data.
    label: free-text tag stored with the check (a date, commit SHA, ...).
    thresholds: optional overrides, any of mean_shift (default 0.2, in baseline
        standard deviations), std_ratio (1.4), category_tvd (0.08),
        count_rel_change (0.2), new_category_mass (0.02), noise_sigmas (auto),
        min_samples (100). A feature is flagged only when a metric exceeds its
        threshold AND a sampling-noise allowance, so lowering a threshold below
        the allowance has no effect; lowering noise_sigmas raises sensitivity
        and false alarms.
    baseline_n_samples: rows the model's training schema was computed from, if
        known. Strongly recommended when the training set was small (a few
        hundred rows), otherwise false alarms are noticeably more frequent.
    align_to_model: summarize each column as the type the model's schema gives it
        (ordinal, count, ...) when its values fit, so hand-tuned types do not
        raise false type_changed findings. Values that do not fit are reported
        as a type change. Costs one extra request; if the schema cannot be
        fetched the check proceeds with types inferred from the CSV. Default True.

    Returns status ("pass" | "drift" | "insufficient_sample"), passed, summary
    counts, and per-feature findings with reasons. Drift is a normal result,
    not an error. A column whose values no longer fit the type the model was
    trained with shows up as type_changed. baseline_source says how the model's
    schema was produced when it was uploaded: "data", "hand_written" or "unknown".
    """
    if not os.path.isfile(csv_path):
        return {"error": True, "detail": f"File not found or not a file: {csv_path}"}
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {"error": True, "detail": "CSV has no rows"}
    baseline = await _baseline_features(model_id) if align_to_model else None
    arrays = [f["name"] for f in (baseline or []) if f.get("type") == "array"]
    if arrays:
        return {
            "error": True,
            "detail": (
                f"The model has array feature(s) {arrays} (embeddings or other vectors). A CSV cannot carry "
                "them, so this tool cannot check them and would report them as changed. Run the check with "
                "the proxyml SDK instead: proxyml.check_data_drift(model_id, df), where the DataFrame column "
                "holds the vectors. Only a summary of each array leaves your machine."
            ),
            "array_features": arrays,
        }
    summary = infer_schema_like(df, baseline) if baseline is not None else infer_schema(df)
    payload: dict = {"n_samples": int(len(df)), "features": summary["features"]}
    if label is not None:
        payload["label"] = label
    if thresholds:
        payload["thresholds"] = thresholds
    if baseline_n_samples is not None:
        payload["baseline_n_samples"] = baseline_n_samples
    async with _client() as c:
        r = await c.post(f"/surrogate/{model_id}/data-drift", content=json.dumps(payload))
        return _result(r)


@mcp.tool()
async def proxyml_list_data_drift_checks(model_id: str, limit: int = 20) -> dict:
    """List a model's recent data drift verdicts, newest first (summaries only).

    Use proxyml_get_data_drift_check for one verdict's per-feature findings.
    """
    async with _client() as c:
        r = await c.get(f"/surrogate/{model_id}/data-drift", params={"limit": limit})
        return _result(r)


@mcp.tool()
async def proxyml_get_data_drift_check(model_id: str, snapshot_id: str) -> dict:
    """Fetch one stored data drift verdict, including per-feature findings."""
    async with _client() as c:
        r = await c.get(f"/surrogate/{model_id}/data-drift/{snapshot_id}")
        return _result(r)


@mcp.tool()
async def proxyml_delete_data_drift_check(model_id: str, snapshot_id: str) -> dict:
    """Delete one stored data drift verdict, e.g. a test run you don't want in the history.

    Permanent. Use proxyml_list_data_drift_checks to find the snapshot_id.
    """
    async with _client() as c:
        r = await c.delete(f"/surrogate/{model_id}/data-drift/{snapshot_id}")
        if r.status_code == 204:
            return {"deleted": True, "model_id": model_id, "snapshot_id": snapshot_id}
        return _result(r)


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
    """Upload or replace a feature schema.

    An optional top-level n_samples (as returned by proxyml_infer_schema) records
    how many rows the statistics came from; data-drift checks use it by default.
    An optional top-level source, "data" (computed from real rows, unedited) or
    "hand_written", records where the statistics came from; it is reported as
    baseline_source on drift checks. Leave it out if unsure.
    """
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

    Array features (embeddings or other vectors) are never sent to the server: put null in an array feature's slot.
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

    Array features (embeddings or other vectors) are never sent to the server: put null in an array feature's slot.
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

    Array features (embeddings or other vectors) are never sent to the server: put null in an array feature's slot.
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

    Array features (embeddings or other vectors) are never sent to the server: put null in an array feature's slot.
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

    Array features (embeddings or other vectors) are never sent to the server: put null in an array feature's slot.
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

    Array features (embeddings or other vectors) are never sent to the server: put null in an array feature's slot.
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

    Array features (embeddings or other vectors) are never sent to the server: put null in an array feature's slot.
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
