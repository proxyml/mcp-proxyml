import asyncio

import httpx
import pytest

from mcp_proxyml import server
from mcp_proxyml.server import (
    proxyml_check_data_drift,
    proxyml_delete_data_drift_check,
    proxyml_get_data_drift_check,
    proxyml_list_data_drift_checks,
    proxyml_get_challenger_summary,
    proxyml_score_champion,
    proxyml_upload_challenger,
)


def test_score_champion_includes_data_fingerprint():
    from proxyml.local import fingerprint_labels

    labels = [1, 0, 1, 1, 0]
    predictions = [1, 0, 1, 0, 0]

    result = asyncio.run(proxyml_score_champion(labels, predictions, task="classification"))

    assert "error" not in result
    assert result["data_fingerprint"] == fingerprint_labels(labels)


def test_score_champion_data_fingerprint_differs_for_different_labels():
    predictions = [1, 0, 1, 0, 0]

    result_a = asyncio.run(
        proxyml_score_champion([1, 0, 1, 1, 0], predictions, task="classification")
    )
    result_b = asyncio.run(
        proxyml_score_champion([0, 0, 1, 1, 0], predictions, task="classification")
    )

    assert result_a["data_fingerprint"] != result_b["data_fingerprint"]


def test_score_champion_rejects_invalid_task():
    result = asyncio.run(proxyml_score_champion([1, 0], [1, 0], task="bogus"))
    assert result["error"] is True
    assert "data_fingerprint" not in result


def test_score_champion_rejects_mismatched_lengths():
    result = asyncio.run(proxyml_score_champion([1, 0, 1], [1, 0], task="classification"))
    assert result["error"] is True
    assert "data_fingerprint" not in result


def _mock_ci(monkeypatch, handler):
    """Route the server's httpx clients through a mock transport."""
    real = httpx.AsyncClient

    def factory(**kwargs):
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(server.httpx, "AsyncClient", factory)


def test_upload_challenger_sends_ci_token_not_api_key(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["headers"] = request.headers
        seen["body"] = request.content
        return httpx.Response(202, json={"challenger_version": "v1"})

    monkeypatch.setenv("PROXYML_API_KEY", "account-key")
    monkeypatch.setenv("PROXYML_CI_TOKEN", "ci-token")
    _mock_ci(monkeypatch, handler)

    result = asyncio.run(proxyml_upload_challenger({"a": 1}))

    assert result == {"challenger_version": "v1"}
    assert seen["path"].endswith("/ci/challenger")
    assert seen["headers"]["x-ci-token"] == "ci-token"
    assert "x-api-key" not in seen["headers"]
    assert seen["body"] == b'{"a": 1}'


def test_upload_challenger_explicit_token_overrides_env(monkeypatch):
    seen = {}

    def handler(request):
        seen["token"] = request.headers["x-ci-token"]
        return httpx.Response(202, json={})

    monkeypatch.setenv("PROXYML_CI_TOKEN", "from-env")
    _mock_ci(monkeypatch, handler)
    asyncio.run(proxyml_upload_challenger({}, ci_token="explicit"))
    assert seen["token"] == "explicit"


def test_upload_challenger_without_token_returns_error(monkeypatch):
    monkeypatch.delenv("PROXYML_CI_TOKEN", raising=False)
    result = asyncio.run(proxyml_upload_challenger({}))
    assert result["error"] is True
    assert "PROXYML_CI_TOKEN" in result["detail"]


def test_upload_challenger_surfaces_server_error(monkeypatch):
    monkeypatch.setenv("PROXYML_CI_TOKEN", "t")
    _mock_ci(
        monkeypatch,
        lambda request: httpx.Response(400, json={"detail": "fingerprint mismatch"}),
    )
    result = asyncio.run(proxyml_upload_challenger({}))
    assert result == {"error": True, "status_code": 400, "detail": "fingerprint mismatch"}


def test_get_challenger_summary(monkeypatch):
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"winner": "challenger"})

    monkeypatch.setenv("PROXYML_CI_TOKEN", "t")
    _mock_ci(monkeypatch, handler)
    result = asyncio.run(proxyml_get_challenger_summary())
    assert result == {"winner": "challenger"}
    assert seen["method"] == "GET"
    assert seen["path"].endswith("/ci/challenger/summary")


def test_get_challenger_summary_without_token_returns_error(monkeypatch):
    monkeypatch.delenv("PROXYML_CI_TOKEN", raising=False)
    result = asyncio.run(proxyml_get_challenger_summary())
    assert result["error"] is True


def test_check_data_drift_sends_statistics_only(monkeypatch, tmp_path):
    import json

    csv = tmp_path / "recent.csv"
    csv.write_text("amount,city\n1.5,Paris\n2.5,Rome\n3.5,Paris\n")
    seen = {}

    def handler(request):
        seen["method"], seen["path"] = request.method, request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "pass", "passed": True})

    monkeypatch.setenv("PROXYML_API_KEY", "k")
    _mock_ci(monkeypatch, handler)
    result = asyncio.run(
        proxyml_check_data_drift("m1", str(csv), label="nightly", thresholds={"mean_shift": 0.5}, baseline_n_samples=900)
    )
    assert result == {"status": "pass", "passed": True}
    assert seen["method"] == "POST" and seen["path"].endswith("/surrogate/m1/data-drift")
    body = seen["body"]
    assert body["n_samples"] == 3
    assert [f["name"] for f in body["features"]] == ["amount", "city"]
    assert body["label"] == "nightly" and body["thresholds"] == {"mean_shift": 0.5}
    assert body["baseline_n_samples"] == 900
    assert "_note" not in body
    assert "1.5" not in json.dumps(body["features"][1])  # no raw values


def test_check_data_drift_missing_file_and_empty_csv(monkeypatch, tmp_path):
    monkeypatch.setenv("PROXYML_API_KEY", "k")
    assert asyncio.run(proxyml_check_data_drift("m1", str(tmp_path / "nope.csv")))["error"] is True
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n")
    assert asyncio.run(proxyml_check_data_drift("m1", str(empty)))["error"] is True


def test_check_data_drift_unknown_model_returns_error(monkeypatch, tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("a\n1.5\n2.5\n")
    monkeypatch.setenv("PROXYML_API_KEY", "k")
    _mock_ci(monkeypatch, lambda request: httpx.Response(404, json={"detail": "Model not found"}))
    result = asyncio.run(proxyml_check_data_drift("nope", str(csv)))
    assert result == {"error": True, "status_code": 404, "detail": "Model not found"}


def test_list_and_get_data_drift_checks(monkeypatch):
    seen = []
    monkeypatch.setenv("PROXYML_API_KEY", "k")
    _mock_ci(monkeypatch, lambda request: (seen.append((request.url.path, dict(request.url.params))), httpx.Response(200, json={"ok": 1}))[1])
    assert asyncio.run(proxyml_list_data_drift_checks("m1", limit=5)) == {"ok": 1}
    assert asyncio.run(proxyml_get_data_drift_check("m1", "s1")) == {"ok": 1}
    assert seen[0][0].endswith("/surrogate/m1/data-drift") and seen[0][1] == {"limit": "5"}
    assert seen[1][0].endswith("/surrogate/m1/data-drift/s1")


def test_infer_schema_reports_row_count_and_put_schema_forwards_it(monkeypatch, tmp_path):
    import json

    from mcp_proxyml.server import proxyml_infer_schema, proxyml_put_schema

    csv = tmp_path / "t.csv"
    csv.write_text("a,b\n1.5,x\n2.5,y\n3.5,x\n")
    schema = asyncio.run(proxyml_infer_schema(str(csv)))
    assert schema["n_samples"] == 3

    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"features": []})

    monkeypatch.setenv("PROXYML_API_KEY", "k")
    _mock_ci(monkeypatch, handler)
    asyncio.run(proxyml_put_schema(schema, "s"))
    assert seen["body"]["n_samples"] == 3


def test_delete_data_drift_check(monkeypatch):
    seen = {}

    def handler(request):
        seen["method"], seen["path"] = request.method, request.url.path
        return httpx.Response(204)

    monkeypatch.setenv("PROXYML_API_KEY", "k")
    _mock_ci(monkeypatch, handler)
    assert asyncio.run(proxyml_delete_data_drift_check("m1", "s1")) == {"deleted": True, "model_id": "m1", "snapshot_id": "s1"}
    assert seen["method"] == "DELETE" and seen["path"].endswith("/surrogate/m1/data-drift/s1")


def test_delete_data_drift_check_not_found_returns_error(monkeypatch):
    monkeypatch.setenv("PROXYML_API_KEY", "k")
    _mock_ci(monkeypatch, lambda request: httpx.Response(404, json={"detail": "Drift check not found"}))
    assert asyncio.run(proxyml_delete_data_drift_check("m1", "zz")) == {
        "error": True, "status_code": 404, "detail": "Drift check not found",
    }
