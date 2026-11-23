import asyncio

import httpx
import pytest

from mcp_proxyml import server
from mcp_proxyml.server import (
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
