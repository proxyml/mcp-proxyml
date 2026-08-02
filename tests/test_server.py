import asyncio

from mcp_proxyml.server import proxyml_score_champion


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
