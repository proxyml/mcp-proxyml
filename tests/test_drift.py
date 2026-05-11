import pytest
from mcp_proxyml.drift import interpret_diff

VA = "ver-a"
VB = "ver-b"
COEFF_T = 0.1
FIDELITY_T = 0.05


def _diff(coefficient_diff=None, metric_diff=None, features_added=None, features_removed=None):
    return {
        "coefficient_diff": coefficient_diff or [],
        "metric_diff": metric_diff or {},
        "features_added": features_added or [],
        "features_removed": features_removed or [],
    }


def _coeff(feature, a, b):
    return {"feature": feature, "a": a, "b": b, "delta": b - a}


def _metric(a, b):
    return {"a": a, "b": b, "delta": b - a}


# ---------------------------------------------------------------------------
# Pass cases
# ---------------------------------------------------------------------------

def test_passes_with_no_changes():
    result = interpret_diff(_diff(), VA, VB, COEFF_T, FIDELITY_T)
    assert result["passed"] is True
    assert result["summary"].startswith("PASSED")


def test_passes_when_deltas_within_threshold():
    diff = _diff(coefficient_diff=[_coeff("age", 0.5, 0.55)])  # delta=0.05 < 0.1
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert result["passed"] is True


def test_passes_when_metric_improves():
    diff = _diff(metric_diff={"r2": _metric(0.8, 0.85)})  # delta=+0.05, improvement
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert result["passed"] is True


def test_passes_when_metric_drops_within_fidelity_threshold():
    diff = _diff(metric_diff={"r2": _metric(0.8, 0.76)})  # delta=-0.04 < 0.05
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# Fail cases
# ---------------------------------------------------------------------------

def test_fails_on_large_coefficient_shift():
    diff = _diff(coefficient_diff=[_coeff("income", 0.3, 0.45)])  # delta=0.15 > 0.1
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert result["passed"] is False
    assert result["summary"].startswith("FAILED")


def test_fails_on_fidelity_drop():
    diff = _diff(metric_diff={"r2": _metric(0.85, 0.78)})  # delta=-0.07 > 0.05
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert result["passed"] is False


def test_fails_on_features_added():
    diff = _diff(features_added=["new_feature"])
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert result["passed"] is False
    assert "new_feature" in result["summary"]


def test_fails_on_features_removed():
    diff = _diff(features_removed=["old_feature"])
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert result["passed"] is False
    assert "old_feature" in result["summary"]


# ---------------------------------------------------------------------------
# Flagged features
# ---------------------------------------------------------------------------

def test_flagged_features_contains_only_breaching_features():
    diff = _diff(coefficient_diff=[
        _coeff("age", 0.5, 0.55),    # delta=0.05, within threshold
        _coeff("income", 0.3, 0.45), # delta=0.15, breaches
    ])
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert len(result["flagged_features"]) == 1
    assert result["flagged_features"][0]["feature"] == "income"


def test_exactly_at_threshold_does_not_flag():
    diff = _diff(coefficient_diff=[_coeff("age", 0.0, 0.1)])  # delta == threshold
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert result["passed"] is True
    assert result["flagged_features"] == []


def test_negative_delta_beyond_threshold_is_flagged():
    diff = _diff(coefficient_diff=[_coeff("age", 0.5, 0.35)])  # delta=-0.15
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert result["passed"] is False
    assert len(result["flagged_features"]) == 1


# ---------------------------------------------------------------------------
# Metric changes
# ---------------------------------------------------------------------------

def test_metric_changes_includes_flagged_key():
    diff = _diff(metric_diff={"f1": _metric(0.9, 0.82)})
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert "flagged" in result["metric_changes"]["f1"]
    assert result["metric_changes"]["f1"]["flagged"] is True


def test_metric_changes_not_flagged_when_within_threshold():
    diff = _diff(metric_diff={"f1": _metric(0.9, 0.86)})  # delta=-0.04
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert result["metric_changes"]["f1"]["flagged"] is False


# ---------------------------------------------------------------------------
# Summary content
# ---------------------------------------------------------------------------

def test_passed_summary_includes_largest_shift():
    diff = _diff(coefficient_diff=[_coeff("age", 0.5, 0.55)])
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert "0.050" in result["summary"]


def test_failed_summary_names_flagged_features():
    diff = _diff(coefficient_diff=[_coeff("income", 0.3, 0.5)])
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert "income" in result["summary"]


def test_failed_summary_with_more_than_three_features():
    diff = _diff(coefficient_diff=[_coeff(f"f{i}", 0.0, 0.5) for i in range(5)])
    result = interpret_diff(diff, VA, VB, COEFF_T, FIDELITY_T)
    assert "and more" in result["summary"]


# ---------------------------------------------------------------------------
# Version IDs passed through
# ---------------------------------------------------------------------------

def test_version_ids_in_result():
    result = interpret_diff(_diff(), "v1", "v2", COEFF_T, FIDELITY_T)
    assert result["version_a"] == "v1"
    assert result["version_b"] == "v2"
