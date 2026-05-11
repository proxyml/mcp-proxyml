def interpret_diff(
    diff: dict,
    version_a: str,
    version_b: str,
    coefficient_threshold: float,
    fidelity_threshold: float,
) -> dict:
    """Apply drift thresholds to a diff_models response and return a pass/fail result.

    Pure function — no I/O. Called by proxyml_detect_drift after fetching the diff.
    """
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

    passed = (
        not flagged_features
        and not any(v["flagged"] for v in metric_changes.values())
        and not features_added
        and not features_removed
    )

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
