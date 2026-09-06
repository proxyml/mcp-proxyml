# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Backfilled from git history 2026-09-06 — this project had no changelog until
now, so entries before that date are reconstructed from commit messages
rather than written at release time.

## [0.1.7] - 2026-08-02

### Added
- `proxyml_score_champion` now returns a `data_fingerprint` (a hash of
  `labels`) alongside its metrics. `proxyml_train_challenger` and
  `proxyml_score_champion` are two independent tool calls with no shared
  state, so there was previously no way to carry proxyml 0.8.0's
  champion/challenger data-consistency fingerprint through a manually
  merged upload when scoring is done via this decoupled path. Requires
  `proxyml[local]>=0.8.0`.

## [0.1.6] - 2026-08-01

### Changed
- `proxyml_train_challenger`/`run_challenger()` now delegate NaN-target
  handling to proxyml 0.6.0's `train_auto_challenger()` instead of a local
  workaround — rows with a missing target are dropped before training, the
  CV split, and champion scoring, and the champion is guaranteed to be
  evaluated on the same labeled population as the challenger (via a new
  `champion_predictions` param aligned to the same pre-drop rows).
  `champion_labels` is dropped from `proxyml_train_challenger`'s signature
  since champion labels are always `target_col` itself now; the response
  and `upload_payload` gain `n_samples_total`, `n_samples_dropped_unlabeled`,
  and `population_note`. Bumps the `challenger` extra's pin to
  `proxyml[local]>=0.6.0`.

## [0.1.5] - 2026-08-01

### Added
- Local challenger model training tools: `proxyml_train_challenger` and
  `proxyml_score_champion`, wrapping the SDK's `proxyml.local` module
  (`train_auto_challenger`, `score_champion`, `to_challenger_upload`) to
  train and score entirely in-process, returning an upload-ready JSON
  payload for the user to submit via the ProxyML dashboard themselves. The
  governance upload endpoints are intentionally not wired up directly —
  they require an interactive email-magic-link auth flow with no static
  token this server could use headlessly. Gated behind a new optional
  `challenger` extra (`proxyml[local]`) to keep scikit-learn/scipy out of
  the default install.

## [0.1.4] - 2026-05-25

### Changed
- `schema_name` is now a required argument on `proxyml_get_schema`,
  `proxyml_put_schema`, `proxyml_synthesize_data`, and
  `proxyml_train_surrogate` — dropped the `"default"` default, since it let
  an LLM silently route to the wrong schema with no error to recover from.
  `proxyml_synthesize_data` reorders `schema_name` to be the first
  parameter, since the remaining parameters have defaults and Python
  requires non-default arguments to precede default ones.

### Documentation
- Documented the `uv`/`uvx` prerequisite in the README's installation
  instructions.

## [0.1.3] - 2026-05-14

### Added
- 8 tools added to cover the full API surface that had gone undocumented
  and unimplemented in the server: `proxyml_list_schemas`,
  `proxyml_delete_schema`, `proxyml_get_model_schema`,
  `proxyml_update_surrogate`, `proxyml_delete_surrogate`, `proxyml_predict`,
  `proxyml_find_counterfactuals`, and `proxyml_health_check` (the last
  bypasses the authenticated client entirely — no auth header, no quota
  charge).
- Security & permissions section in the README, documenting what the
  server can access and what is sent to ProxyML's servers.

### Fixed
- Raised the client timeout from 60s to 120s to match the server-side
  training budget.
- Replaced bare `raise_for_status()` calls throughout with the `_result()`
  helper, so a failed request returns a structured
  `{"error": True, "status_code": ..., "detail": ...}` dict — surfacing the
  API's own error detail (training-timeout codes, validation messages,
  etc.) to the calling agent instead of an unhandled `httpx` exception.
- `proxyml_infer_schema` now checks that `csv_path` exists before handing
  it to pandas, instead of letting pandas raise its own less-actionable
  error.
- `schema.py`'s schema inference now warns when `immutable_cols` contains
  a name absent from the DataFrame.

## [0.1.2] - 2026-05-11

### Fixed
- Corrected the license and author email in `pyproject.toml`.

## [0.1.1] - 2026-05-11

### Added
- PyPI packaging metadata: readme, license, authors, keywords, classifiers,
  and project URLs.
- `proxyml_predict_batch` tool, plus a "dev model validation without
  production data" workflow example in the README.
- `proxyml_get_usage`, `proxyml_export_surrogate`, and
  `proxyml_explain_local_batch` tools.
- `proxyml_detect_drift` tool (wraps `explain/diff` with coefficient and
  fidelity thresholds for a structured CI/CD pass/fail), plus agentic
  workflow documentation in the README.
- Schema inference tests.
- Drift-interpretation logic extracted into its own module with its own
  test coverage, plus two more README workflow examples.
- MIT license.
- Claude Code `mcp add-json` install instructions in the README; removed a
  stray `PROXYML_BASE_URL` reference that didn't apply to the documented
  install path.

## [0.1.0] - 2026-05-11

### Added
- Initial MCP server scaffold for ProxyML: `proxyml_infer_schema` (local
  CSV → feature schema, no data sent to the server), `proxyml_get_schema`,
  `proxyml_put_schema`, `proxyml_synthesize_data`, `proxyml_train_surrogate`,
  `proxyml_list_surrogates`, `proxyml_get_summary`, `proxyml_explain_local`,
  `proxyml_find_counterfactual`, and `proxyml_diff_models`.
- Initial README with setup and usage instructions.
