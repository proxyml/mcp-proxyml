# mcp-proxyml

MCP server for the [ProxyML](https://proxyml.ai) API. Gives Claude (and other MCP clients) direct access to ProxyML's surrogate modelling and explainability tools.

## Prerequisites

- A ProxyML API key — sign up at [proxyml.ai](https://proxyml.ai)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (required for `uvx` installation method)

```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or if you use `pip`:

```pip install uv```

## Installation

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "proxyml": {
      "command": "uvx",
      "args": ["mcp-proxyml"],
      "env": {
        "PROXYML_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

The config file is at:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Then restart Claude Desktop.

### Claude Code

```bash
claude mcp add-json proxyml '{"command":"uvx","args":["mcp-proxyml"],"env":{"PROXYML_API_KEY":"your-api-key-here"}}'
```

### Other MCP clients

```bash
pip install mcp-proxyml
PROXYML_API_KEY=your-key mcp-proxyml
```

### Local challenger training (optional)

`proxyml_train_challenger` and `proxyml_score_champion` train a small model
locally via the `proxyml` SDK and require the `challenger` extra, which
pulls in scikit-learn and scipy:

```bash
pip install 'mcp-proxyml[challenger]'
```

Without it, those two tools return a clear error telling you to install the
extra rather than failing to start the server.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `PROXYML_API_KEY` | Yes | Your ProxyML API key |

## Security & permissions

### What this server can access

**Filesystem** — `proxyml_infer_schema` and `proxyml_train_challenger` each
read one CSV file at the path you provide. No other tool reads or writes
local files.

**Network** — outbound HTTPS only, to `https://api.proxyml.ai` (or the URL
set in `PROXYML_BASE_URL`). No other hosts are contacted.

**Environment** — only `PROXYML_API_KEY` and `PROXYML_BASE_URL` are read.
No other environment variables are accessed.

**No shell execution, no local ports, no other system resources.**

### What is sent to ProxyML's servers

| Tool | What is transmitted |
|---|---|
| `proxyml_infer_schema` | Nothing — CSV is read and processed locally only |
| `proxyml_train_challenger` | Nothing — training happens entirely locally; the returned payload is only sent if you choose to upload it yourself |
| `proxyml_score_champion` | Nothing — scoring happens entirely locally |
| `proxyml_put_schema` | Feature schema: column names, types, and constraints |
| `proxyml_get_schema` | Schema name (string) |
| `proxyml_synthesize_data` | Schema name, sample count, and optionally one feature vector |
| `proxyml_train_surrogate` | Synthetic feature vectors + your model's predictions for those vectors |
| `proxyml_predict_batch` | Feature vectors you pass explicitly |
| `proxyml_explain_local` / `proxyml_explain_local_batch` | Feature vector(s) you pass explicitly |
| `proxyml_find_counterfactual` | One feature vector and a target label |
| `proxyml_list_surrogates` / `proxyml_get_summary` / `proxyml_export_surrogate` / `proxyml_diff_models` / `proxyml_detect_drift` | Version IDs and/or threshold values (no feature data) |
| `proxyml_get_usage` | Nothing |

The typical workflow is designed so that raw training data never leaves your
environment. `proxyml_infer_schema` derives column metadata from your CSV
locally; `proxyml_synthesize_data` generates samples server-side from that
schema; and `proxyml_train_surrogate` sends only those synthetic samples
plus your model's predictions for them. Your original dataset is not
transmitted.

If you use `proxyml_explain_local`, `proxyml_predict_batch`, or
`proxyml_find_counterfactual` with real data points rather than synthetic
ones, those feature vectors will be sent to the ProxyML API.

`proxyml_train_challenger` and `proxyml_score_champion` never call the
ProxyML API at all — they run the SDK's local training/scoring code
in-process and hand back a JSON payload. This server does not upload that
payload for you; save it and use the ProxyML dashboard's "Upload
challenger" button, or POST it yourself to
`/app/projects/{project_id}/challenger`.

### Authentication

Your API key is sent on every request as an `X-API-KEY` HTTP header over TLS.
It is read from the `PROXYML_API_KEY` environment variable and is never
logged or written to disk by this server.

## Tools

### Schema

| Tool | Description |
|---|---|
| `proxyml_infer_schema` | Infer a feature schema from a local CSV file — no data sent to the server |
| `proxyml_get_schema` | Retrieve a stored schema by name |
| `proxyml_put_schema` | Upload or replace a feature schema |

### Training

| Tool | Description |
|---|---|
| `proxyml_synthesize_data` | Generate synthetic samples from the stored schema |
| `proxyml_train_surrogate` | Train a linear surrogate on samples scored by your model |
| `proxyml_list_surrogates` | List trained surrogate models, newest first |
| `proxyml_predict_batch` | Get surrogate predictions for a list of instances |

### Local challenger training

Requires the `challenger` extra (`pip install 'mcp-proxyml[challenger]'`).

| Tool | Description |
|---|---|
| `proxyml_train_challenger` | Train a challenger model locally on a CSV and assemble an upload-ready payload |
| `proxyml_score_champion` | Score a champion model's predictions locally, for an apples-to-apples comparison |

### Explainability

| Tool | Description |
|---|---|
| `proxyml_get_summary` | Feature importances and model summary |
| `proxyml_export_surrogate` | Full coefficient export for audit and governance |
| `proxyml_explain_local` | Per-feature contribution breakdown for a single instance |
| `proxyml_explain_local_batch` | Per-feature contributions for multiple instances in one call |
| `proxyml_find_counterfactual` | Find the nearest point that flips the prediction |
| `proxyml_diff_models` | Compare feature importances between two surrogate versions |

### CI/CD

| Tool | Description |
|---|---|
| `proxyml_detect_drift` | Compare two versions and return a structured pass/fail against coefficient and fidelity thresholds |

### Account

| Tool | Description |
|---|---|
| `proxyml_get_usage` | Current tier, request count, and quota — useful as a pre-flight check |

## Typical workflow

```
1. proxyml_infer_schema      — point at a CSV, get a schema back
2. proxyml_put_schema        — upload it
3. proxyml_synthesize_data   — generate synthetic samples
4. [score samples with your model]
5. proxyml_train_surrogate   — send samples + predictions, get a surrogate
6. proxyml_get_summary       — see which features drive predictions
7. proxyml_explain_local     — explain a specific decision
8. proxyml_find_counterfactual — find what would need to change
```

Steps 1–2 are one-time setup. Steps 3–5 can be repeated to retrain as your model changes; use `proxyml_diff_models` to compare versions.

## Agentic workflows

### Drift detection in CI/CD

`proxyml_detect_drift` is designed for use in deployment pipelines. It wraps `proxyml_diff_models` and applies thresholds to produce a structured pass/fail:

```
On model deployment:
1. proxyml_train_surrogate          — train surrogate on new model version
2. proxyml_detect_drift(a, b)       — compare against previous version
   → passed: false                  — block deployment or flag for review
   → passed: true                   — proceed
```

Thresholds can be tuned per use case:

```
proxyml_detect_drift(
  version_a="<previous>",
  version_b="<new>",
  coefficient_threshold=0.15,   # tighter for high-stakes models
  fidelity_threshold=0.02
)
```

### Dev model validation without production data

Validate a model trained in a lower environment by comparing its predictions
against a surrogate trained on production data — no production data required
in the dev environment.

This workflow requires a step the MCP server can't do on its own (scoring
with your dev model), but works naturally in Claude Code where the agent can
execute code directly:

```
1. proxyml_synthesize_data(num_points=100)   → synthetic samples
2. [agent runs: dev_predictions = dev_model.predict(samples)]
3. proxyml_predict_batch(samples)            → surrogate predictions
4. [agent computes MAE and compares to tolerance]
```

Example prompt for Claude Code:

```
Using ProxyML, validate my dev model against the production surrogate.
Synthesize 100 samples from the "default" schema, score them with my model
at dev_model.predict(), get surrogate predictions with proxyml_predict_batch,
then compute the mean absolute error and tell me whether it's within 0.1.
```

The surrogate acts as a proxy for production behaviour — if the dev model
agrees with it within tolerance, it's likely behaving consistently with what
was trained on real data.

### Counterfactual investigation

When a model makes a decision that needs explaining — a rejected loan application,
a flagged transaction, a declined insurance quote — chain `proxyml_explain_local`
and `proxyml_find_counterfactual` to answer both "why?" and "what would need to
change?":

```
1. proxyml_explain_local(instance)              → which features drove this decision
2. proxyml_find_counterfactual(instance, target) → nearest point that flips it
```

Example prompt:

```
My model rejected this application: [age=34, income=42000, loan_amount=15000, ...].
Using ProxyML, explain why it was rejected and find the minimum changes that
would result in an approval. Highlight which changes are realistic given that
age is immutable.
```

Claude will call `proxyml_explain_local` to surface the top contributing features,
then `proxyml_find_counterfactual` with the target outcome, and interpret the
difference in plain language.

### Iterative surrogate improvement

When `proxyml_train_surrogate` returns a low fidelity warning or other training
diagnostic, the agent can use it to guide the next iteration rather than stopping:

```
1. proxyml_train_surrogate(samples, predictions)
   → warning: "Surrogate fidelity is low (R²=0.52)..."
2. proxyml_synthesize_data(num_points=500)       — increase sample count
3. [re-score with model]
4. proxyml_train_surrogate(larger_samples, predictions)
5. proxyml_detect_drift(v1, v2)                  — confirm improvement, not regression
```

Example prompt:

```
Train a surrogate for my regression model using the "default" schema with 200
samples. If fidelity is below 0.7, keep doubling the sample count and retraining
until it passes or you reach 1600 samples. Use proxyml_detect_drift after each
retrain to confirm the model is improving rather than just changing.
```

The training warnings (convergence, sparsity, class imbalance, high correlation)
are designed to be actionable — the agent can read them and decide whether to
adjust `num_samples`, revisit the schema, or flag for human review.

### Governance report

Claude can generate a governance report from existing tools without a dedicated endpoint. Example prompt:

```
Using ProxyML, generate a governance report for surrogate version <id>.
Include: task type, training date, fidelity metrics, top 5 features by importance,
any warnings from training, and a plain-English summary of what drives predictions.
Format it as a structured document suitable for attaching to a deployment ticket.
```

Claude will call `proxyml_get_summary` (and `proxyml_list_surrogates` to find metadata) and compose the report.
