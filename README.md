# mcp-proxyml

MCP server for the [ProxyML](https://proxyml.ai) API. Gives Claude (and other MCP clients) direct access to ProxyML's surrogate modelling and explainability tools.

## Prerequisites

A ProxyML API key. Sign up at [proxyml.ai](https://proxyml.ai).

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

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `PROXYML_API_KEY` | Yes | Your ProxyML API key |

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

### Governance report

Claude can generate a governance report from existing tools without a dedicated endpoint. Example prompt:

```
Using ProxyML, generate a governance report for surrogate version <id>.
Include: task type, training date, fidelity metrics, top 5 features by importance,
any warnings from training, and a plain-English summary of what drives predictions.
Format it as a structured document suitable for attaching to a deployment ticket.
```

Claude will call `proxyml_get_summary` (and `proxyml_list_surrogates` to find metadata) and compose the report.
