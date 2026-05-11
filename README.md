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
| `proxyml_explain_local` | Per-feature contribution breakdown for a single instance |
| `proxyml_find_counterfactual` | Find the nearest point that flips the prediction |
| `proxyml_diff_models` | Compare feature importances between two surrogate versions |

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
