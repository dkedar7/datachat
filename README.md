# DataChat

**Chat with any dataset — a data-analyst agent that writes and runs code to plot your answers. Built on [Fast Dash](https://github.com/dkedar7/fast_dash).**

Upload a CSV/Excel file (or paste a link to one), then open the **Data Analyst** and ask: *"what drives sales?"*, *"plot tip vs total bill by day"*, *"show the correlation heatmap"*. The agent writes pandas/plotly code, runs it in a sandbox, and the resulting **charts and tables render inline in the chat**.

<!-- ![DataChat](docs/hero.png) -->

## How it works

- **Upload / link any dataset** → a normal Fast Dash app input decodes it to a DataFrame and shows a preview + summary.
- **The Data Analyst sidecar** is a **LangGraph + OpenRouter** agent with one tool: `run_python(code)`. It sees the schema, writes analysis code, and gets back the figure/table/output.
- **Charts render inline** — the agent's plotly figures and pandas tables are yielded as Fast Dash `artifact` frames, so they appear as real interactive charts and sortable tables in the conversation.

## Safety — the code runs sandboxed

LLM-written code executes in a **separate subprocess** with:
- a **scrubbed environment** (no API keys / tokens — verified by test),
- **network disabled** (analysis runs on the provided DataFrame only),
- a **wall-clock timeout** and CPU/memory limits.

The DataFrame is handed over as parquet; the child returns only a figure (JSON), a table, printed output, or an error.

## Run locally

```bash
uv sync
export OPENROUTER_API_KEY=...        # https://openrouter.ai/keys
uv run python -m app                 # http://127.0.0.1:8080
```

Optional: `DATACHAT_MODEL` (default `anthropic/claude-haiku-4.5`), `DATACHAT_MAX_ROWS`.

## Deploy

Served by gunicorn (gthread, single worker) so chat history and per-session data live in one process. See the `Dockerfile`; set `OPENROUTER_API_KEY` as a secret.

## Architecture

| File | Role |
| ---- | ---- |
| `app.py` | The upload/preview app + the Data Analyst sidecar |
| `analyst/agent.py` | LangGraph agent, `run_python` tool, artifacts → chat |
| `analyst/sandbox.py` + `_runner.py` | The subprocess code sandbox |
| `analyst/data.py` | Upload/URL loading + per-session DataFrame cache |

## Stack

Fast Dash (chat sidecar + artifacts) · LangGraph · LangChain (OpenRouter) · pandas · plotly.

## License

MIT — see [LICENSE](LICENSE).
