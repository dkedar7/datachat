# DataChat

**Chat with any dataset — a data-analyst agent that is a full complement to the input form: it drives the app *and* plots into it. Built on [Fast Dash](https://github.com/dkedar7/fast_dash).**

Upload a CSV/Excel file (or paste a link to one), then ask the **Data Analyst** in the left sidebar: *"what drives sales?"*, *"plot tip vs total bill by day"*, *"show the correlation heatmap"*. The agent both **operates the app's form** — *"set dataset_url to &lt;link&gt; and run the app"* stages the input and re-runs the deterministic app — and has **full control of the app's three output panels**: it renders a **chart** into the Chart panel, a **computed table** into the Data Preview panel, and a **written analysis** into the Summary panel, and it can **rearrange the panels** into any layout. Everything lands in the app's panels, not just inline in the chat.

![DataChat — a dataset loaded, with the Data Analyst plotting a chart from it](docs/hero.png)

## How it works

- **Upload / link any dataset** → a normal Fast Dash app decodes it to a DataFrame and shows a **preview**, a default **overview chart**, and a **summary** (mosaic `AC\nBB`: preview and summary on top, the chart full-width below).
- **The Data Analyst sidecar** is a **LangGraph + OpenRouter** agent that is a full complement to the form and owns the three output panels (A = Data Preview, B = Chart, C = Summary). Its tools:
  - `run_analysis(code)` — runs pandas/plotly in the sandbox and routes each artifact to its panel: a produced **figure → Chart panel (B)** and a produced **table → Data Preview panel (A)** (each a `set_output` frame). Nothing renders inline in the chat.
  - `write_summary(markdown)` — puts a **written analysis into the Summary panel (C)** (a `set_output` frame).
  - `arrange_panels(mosaic)` — **rearranges the panels** into any valid mosaic over A/B/C, e.g. `AC\nBB` (default), `B` (chart full-screen), `A\nB\nC` (stacked) (a `set_layout` frame).
  - `set_app_input(name, value)` + `run_app()` — **drive the form**: stage an input (e.g. `dataset_url`) and re-run the deterministic app. (File uploads can't be set from chat — use the Upload button.)
- **Frames, not return values** — the tools call `fast_dash.agent_tools.emit_frame(...)`; Fast Dash's turn runner drains and dispatches `set_input` / `run_app` / `set_output` / `set_layout` frames per session (gated by the app's `chat_tools` allowlist), so the chat can change the app's real inputs and outputs — a full complement (or replacement) for the form. A `set_output` frame carries a raw value (a plotly dict, a DataFrame, or a markdown string) that Fast Dash transforms through the target panel's component server-side.

The analyst runs on **Fast Dash 0.6.1**'s built-ins: the code runs in fast_dash's built-in sandbox (`fast_dash.sandbox.run_code`, originally upstreamed from this app), and the drive/render frames go through fast_dash's agent-tools frame bus and per-panel output transforms — so DataChat ships no sandbox or dispatch of its own.

## Safety — the code runs sandboxed

LLM-written code executes in a **separate subprocess** (Fast Dash's built-in sandbox) with:
- a **scrubbed environment** (no API keys / tokens),
- **network disabled** (analysis runs on the provided DataFrame only),
- a **wall-clock timeout** and CPU/memory limits.

The DataFrame is injected into the child; it returns only a figure (JSON), a table, printed output, or an error.

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
| `app.py` | The preview/chart/summary app (outputs `[Table, Graph, Markdown]`, mosaic `AC\nBB`) + the Data Analyst sidecar (drive + render + layout tools allowlisted) |
| `analyst/agent.py` | LangGraph agent: `run_analysis` (figure → Chart panel B, table → Data Preview panel A), `write_summary` (markdown → Summary panel C), `arrange_panels` (re-mosaic A/B/C), `set_app_input` / `run_app` (drive the form); all emit fast_dash frames |
| `analyst/data.py` | Upload/URL loading, per-session DataFrame cache, and the default overview chart |

## Stack

Fast Dash (chat sidecar + built-in sandbox + agent-tools frame bus + typed rendering) · LangGraph · LangChain (OpenRouter) · pandas · plotly.

## License

MIT — see [LICENSE](LICENSE).
