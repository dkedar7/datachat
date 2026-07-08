"""The Data Analyst -- a LangGraph agent that drives the app and renders into it.

Wired into Fast Dash as a ``chat=`` agent, it is a full complement (or
replacement) for the input form. It has FULL control of the app's three output
panels AND their layout, plus it can drive the inputs. It can:

* **drive the form** -- ``set_app_input("dataset_url", <url>)`` then ``run_app()``
  set an input and re-run the deterministic app (files still use the Upload
  button; a file input can't be set from chat);
* **render into ALL THREE output panels** -- ``run_analysis`` runs pandas/plotly
  code in fast_dash's built-in subprocess sandbox and routes a produced *figure*
  into the **Chart** panel (B) and a produced *table* into the **Data Preview**
  panel (A); ``write_summary`` puts a written markdown analysis into the
  **Summary** panel (C). Charts / tables / summaries land in the app's panels,
  NOT inline in the chat;
* **rearrange the panels** -- ``arrange_panels(mosaic)`` sets ANY valid mosaic
  over the three panels (A/B/C).

Tools emit frames through ``fast_dash.agent_tools.emit_frame``; fast_dash's turn
runner drains and dispatches them during the turn (the same mechanism fast_dash's
own toolkit uses). A ``set_output`` frame carries a raw Python value (a plotly
dict, a DataFrame, or a markdown string) that fast_dash transforms through the
TARGET slot's component pipeline server-side. Provider: OpenRouter.
"""

from __future__ import annotations

import json

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

import fast_dash.sandbox
from fast_dash.adapters.langstage import build_chat_callback
from fast_dash.agent_tools import emit_frame

from . import data
from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, has_llm
from .data import current_dataframe, schema_text

UPLOAD_ID, URL_ID = "dataset_file", "dataset_url"

# The output slots the analyst renders into. Outputs are ``[Table, Graph,
# Markdown]`` and mosaic slot letters are assigned by output order, so:
#   A -> Table    (output index 0, "Data preview")
#   B -> Graph    (output index 1, "Chart")
#   C -> Markdown (output index 2, "Summary")
# A ``set_output`` frame carries a raw value that fast_dash transforms through
# that slot's component: a plotly figure DICT -> Graph.figure; a pandas
# DataFrame -> Table.data (to_dict("records")); a markdown STRING ->
# Markdown.children. Verified against a real FastDash instance's
# ``_resolve_slot`` / ``_sidecar_set_output`` (see test_agent.py).
PREVIEW_SLOT = "A"   # Data Preview  (Table)
CHART_SLOT = "B"     # Chart         (Graph)
SUMMARY_SLOT = "C"   # Summary       (Markdown)

SYSTEM_PROMPT = """You are a data analyst embedded in the DataChat app. You have \
FULL control of the app's three output panels AND their layout, and you can drive \
the app's inputs. You are a full complement to (or replacement for) the form.

The app has THREE output panels you own:
- A = Data Preview (a table)
- B = Chart (a plotly figure)
- C = Summary (written markdown)

Ways to act:

1. RENDER INTO THE PANELS via `run_analysis(code)`. Write pandas/plotly code; \
what you produce is routed to the right panel (NOT shown inline in the chat):
- `pd`, `np`, `px` (plotly.express), and `go` are preloaded.
- When a dataset is loaded it is available as `df` -- use it, never reload it. \
When NO dataset is loaded, `df` is not defined; if the user wants a demo or \
example, create your own data (e.g. `pd.DataFrame` or `np.random`) in the code.
- For a CHART, assign a plotly figure to `fig` (or end with a bare figure \
expression). It goes to the Chart panel (B). Design it well: clear title, \
labeled axes, sensible colors/marks. Prefer `px`.
- For a COMPUTED TABLE, assign a DataFrame to `result` (or end with a bare \
DataFrame). It goes to the Data Preview panel (A).
- You may produce BOTH a `fig` and a `result` in one call; each lands in its \
own panel.
- Use `print(...)` for any text/numbers you want to reason over. The tool \
result's `note` is what you read (printed output, errors, and which panels were \
updated); the figure/table data are shown to the user, not returned to you.

2. WRITE AN ANALYSIS into the Summary panel (C) via `write_summary(markdown)`. \
Pass markdown (headings, bullets, a short narrative of your findings). Use this \
for insights/conclusions in prose -- not for charts or tables.

3. REARRANGE THE PANELS via `arrange_panels(mosaic)`. The mosaic is a small grid \
of the panel letters A (Data Preview) / B (Chart) / C (Summary); each letter's \
cells must form a rectangle. Examples: "AC\\nBB" (preview and summary on top, \
chart wide below -- the default), "B" (chart full-screen), "BB\\nAC" (chart wide \
on top, preview and summary below), "A\\nB\\nC" (stacked). Use ONLY A/B/C.

4. LOAD DATA / DRIVE THE APP. To load a dataset from a URL, call \
`load_dataset("<url>")` -- it fetches the data, fills the Data Preview and \
Summary panels, and makes `df` available so you can plot it with `run_analysis` \
IN THE SAME REPLY. (Prefer this over set_app_input for loading: set_app_input \
only fills the input box, so `df` would not be ready to plot until a later \
turn.) You CANNOT load a local file from chat; ask the user to use the Upload \
button on the left, then Run. You may also set other inputs with \
`set_app_input(name, value)` and re-run the deterministic app with `run_app()`.

Workflow: when a dataset is loaded, inspect its schema first; run focused code \
and give a short, concrete answer grounded in the results. Put charts, tables, \
and summaries into the panels; keep the chat reply itself concise. If code \
errors, read the traceback and fix it. Don't describe output you didn't actually \
create.

If no dataset is loaded and the user just wants to chat, help them get started: \
offer to load a URL for them (load_dataset), invite them to upload a CSV/Excel \
file, or render demo data on request."""


def _thread_id(config: RunnableConfig) -> str:
    return ((config or {}).get("configurable") or {}).get("thread_id") or "default"


@tool
def run_analysis(code: str, config: RunnableConfig) -> str:
    """Run Python (pandas/plotly) and render its output into the app's panels.

    Assign a plotly figure to `fig` and it is placed into the Chart panel (B);
    assign a DataFrame to `result` and it is placed into the Data Preview panel
    (A). You may produce both in one call. Use print() for text you want to
    reason over. Returns a short note (printed output, errors, and which panels
    were updated) -- never the figure/table data.
    """
    tid = _thread_id(config)
    df = current_dataframe(tid)
    # Run whether or not a dataset is loaded: with data, `df` is injected; with
    # none, the code runs on its own (e.g. synthetic/demo data the agent builds).
    inject = {"df": df} if df is not None else {}
    r = fast_dash.sandbox.run_code(code, inject=inject)

    note_parts = []
    if r.get("error"):
        note_parts.append("Error:\n" + r["error"])
    if r.get("stdout"):
        note_parts.append("Printed output:\n" + r["stdout"])

    # Route each produced artifact to its panel via a set_output frame. The frame
    # value stays a raw Python object (a plotly dict / a DataFrame); fast_dash
    # transforms it through the TARGET slot's component server-side and pushes it
    # per session. The model-facing note never carries the figure/table data.
    rendered = []
    if r.get("figure"):
        try:
            fig_dict = json.loads(r["figure"])  # plotly JSON string -> dict
            emit_frame({"type": "set_output", "slot": CHART_SLOT, "value": fig_dict})
            rendered.append("a chart in the Chart panel")
        except (json.JSONDecodeError, TypeError, ValueError):
            note_parts.append("Produced a figure but could not render it.")
    if r.get("table"):
        # Reconstruct a DataFrame; the Table slot's transform runs
        # DataFrame.to_dict("records") -> DataTable.data (verified empirically).
        table_df = pd.DataFrame(r["table"]["records"])
        emit_frame({"type": "set_output", "slot": PREVIEW_SLOT, "value": table_df})
        rendered.append("a table in the Data Preview panel")

    if rendered:
        if len(rendered) == 1:
            note_parts.append("Rendered %s." % rendered[0])
        else:
            note_parts.append("Rendered " + " and ".join(rendered) + ".")

    note = "\n\n".join(note_parts) or "Ran with no printed output."
    # Only the short note goes back to the model -- charts and tables are pushed
    # to the app's panels, never returned here.
    return json.dumps({"note": note})


@tool
def write_summary(markdown: str) -> str:
    """Write a summary/insights (markdown) into the app's Summary panel (C).

    Use for a written analysis in prose -- findings, conclusions, a short
    narrative -- not for charts or tables. Pass markdown (headings, bullets,
    text); it renders in the Summary panel, not inline in the chat.
    """
    emit_frame({"type": "set_output", "slot": SUMMARY_SLOT, "value": markdown})
    return "Wrote the summary into the Summary panel."


@tool
def arrange_panels(mosaic: str) -> str:
    """Rearrange the app's three output panels into a layout (mosaic).

    The mosaic is a small text grid: rows separated by newlines, each cell a
    panel letter -- A = Data Preview, B = Chart, C = Summary. Each letter's cells
    must form a filled rectangle, and letters must be among A/B/C (you can drop a
    panel by omitting its letter, but cannot invent new ones). Examples:
    "AC\\nBB" (preview and summary on top, chart wide below -- the default),
    "B" (chart full-screen), "BB\\nAC" (chart wide on top, preview and summary
    below), "A\\nB\\nC" (stacked). fast_dash validates the mosaic and refuses an
    invalid one with a note -- use only A/B/C.
    """
    emit_frame({"type": "set_layout", "mosaic": mosaic})
    return "Rearranged the panels to layout: %s" % mosaic


@tool
def set_app_input(name: str, value: str) -> str:
    """Set one of the app's inputs (e.g. dataset_url) so a Run picks it up.

    Emits a set_input frame the app applies to its live control. Use for
    `dataset_url`; the file upload input CANNOT be set from chat -- ask the user
    to use the Upload button for local files. Call run_app after to apply it.
    """
    emit_frame({"type": "set_input", "name": name, "value": value})
    return "Set input '%s'. Call run_app() to run the app on it." % name


@tool
def run_app() -> str:
    """Run the app on its current inputs and stream its outputs to the UI.

    Use after set_app_input to apply the change (e.g. load a dataset from a URL).
    """
    emit_frame({"type": "run_app"})
    return "Ran the app on the current inputs."


@tool
def load_dataset(url: str, config: RunnableConfig) -> str:
    """Load a dataset from a URL and show its preview + summary in the app.

    Fetches the CSV/Excel/JSON/Parquet at `url`, makes it available as `df`, and
    fills the Data Preview (A) and Summary (C) panels. Crucially, the data is
    ready to plot IN THE SAME REPLY: call `run_analysis` right after to chart it.
    Use this to load data from a link; for a local file, ask the user to use the
    Upload button. (This is the preferred way to load a URL -- it loads `df` for
    you, unlike set_app_input which only fills the input box.)
    """
    tid = _thread_id(config)
    try:
        df = data.get_dataframe(tid, None, url)   # loads + caches for this thread
    except Exception as exc:                       # noqa: BLE001 -- report, don't crash
        return "Could not load a dataset from that URL (%s)." % type(exc).__name__
    if df is None:
        return "Could not load a dataset from that URL -- check the link."
    # Caching the df above is the fix for load+plot-in-one-turn: run_analysis
    # (same thread) now sees `df` instead of finding an empty cache. Set the URL
    # input and Run the deterministic app to fill the preview/chart/summary
    # panels -- run_app drives all three atomically through the proven drive
    # reducer; the agent's run_analysis then overwrites the Chart panel.
    emit_frame({"type": "set_input", "name": URL_ID, "value": url})
    emit_frame({"type": "run_app"})
    return ("Loaded %d rows x %d columns and filled the panels. `df` is ready -- "
            "call run_analysis to chart it." % (df.shape[0], df.shape[1]))


def build_graph(model=None):
    if model is None:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=OPENROUTER_MODEL, base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY, temperature=0, streaming=True,
            default_headers={
                "HTTP-Referer": "https://github.com/dkedar7/datachat",
                "X-Title": "DataChat",
            },
        )
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(
        model,
        [run_analysis, write_summary, arrange_panels,
         load_dataset, set_app_input, run_app],
        prompt=SYSTEM_PROMPT)


def _make_stub_analyst():
    def analyst(query, ctx):
        yield ("This is the offline analyst. Set `OPENROUTER_API_KEY` to enable the "
               "AI data analyst that writes and runs analysis code.")

    return analyst


def make_analyst():
    if not has_llm():
        return _make_stub_analyst()

    # No custom extractors: charts / tables / summaries go to the app's panels
    # (set_output frames), not inline in the chat. The langstage default
    # extractors still apply for the agent's own typed events.
    bridge = build_chat_callback(build_graph())

    async def analyst(query, ctx):
        tid = getattr(ctx, "thread_id", None) or "default"
        inputs = getattr(ctx, "inputs", None) or {}
        df = data.get_dataframe(tid, inputs.get(UPLOAD_ID), inputs.get(URL_ID))
        # No gate on a loaded dataset: the user can chat immediately. When there's
        # no data, the agent can still drive the app (load a URL) or render demo
        # data; a loaded df is surfaced via its schema.
        if df is not None:
            prompt = f"[Dataset loaded: {schema_text(df)}]\n\n{query}"
        else:
            prompt = ("[No dataset is loaded yet -- `df` is not defined. You can "
                      "load one for the user with set_app_input('dataset_url', <url>) "
                      "then run_app(), run code to render example/synthetic data, or "
                      "invite them to upload a CSV/Excel file via the Upload "
                      "button.]\n\n" + query)
        try:
            async for frame in bridge(prompt, ctx):
                yield frame
        except Exception as exc:  # noqa: BLE001
            yield f"\n\nThe analyst hit an error: {type(exc).__name__}."

    return analyst
