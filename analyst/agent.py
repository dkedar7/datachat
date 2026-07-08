"""The Data Analyst -- a LangGraph agent that drives the app and plots into it.

Wired into Fast Dash as a ``chat=`` agent, it is a full complement (or
replacement) for the input form. It can:

* **drive the form** -- ``set_app_input("dataset_url", <url>)`` then ``run_app()``
  set an input and re-run the deterministic app (files still use the Upload
  button; a file input can't be set from chat);
* **plot anything into the app** -- ``run_analysis`` runs pandas/plotly code in
  fast_dash's built-in subprocess sandbox and pushes any produced *figure* into
  the app's main-area **Chart** panel via a ``set_output`` frame (a *table* stays
  inline in the chat).

Tools emit frames through ``fast_dash.agent_tools.emit_frame``; fast_dash's turn
runner drains and dispatches them during the turn (the same mechanism fast_dash's
own toolkit uses). The streaming + typed inline rendering are handled by
fast_dash's langstage bridge. Provider: OpenRouter.
"""

from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

import fast_dash.sandbox
from fast_dash.adapters.langstage import build_chat_callback
from fast_dash.agent_tools import emit_frame

from . import data
from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, has_llm
from .data import current_dataframe, schema_text

UPLOAD_ID, URL_ID = "dataset_file", "dataset_url"

# The output slot the analyst renders figures into. Outputs are
# ``[Table, Graph, Markdown]`` and slot letters are assigned by output order, so
# the Graph (chart) is the second output -> "B". Verified against fast_dash's
# ``layout_object.output_slot_letters`` / ``_resolve_slot`` (see test_agent.py).
CHART_SLOT = "B"

SYSTEM_PROMPT = """You are a data analyst embedded in the DataChat app. You are a \
full complement to the app's input form: you can operate the app AND plot into it.

You have three ways to act:

1. DRIVE THE APP (its inputs + Run). To load a dataset from a URL, call \
`set_app_input("dataset_url", "<url>")` then `run_app()` -- this sets the app's \
input and re-runs the deterministic app, updating its preview/summary. You \
CANNOT set the file input from chat; if the user has a local file, tell them to \
use the Upload button on the left, then click Run (or ask you to run).

2. PLOT ANYTHING INTO THE APP'S CHART PANEL, via `run_analysis`. Build the best \
plotly figure you can and it renders in the app's main area (the Chart panel), \
NOT in the chat. In the code you pass to run_analysis:
- `pd`, `np`, `px` (plotly.express), and `go` are preloaded.
- When a dataset is loaded it is available as `df` -- use it, and never reload \
it. When NO dataset is loaded, `df` is not defined; if the user asks for a demo \
or example chart, create your own data (e.g. `pd.DataFrame` or `np.random`) right \
in the code.
- To PLOT, assign a plotly figure to `fig` (or end with a bare figure \
expression). Prefer `px`. Design it well: clear title, labeled axes, sensible \
colors/marks. The figure is placed into the app's Chart panel automatically.
- To show a TABLE, assign a DataFrame to `result` (or end with a bare \
DataFrame). Tables render inline in the chat.
- Use `print(...)` for any text/numbers you want to reason over. The tool \
result's `note` is what you read (printed output or the error trace); the \
figure/table are shown to the user, not returned to you as data.

3. FEATURE THE CHART. If the user asks for a big / hero / full-screen plot, call \
`feature_chart()` after plotting to expand the Chart panel to fill the app.

Workflow: when a dataset is loaded, inspect its schema first; run focused code \
and give a short, concrete answer grounded in the results. If code errors, read \
the traceback and fix it. Don't describe a chart you didn't actually create. \
Keep replies concise.

If no dataset is loaded and the user just wants to chat, help them get started: \
offer to load a URL for them (set_app_input + run_app), invite them to upload a \
CSV/Excel file, or plot demo data on request."""


def _thread_id(config: RunnableConfig) -> str:
    return ((config or {}).get("configurable") or {}).get("thread_id") or "default"


@tool
def run_analysis(code: str, config: RunnableConfig) -> str:
    """Run Python (pandas/plotly) and render its output into the app.

    Assign a chart to `fig` and it is placed into the app's Chart panel; assign a
    table to `result` and it shows inline in the chat; use print() for text.
    Returns a short note (printed output or error) for you to reason over -- not
    the figure data.
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

    # A produced FIGURE goes to the app's Chart panel (not inline in the chat):
    # emit a set_output frame carrying the plotly dict, which fast_dash renders
    # through the Chart slot's Graph transform and pushes per-session.
    if r.get("figure"):
        try:
            fig_dict = json.loads(r["figure"])  # plotly JSON string -> dict
            emit_frame({"type": "set_output", "slot": CHART_SLOT, "value": fig_dict})
            note_parts.append("Rendered a chart in the Chart panel.")
        except (json.JSONDecodeError, TypeError, ValueError):
            note_parts.append("Produced a figure but could not render it.")

    note = "\n\n".join(note_parts) or "Ran with no printed output."

    # The note never carries the figure JSON (charts are pushed to the panel, not
    # returned to the model). A table is passed through so the inline extractor
    # can render it in the chat.
    payload = {"note": note}
    if r.get("table"):
        payload["table"] = r["table"]  # {"records": [...], "shape": [...]}
    return json.dumps(payload)


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
def feature_chart() -> str:
    """Expand the Chart panel to fill the app (a big / hero / full-screen plot).

    Only use when the user explicitly wants a large or full-screen chart; emits a
    set_layout frame that gives the Chart slot the whole layout.
    """
    emit_frame({"type": "set_layout", "mosaic": CHART_SLOT})
    return "Featured the Chart panel full-screen."


class AnalysisDisplay:
    """Turn a ``run_analysis`` TABLE result into an inline table for the chat.

    A custom langstage extractor: fast_dash's chat renderer dispatches on
    ``extracted_type == "display_inline"`` and reads the envelope this returns
    (``display_type`` + ``data`` + ``title``). Figures are NOT handled here --
    they go to the app's Chart panel via a ``set_output`` frame; only tables
    render inline (as a list of records).
    """

    tool_name = "run_analysis"
    extracted_type = "display_inline"

    def extract(self, content):
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        table = payload.get("table")
        if table and table.get("records"):
            return {"display_type": "table",
                    "data": table["records"], "title": "Table"}
        return None


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
        model, [run_analysis, set_app_input, run_app, feature_chart],
        prompt=SYSTEM_PROMPT)


def _make_stub_analyst():
    def analyst(query, ctx):
        yield ("This is the offline analyst. Set `OPENROUTER_API_KEY` to enable the "
               "AI data analyst that writes and runs analysis code.")

    return analyst


def make_analyst():
    if not has_llm():
        return _make_stub_analyst()

    bridge = build_chat_callback(build_graph(), extractors=[AnalysisDisplay()])

    async def analyst(query, ctx):
        tid = getattr(ctx, "thread_id", None) or "default"
        inputs = getattr(ctx, "inputs", None) or {}
        df = data.get_dataframe(tid, inputs.get(UPLOAD_ID), inputs.get(URL_ID))
        # No gate on a loaded dataset: the user can chat immediately. When there's
        # no data, the agent can still drive the app (load a URL) or plot demo
        # data; a loaded df is surfaced via its schema.
        if df is not None:
            prompt = f"[Dataset loaded: {schema_text(df)}]\n\n{query}"
        else:
            prompt = ("[No dataset is loaded yet -- `df` is not defined. You can "
                      "load one for the user with set_app_input('dataset_url', <url>) "
                      "then run_app(), run code to plot example/synthetic data, or "
                      "invite them to upload a CSV/Excel file via the Upload "
                      "button.]\n\n" + query)
        try:
            async for frame in bridge(prompt, ctx):
                yield frame
        except Exception as exc:  # noqa: BLE001
            yield f"\n\nThe analyst hit an error: {type(exc).__name__}."

    return analyst
