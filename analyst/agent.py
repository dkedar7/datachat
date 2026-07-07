"""The Data Analyst -- a LangGraph agent that runs sandboxed code and plots.

Wired into Fast Dash as a ``chat=`` agent. It reads the uploaded/linked dataset
from ``ctx.inputs``, runs pandas/plotly code in fast_dash's built-in subprocess
sandbox via a ``run_analysis`` tool, and streams any produced figure/table as an
inline ``display_inline`` extraction so it renders inline in the chat. The
streaming + typed inline rendering are handled by fast_dash's langstage bridge.
Provider: OpenRouter.
"""

from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

import fast_dash.sandbox
from fast_dash.adapters.langstage import build_chat_callback

from . import data
from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, has_llm
from .data import current_dataframe, schema_text

UPLOAD_ID, URL_ID = "dataset_file", "dataset_url"

SYSTEM_PROMPT = """You are a data analyst. A pandas DataFrame is ALREADY loaded \
as `df` -- never load data yourself. Use the `run_analysis` tool to explore it \
and answer questions.

In the code you pass to run_analysis:
- The DataFrame is `df`; `pd`, `np`, `px` (plotly.express), and `go` are preloaded.
- To PLOT, build a plotly figure and assign it to `fig` (or end with a bare \
figure expression). Prefer `px` for charts. Always label axes/titles.
- To show a TABLE, assign a DataFrame to `result` (or end with a bare DataFrame).
- Use `print(...)` for any text/numbers you want to reason over.
- Figures and tables you produce render to the user automatically -- don't \
describe a chart you didn't actually create.
- The tool result's `note` field is what you read (printed output or the error \
trace); the figure/table are shown to the user, not returned to you as data.

Workflow: inspect the schema, run focused code, and give a short, concrete \
answer grounded in the results. If code errors, read the traceback and fix it. \
Keep replies concise."""


def _thread_id(config: RunnableConfig) -> str:
    return ((config or {}).get("configurable") or {}).get("thread_id") or "default"


@tool
def run_analysis(code: str, config: RunnableConfig) -> str:
    """Run Python (pandas/plotly) against the dataset `df` and return its output.

    Assign a chart to `fig` and/or a table to `result`; use print() for text.
    Any figure or table is shown to the user automatically.
    """
    tid = _thread_id(config)
    df = current_dataframe(tid)
    if df is None:
        return json.dumps({"note": "No dataset is loaded yet. Ask the user to "
                                   "upload a file or paste a link."})
    r = fast_dash.sandbox.run_code(code, inject={"df": df})

    note_parts = []
    if r.get("error"):
        note_parts.append("Error:\n" + r["error"])
    if r.get("stdout"):
        note_parts.append("Printed output:\n" + r["stdout"])
    note = "\n\n".join(note_parts) or "Ran with no printed output."

    payload = {"note": note}
    if r.get("figure"):
        payload["figure"] = r["figure"]        # plotly JSON string
    if r.get("table"):
        payload["table"] = r["table"]          # {"records": [...], "shape": [...]}
    return json.dumps(payload)


class AnalysisDisplay:
    """Turn a ``run_analysis`` result into an inline figure/table for the chat.

    A custom langstage extractor: fast_dash's chat renderer dispatches on
    ``extracted_type == "display_inline"`` and reads the envelope this returns
    (``display_type`` + ``data`` + ``title``). Figures carry a plotly figure
    *dict*; tables carry a list of records.
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
        figure = payload.get("figure")
        if figure:
            return {"display_type": "figure",
                    "data": json.loads(figure), "title": "Chart"}
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

    return create_react_agent(model, [run_analysis], prompt=SYSTEM_PROMPT)


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
        if df is None:
            yield ("Upload a CSV or Excel file (or paste a link to one) using the "
                   "panel on the left, and I'll analyze it for you.")
            return
        prompt = f"[Dataset: {schema_text(df)}]\n\n{query}"
        try:
            async for frame in bridge(prompt, ctx):
                yield frame
        except Exception as exc:  # noqa: BLE001
            yield f"\n\nThe analyst hit an error: {type(exc).__name__}."

    return analyst
