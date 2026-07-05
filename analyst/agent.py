"""The Data Analyst — a LangGraph agent that runs sandboxed code and plots.

Mounted as a Fast Dash ``chat_agent=`` sidecar. It reads the uploaded/linked
dataset from ``ctx.inputs``, runs pandas/plotly code in the subprocess sandbox
via a ``run_python`` tool, and yields any produced figure/table as an
``artifact`` frame so it renders inline in the chat. Provider: OpenRouter.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, has_llm
from .data import current_dataframe, get_dataframe, schema_text
from .sandbox import run_code

UPLOAD_ID, URL_ID = "dataset_file", "dataset_url"

SYSTEM_PROMPT = """You are a data analyst. A pandas DataFrame is ALREADY loaded \
as `df` — never load data yourself. Use the `run_python` tool to explore it and \
answer questions.

In the code you pass to run_python:
- The DataFrame is `df`; `pd`, `np`, `px` (plotly.express), and `go` are imported.
- To PLOT, build a plotly figure and assign it to `fig` (or end with a bare \
figure expression). Prefer `px` for charts. Always label axes/titles.
- To show a TABLE, assign a DataFrame to `result` (or end with a bare DataFrame).
- Use `print(...)` for any text/numbers you want to reason over.
- Figures and tables you produce are shown to the user automatically — don't \
describe a chart you didn't actually create.

Workflow: inspect the schema, run focused code, and give a short, concrete \
answer grounded in the results. If code errors, read the traceback and fix it. \
Keep replies concise."""

# Figures/tables produced by run_python, per chat session, drained by the sidecar.
_ARTIFACTS: dict[str, list] = {}


def _thread_id(config: RunnableConfig) -> str:
    return ((config or {}).get("configurable") or {}).get("thread_id") or "default"


@tool
def run_python(code: str, config: RunnableConfig) -> str:
    """Run Python (pandas/plotly) against the dataset `df` and return its output.

    Assign a chart to `fig` and/or a table to `result`; use print() for text.
    Any figure or table is shown to the user automatically.
    """
    tid = _thread_id(config)
    df = current_dataframe(tid)
    if df is None:
        return "No dataset is loaded yet. Ask the user to upload a file or paste a link."
    r = run_code(code, df)
    if r.get("figure"):
        _ARTIFACTS.setdefault(tid, []).append(("figure", r["figure"]))
    if r.get("table"):
        _ARTIFACTS.setdefault(tid, []).append(("table", r["table"]))
    parts = []
    if r.get("error"):
        parts.append("Error:\n" + r["error"])
    if r.get("stdout"):
        parts.append("Printed output:\n" + r["stdout"])
    if r.get("figure"):
        parts.append("(A chart was produced and shown to the user.)")
    if r.get("table"):
        parts.append(f"(A table of shape {r['table']['shape']} was shown to the user.)")
    return "\n\n".join(parts) or "Ran successfully with no output."


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

    return create_react_agent(model, [run_python], prompt=SYSTEM_PROMPT)


# --- frame helpers -------------------------------------------------------- #
def _is_ai(message) -> bool:
    t = getattr(message, "type", "")
    return t == "ai" or "AIMessage" in message.__class__.__name__


def _text_of(message) -> str:
    c = getattr(message, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _rehydrate(kind: str, payload):
    if kind == "figure":
        import plotly.io as pio

        return pio.from_json(payload)
    if kind == "table":
        import pandas as pd

        return pd.DataFrame(payload["records"])
    return str(payload)


# ========================================================================== #
def _make_live_analyst():
    state = {"graph": None}

    async def analyst(query, ctx):
        tid = getattr(ctx, "thread_id", None) or "default"
        inputs = getattr(ctx, "inputs", None) or {}
        df = get_dataframe(tid, inputs.get(UPLOAD_ID), inputs.get(URL_ID))
        if df is None:
            yield ("Upload a CSV or Excel file (or paste a link to one) using the "
                   "panel on the left, and I'll analyze it for you.")
            return

        if state["graph"] is None:
            state["graph"] = build_graph()
        _ARTIFACTS[tid] = []
        prompt = f"[Dataset: {schema_text(df)}]\n\n{query}"
        config = {"configurable": {"thread_id": tid}, "recursion_limit": 24}
        try:
            async for mode, chunk in state["graph"].astream(
                {"messages": [("user", prompt)]}, config=config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    msg, _meta = chunk
                    if _is_ai(msg):
                        text = _text_of(msg)
                        if text:
                            yield {"type": "content", "content": text}
                elif mode == "updates":
                    while _ARTIFACTS.get(tid):
                        kind, payload = _ARTIFACTS[tid].pop(0)
                        yield {"type": "artifact", "content": _rehydrate(kind, payload)}
        except Exception as exc:  # noqa: BLE001
            yield f"\n\nThe analyst hit an error: {type(exc).__name__}."

    return analyst


def _make_stub_analyst():
    def analyst(query, ctx):
        yield ("This is the offline analyst. Set `OPENROUTER_API_KEY` to enable the "
               "AI data analyst that writes and runs analysis code.")

    return analyst


def make_analyst():
    return _make_live_analyst() if has_llm() else _make_stub_analyst()
