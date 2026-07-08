"""The agent: run_analysis + drive tools emit frames; the wiring and extractor.

Charts render into the app's Chart panel (a ``set_output`` frame emitted through
``fast_dash.agent_tools``), tables render inline in the chat (a ``display_inline``
extraction), and the drive tools set inputs / run the app.
"""

import json

import pandas as pd

from fast_dash.agent_tools import drain_frames, turn_buffer

from analyst import agent, data
from analyst.agent import (
    CHART_SLOT,
    AnalysisDisplay,
    build_graph,
    feature_chart,
    make_analyst,
    run_analysis,
    run_app,
    set_app_input,
)


# --- CHART_SLOT ----------------------------------------------------------- #
def test_chart_slot_is_the_graph_output_index_1():
    """Outputs are [Table, Graph, Markdown]; slot letters go by output order, so
    the Graph (chart) is the second output -> "B". Verified against a real app's
    resolver so the constant tracks fast_dash's slot mapping."""
    from fast_dash import FastDash, Graph, Markdown, Table

    def explore(x: str = ""):
        return None, None, None

    def chatfn(query, ctx):
        yield "hi"

    app = FastDash(
        callback_fn=explore, outputs=[Table, Graph, Markdown],
        output_labels=["Data preview", "Chart", "Summary"], mosaic="AC\nBB",
        title="T", chat=chatfn,
        chat_tools=("read_app", "set_input", "run_app", "set_output", "set_layout"))
    assert app.output_tags[1] == "Graph"            # the Graph is output index 1
    assert app._resolve_slot(CHART_SLOT) == 1       # ...which CHART_SLOT addresses
    assert CHART_SLOT == "B"


# --- run_analysis emits a chart into the app panel ------------------------ #
def test_run_analysis_emits_a_set_output_chart_frame():
    data._CACHE["ra1"] = ("h", pd.DataFrame({"g": ["a", "b"], "v": [1, 2]}))
    with turn_buffer():
        out = run_analysis.invoke({"code": "fig = px.bar(df, x='g', y='v')"},
                                  {"configurable": {"thread_id": "ra1"}})
        frames = drain_frames()

    set_outputs = [f for f in frames if f.get("type") == "set_output"]
    assert set_outputs, "expected a set_output frame carrying the chart"
    frame = set_outputs[0]
    assert frame["slot"] == CHART_SLOT
    value = frame["value"]
    assert isinstance(value, dict) and "data" in value and "layout" in value  # plotly dict

    payload = json.loads(out)
    assert list(payload.keys())[0] == "note"        # note leads
    assert "Chart panel" in payload["note"]
    assert "figure" not in payload                  # the figure is NOT returned
    assert "data" not in payload["note"] or '"data"' not in payload["note"]  # no fig JSON


def test_run_analysis_note_does_not_dump_the_figure_json():
    data._CACHE["ra1b"] = ("h", pd.DataFrame({"g": ["a", "b"], "v": [1, 2]}))
    with turn_buffer():
        out = run_analysis.invoke({"code": "fig = px.bar(df, x='g', y='v')"},
                                  {"configurable": {"thread_id": "ra1b"}})
        drain_frames()
    note = json.loads(out)["note"]
    # The model-facing note is a short ack, not a serialized plotly figure.
    assert "plotly" not in note.lower()
    assert '"marker"' not in note and '"xaxis"' not in note
    assert len(note) < 200


def test_run_analysis_table_stays_inline_no_chart_frame():
    data._CACHE["ra2"] = ("h", pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 2, 3]}))
    with turn_buffer():
        out = run_analysis.invoke(
            {"code": "print('rows', len(df)); result = df.groupby('g')['v'].sum().reset_index()"},
            {"configurable": {"thread_id": "ra2"}})
        frames = drain_frames()
    assert not any(f.get("type") == "set_output" for f in frames)  # tables aren't charts
    payload = json.loads(out)
    assert list(payload.keys())[0] == "note"
    assert "rows 3" in payload["note"]
    assert payload["table"]["shape"][0] == 2        # table passed through for the extractor


def test_run_analysis_emits_a_chart_without_a_dataset():
    """No dataset loaded: run_analysis still runs self-contained demo code (no hard
    refusal) and emits a set_output chart from it."""
    with turn_buffer():
        out = run_analysis.invoke(
            {"code": "fig = px.scatter(x=[1, 2, 3], y=[3, 1, 2], title='Demo')"},
            {"configurable": {"thread_id": "no-data-here"}})
        frames = drain_frames()
    payload = json.loads(out)
    assert "No dataset" not in payload["note"]       # no hard refusal
    set_outputs = [f for f in frames if f.get("type") == "set_output"]
    assert set_outputs and set_outputs[0]["slot"] == CHART_SLOT
    assert isinstance(set_outputs[0]["value"], dict)


def test_run_analysis_reports_a_missing_df_reference_as_an_error():
    """Referencing `df` with no dataset yields a traceback the agent can act on
    (fix by generating data), not a canned refusal, and emits no chart."""
    with turn_buffer():
        out = run_analysis.invoke({"code": "df.head()"},
                                  {"configurable": {"thread_id": "no-data-here-2"}})
        frames = drain_frames()
    payload = json.loads(out)
    assert "Error" in payload["note"] and "df" in payload["note"]
    assert not any(f.get("type") == "set_output" for f in frames)


# --- drive tools: set_app_input / run_app / feature_chart ----------------- #
def test_set_app_input_emits_a_set_input_frame():
    with turn_buffer():
        ack = set_app_input.invoke({"name": "dataset_url", "value": "http://x/y.csv"})
        frames = drain_frames()
    assert frames == [{"type": "set_input", "name": "dataset_url",
                       "value": "http://x/y.csv"}]
    assert "run_app" in ack


def test_run_app_emits_a_run_app_frame():
    with turn_buffer():
        ack = run_app.invoke({})
        frames = drain_frames()
    assert frames == [{"type": "run_app"}]
    assert "Ran the app" in ack


def test_feature_chart_emits_a_set_layout_frame_for_the_chart_slot():
    with turn_buffer():
        feature_chart.invoke({})
        frames = drain_frames()
    assert frames == [{"type": "set_layout", "mosaic": CHART_SLOT}]


# --- AnalysisDisplay extractor (table-only) ------------------------------- #
def test_extract_table_envelope_carries_records():
    table = {"records": [{"g": "a", "v": 1}], "shape": [1, 2]}
    env = AnalysisDisplay().extract(json.dumps({"note": "ok", "table": table}))
    assert env["display_type"] == "table"
    assert env["data"] == [{"g": "a", "v": 1}]       # a list of records


def test_extract_ignores_figures_and_note_only():
    # Figures go to the app panel, not inline -- the extractor never handles them.
    fig_str = json.dumps({"data": [{"type": "bar"}], "layout": {}})
    assert AnalysisDisplay().extract(json.dumps({"note": "ok", "figure": fig_str})) is None
    assert AnalysisDisplay().extract("not json at all") is None
    assert AnalysisDisplay().extract(json.dumps({"note": "just text"})) is None
    assert AnalysisDisplay().extract(json.dumps(["a", "list"])) is None


# --- graph + make_analyst wiring ------------------------------------------ #
def test_build_graph_is_a_compiled_graph():
    g = build_graph(model=_scripted_tool_model([]))
    assert hasattr(g, "get_graph") and hasattr(g, "astream")


def test_make_analyst_without_key_returns_the_stub(monkeypatch):
    monkeypatch.setattr("analyst.agent.has_llm", lambda: False)
    analyst = make_analyst()
    assert callable(analyst)
    out = list(analyst("hello", None))
    assert out and "offline analyst" in out[0]


def test_make_analyst_wrapper_streams_and_charts_into_the_app(monkeypatch):
    """The make_analyst wrapper loads the df from ctx.inputs, prefixes the schema,
    and streams the graph. A plotted figure is emitted as a set_output frame
    (drained by fast_dash), and the model still streams a text reply."""
    import asyncio

    from langchain_core.messages import AIMessage

    from fast_dash.agent_tools import drain_frames as _drain, turn_buffer as _buf

    model = _scripted_tool_model([
        AIMessage(content="", tool_calls=[{
            "name": "run_analysis",
            "args": {"code": "fig = px.bar(df, x='g', y='v')"}, "id": "c1"}]),
        AIMessage(content="Here is the chart."),
    ])
    monkeypatch.setattr("analyst.agent.has_llm", lambda: True)
    monkeypatch.setattr("analyst.agent.build_graph", lambda: build_graph(model=model))

    df = pd.DataFrame({"g": ["a", "b"], "v": [1, 2]})
    url = _csv_url(df)

    class Ctx:
        thread_id = "wrap1"
        resume = None
        inputs = {"dataset_file": url, "dataset_url": ""}
        input_specs = []

    async def run():
        analyst = make_analyst()
        frames = []
        async for f in analyst("Plot v by g.", Ctx()):
            frames.append(f)
        return frames

    # Open a turn buffer so the tool's emit_frame lands somewhere we can inspect
    # (in the app this is opened by fast_dash's turn runner).
    with _buf():
        frames = asyncio.run(run())
        emitted = _drain()

    set_outputs = [f for f in emitted if f.get("type") == "set_output"]
    assert set_outputs and set_outputs[0]["slot"] == CHART_SLOT
    assert isinstance(set_outputs[0]["value"], dict)         # a plotly figure dict
    assert any(isinstance(f, dict) and f.get("type") == "content" for f in frames)


def test_make_analyst_wrapper_chats_without_a_dataset(monkeypatch):
    """No gate on a loaded dataset: with no data the wrapper still streams the
    graph (the agent converses); it does not hard-return a nag message."""
    import asyncio

    from langchain_core.messages import AIMessage

    model = _scripted_tool_model([AIMessage(content="I can analyze any dataset.")])
    monkeypatch.setattr("analyst.agent.has_llm", lambda: True)
    monkeypatch.setattr("analyst.agent.build_graph", lambda: build_graph(model=model))

    class Ctx:
        thread_id = "empty1"
        resume = None
        inputs = {"dataset_file": None, "dataset_url": ""}
        input_specs = []

    async def run():
        analyst = make_analyst()
        return [f async for f in analyst("What can you do?", Ctx())]

    frames = asyncio.run(run())
    text = " ".join(
        f.get("content", "") for f in frames
        if isinstance(f, dict) and f.get("type") == "content")
    assert "I can analyze any dataset." in text
    assert not any(isinstance(f, dict) and f.get("type") == "extraction" for f in frames)


# --- test helpers --------------------------------------------------------- #
def _csv_url(df):
    import base64
    return "data:text/csv;base64," + base64.b64encode(
        df.to_csv(index=False).encode()).decode()


def _scripted_tool_model(responses):
    """A streaming, tool-binding fake chat model (mirrors fast_dash's test fake).

    GenericFakeChatModel cannot stream tool-call-only (empty-content) messages,
    which is exactly what create_react_agent + the langstage bridge need; this
    fake emits tool_call_chunks so a scripted tool call streams like a real model.
    """
    import json as _json

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    class _ScriptedToolFake(BaseChatModel):
        responses: list
        i: int = 0

        @property
        def _llm_type(self):
            return "scripted-tool-fake"

        def bind_tools(self, tools, **kwargs):
            return self

        def _next(self):
            msg = self.responses[min(self.i, len(self.responses) - 1)]
            self.i += 1
            return msg

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=self._next())])

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            msg = self._next()
            if msg.tool_calls:
                for idx, tc in enumerate(msg.tool_calls):
                    yield ChatGenerationChunk(message=AIMessageChunk(
                        content="", tool_call_chunks=[{
                            "name": tc["name"], "args": _json.dumps(tc["args"]),
                            "id": tc["id"], "index": idx}]))
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content=msg.content))

    return _ScriptedToolFake(responses=list(responses))
