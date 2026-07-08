"""The agent: run_analysis + write_summary + arrange_panels + drive tools emit
frames; plus the graph/make_analyst wiring.

The analyst has FULL control of the app's three output panels: charts render into
the Chart panel (B), computed tables into the Data Preview panel (A), and written
summaries into the Summary panel (C) -- each a ``set_output`` frame emitted
through ``fast_dash.agent_tools``. ``arrange_panels`` re-mosaics the panels (a
``set_layout`` frame), and the drive tools set inputs / run the app. Nothing
renders inline in the chat any more (no custom extractor).
"""

import json

import pandas as pd

from fast_dash.agent_tools import drain_frames, turn_buffer

from analyst import agent, data
from analyst.agent import (
    CHART_SLOT,
    PREVIEW_SLOT,
    SUMMARY_SLOT,
    arrange_panels,
    build_graph,
    make_analyst,
    load_dataset,
    run_analysis,
    run_app,
    set_app_input,
    write_summary,
)


# --- slot constants map to the output indices ----------------------------- #
def _build_app():
    from fast_dash import FastDash, Graph, Markdown, Table

    def explore(x: str = ""):
        return None, None, None

    def chatfn(query, ctx):
        yield "hi"

    return FastDash(
        callback_fn=explore, outputs=[Table, Graph, Markdown],
        output_labels=["Data preview", "Chart", "Summary"], mosaic="AC\nBB",
        title="T", chat=chatfn,
        chat_tools=("read_app", "set_input", "run_app", "set_output", "set_layout"))


def test_slot_constants_map_to_table_graph_markdown_indices():
    """Outputs are [Table, Graph, Markdown]; slot letters go by output order, so
    A=Table (0, Data preview), B=Graph (1, Chart), C=Markdown (2, Summary).
    Verified against a real app's resolver so the constants track fast_dash's slot
    mapping."""
    app = _build_app()
    assert (PREVIEW_SLOT, CHART_SLOT, SUMMARY_SLOT) == ("A", "B", "C")
    assert app.output_tags[0] is None                # a Table (no tag)
    assert app.output_tags[1] == "Graph"
    assert app.output_tags[2] == "Text"              # Markdown carries tag "Text"
    assert app._resolve_slot(PREVIEW_SLOT) == 0
    assert app._resolve_slot(CHART_SLOT) == 1
    assert app._resolve_slot(SUMMARY_SLOT) == 2


# --- value-shape sanity: pin the shapes to fast_dash's transforms --------- #
def test_set_output_value_shapes_match_fast_dash_transforms():
    """Build a real app and confirm the exact (prop, value) each slot's transform
    produces for the values the tools emit: a DataFrame -> Table.data (records),
    a markdown string -> Markdown.children, a plotly dict -> Graph.figure."""
    app = _build_app()

    # Table (A, index 0): a DataFrame -> to_dict("records") on prop "data".
    _id, prop, val = app._sidecar_set_output(0, pd.DataFrame({"g": ["a"], "v": [1]}))
    assert prop == "data"
    assert val == [{"g": "a", "v": 1}]

    # Graph (B, index 1): a plotly figure dict passes through on prop "figure".
    fig_dict = {"data": [{"type": "bar", "x": ["a"], "y": [1]}], "layout": {}}
    _id, prop, val = app._sidecar_set_output(1, fig_dict)
    assert prop == "figure"
    assert isinstance(val, dict) and "data" in val and "layout" in val

    # Markdown (C, index 2): a plain string passes through on prop "children".
    _id, prop, val = app._sidecar_set_output(2, "**hi**")
    assert prop == "children"
    assert val == "**hi**"


# --- run_analysis: a chart -> the Chart panel (B) ------------------------- #
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
    assert list(payload.keys()) == ["note"]         # only the note goes back
    assert "Chart panel" in payload["note"]
    assert "figure" not in payload                  # the figure is NOT returned


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


# --- run_analysis: a computed table -> the Data Preview panel (A) --------- #
def test_run_analysis_emits_a_table_into_the_preview_panel():
    data._CACHE["ra2"] = ("h", pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 2, 3]}))
    with turn_buffer():
        out = run_analysis.invoke(
            {"code": "print('rows', len(df)); result = df.groupby('g')['v'].sum().reset_index()"},
            {"configurable": {"thread_id": "ra2"}})
        frames = drain_frames()

    set_outputs = [f for f in frames if f.get("type") == "set_output"]
    assert len(set_outputs) == 1
    frame = set_outputs[0]
    assert frame["slot"] == PREVIEW_SLOT
    # The frame value is a DataFrame (fast_dash transforms it to records server-side).
    assert isinstance(frame["value"], pd.DataFrame)
    assert list(frame["value"].columns) == ["g", "v"]
    assert len(frame["value"]) == 2

    payload = json.loads(out)
    assert list(payload.keys()) == ["note"]
    assert "rows 3" in payload["note"]
    assert "Data Preview panel" in payload["note"]
    assert "table" not in payload                    # the table data is NOT returned


# --- run_analysis: BOTH a chart and a table -> two panels ----------------- #
def test_run_analysis_routes_both_chart_and_table_to_their_panels():
    data._CACHE["ra3"] = ("h", pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 2, 3]}))
    with turn_buffer():
        out = run_analysis.invoke(
            {"code": ("result = df.groupby('g')['v'].sum().reset_index()\n"
                      "fig = px.bar(result, x='g', y='v')")},
            {"configurable": {"thread_id": "ra3"}})
        frames = drain_frames()

    set_outputs = [f for f in frames if f.get("type") == "set_output"]
    slots = {f["slot"] for f in set_outputs}
    assert slots == {CHART_SLOT, PREVIEW_SLOT}       # one frame per panel
    by_slot = {f["slot"]: f["value"] for f in set_outputs}
    assert isinstance(by_slot[CHART_SLOT], dict)     # a plotly figure dict
    assert isinstance(by_slot[PREVIEW_SLOT], pd.DataFrame)

    note = json.loads(out)["note"]
    assert "Chart panel" in note and "Data Preview panel" in note


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
    (fix by generating data), not a canned refusal, and emits no output frame."""
    with turn_buffer():
        out = run_analysis.invoke({"code": "df.head()"},
                                  {"configurable": {"thread_id": "no-data-here-2"}})
        frames = drain_frames()
    payload = json.loads(out)
    assert "Error" in payload["note"] and "df" in payload["note"]
    assert not any(f.get("type") == "set_output" for f in frames)


# --- write_summary -> the Summary panel (C) ------------------------------- #
def test_write_summary_emits_a_set_output_markdown_frame():
    with turn_buffer():
        ack = write_summary.invoke({"markdown": "# Findings\n\n- Sales rose 12%."})
        frames = drain_frames()
    set_outputs = [f for f in frames if f.get("type") == "set_output"]
    assert len(set_outputs) == 1
    frame = set_outputs[0]
    assert frame["slot"] == SUMMARY_SLOT
    assert frame["value"] == "# Findings\n\n- Sales rose 12%."   # a markdown string
    assert "Summary panel" in ack


# --- arrange_panels -> a set_layout frame with any mosaic ----------------- #
def test_arrange_panels_emits_a_set_layout_frame_with_the_mosaic():
    with turn_buffer():
        ack = arrange_panels.invoke({"mosaic": "BB\nAC"})
        frames = drain_frames()
    assert frames == [{"type": "set_layout", "mosaic": "BB\nAC"}]
    assert "BB\nAC" in ack


def test_arrange_panels_passes_a_single_letter_mosaic_through():
    with turn_buffer():
        arrange_panels.invoke({"mosaic": "B"})
        frames = drain_frames()
    assert frames == [{"type": "set_layout", "mosaic": "B"}]


# --- drive tools: set_app_input / run_app (unchanged) --------------------- #
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


def test_load_dataset_makes_df_available_in_the_same_turn(monkeypatch):
    """The load+plot-in-one-turn fix: load_dataset caches the df immediately (so a
    same-turn run_analysis sees it) AND fills the preview/summary panels + the URL
    input -- without run_app, so no Run-reset can race the panels."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    monkeypatch.setattr("analyst.data.load_from_url", lambda url: df)
    tid = "loadds1"

    with turn_buffer():
        ack = load_dataset.invoke({"url": "http://example.com/data.csv"},
                                  {"configurable": {"thread_id": tid}})
        frames = drain_frames()

    # The df is now cached for this thread -> run_analysis (same tid) will see it.
    assert data.current_dataframe(tid) is not None
    assert "Loaded 3 rows" in ack

    # It filled the URL input + the preview (A) and summary (C) panels.
    assert {"type": "set_input", "name": "dataset_url",
            "value": "http://example.com/data.csv"} in frames
    set_outputs = {f["slot"]: f["value"] for f in frames if f.get("type") == "set_output"}
    assert PREVIEW_SLOT in set_outputs and SUMMARY_SLOT in set_outputs
    assert isinstance(set_outputs[PREVIEW_SLOT], pd.DataFrame)      # -> Table.data
    assert isinstance(set_outputs[SUMMARY_SLOT], str)              # -> Markdown
    assert not any(f.get("type") == "run_app" for f in frames)     # no run_app race

    # Same-turn plot: run_analysis now has `df` and charts into panel B.
    with turn_buffer():
        run_analysis.invoke({"code": "fig = px.scatter(df, x='a', y='b')"},
                            {"configurable": {"thread_id": tid}})
        chart_frames = drain_frames()
    assert any(f.get("type") == "set_output" and f.get("slot") == CHART_SLOT
               for f in chart_frames)


# --- AnalysisDisplay is gone; no custom extractor is wired ---------------- #
def test_analysis_display_extractor_is_removed():
    """Tables now go to the Data Preview panel, not inline -- the custom
    extractor and its class no longer exist."""
    assert not hasattr(agent, "AnalysisDisplay")


def test_make_analyst_wires_no_custom_extractors(monkeypatch):
    """build_chat_callback is called with only the graph -- no extractors kwarg --
    so the langstage default extractors apply and nothing renders inline."""
    seen = {}

    def _fake_bridge(graph, *args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs

        async def _bridge(prompt, ctx):
            if False:
                yield None

        return _bridge

    monkeypatch.setattr("analyst.agent.has_llm", lambda: True)
    monkeypatch.setattr("analyst.agent.build_graph", lambda: object())
    monkeypatch.setattr("analyst.agent.build_chat_callback", _fake_bridge)
    make_analyst()
    assert seen["args"] == ()                        # no positional extractors
    assert "extractors" not in seen["kwargs"]        # and no extractors kwarg


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
