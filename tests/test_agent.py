"""The agent: run_analysis tool, the AnalysisDisplay extractor, and wiring."""

import json

import pandas as pd

from analyst import agent, data
from analyst.agent import AnalysisDisplay, build_graph, make_analyst, run_analysis


# --- run_analysis tool ---------------------------------------------------- #
def test_run_analysis_payload_note_leads_with_a_figure():
    data._CACHE["ra1"] = ("h", pd.DataFrame({"g": ["a", "b"], "v": [1, 2]}))
    out = run_analysis.invoke({"code": "fig = px.bar(df, x='g', y='v')"},
                              {"configurable": {"thread_id": "ra1"}})
    payload = json.loads(out)
    assert list(payload.keys())[0] == "note"          # note leads
    assert isinstance(payload["figure"], str)         # plotly JSON string
    assert "table" not in payload


def test_run_analysis_payload_carries_a_table():
    data._CACHE["ra2"] = ("h", pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 2, 3]}))
    out = run_analysis.invoke(
        {"code": "print('rows', len(df)); result = df.groupby('g')['v'].sum().reset_index()"},
        {"configurable": {"thread_id": "ra2"}})
    payload = json.loads(out)
    assert list(payload.keys())[0] == "note"
    assert "rows 3" in payload["note"]
    assert payload["table"]["shape"][0] == 2


def test_run_analysis_reports_missing_dataset():
    out = run_analysis.invoke({"code": "df.head()"},
                              {"configurable": {"thread_id": "no-data-here"}})
    payload = json.loads(out)
    assert "No dataset" in payload["note"]


# --- AnalysisDisplay extractor -------------------------------------------- #
def test_extract_figure_envelope_carries_a_plotly_dict():
    fig_str = json.dumps({"data": [{"type": "bar"}], "layout": {}})
    env = AnalysisDisplay().extract(json.dumps({"note": "ok", "figure": fig_str}))
    assert env["display_type"] == "figure"
    assert isinstance(env["data"], dict) and "data" in env["data"]  # a plotly dict


def test_extract_table_envelope_carries_records():
    table = {"records": [{"g": "a", "v": 1}], "shape": [1, 2]}
    env = AnalysisDisplay().extract(json.dumps({"note": "ok", "table": table}))
    assert env["display_type"] == "table"
    assert env["data"] == [{"g": "a", "v": 1}]         # a list of records


def test_extract_returns_none_for_non_json_and_note_only():
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


def test_make_analyst_wrapper_streams_display_inline_for_a_seeded_df(monkeypatch):
    """The make_analyst wrapper: it loads the df from ctx.inputs, prefixes the
    schema, and streams the graph -- yielding a display_inline figure frame."""
    import asyncio

    from langchain_core.messages import AIMessage

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

    frames = asyncio.run(run())
    inline = [f for f in frames
              if isinstance(f, dict) and f.get("type") == "extraction"
              and f.get("extracted_type") == "display_inline"]
    assert inline, "expected an inline display frame"
    assert inline[0]["data"]["display_type"] == "figure"
    assert isinstance(inline[0]["data"]["data"], dict)      # a plotly figure dict
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
    # The turn streamed real content frames from the model (not a hardcoded nag),
    # and no run_analysis tool call was forced with no data present.
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
