"""The agent: run_python tool, rehydration, graph build, and a live plot."""

import base64
import os

import pandas as pd
import plotly.graph_objects as go
import pytest

from analyst import agent, data
from analyst.agent import _rehydrate, build_graph, run_python


def test_run_python_tool_stores_a_figure():
    data._CACHE["ta1"] = ("h", pd.DataFrame({"g": ["a", "b"], "v": [1, 2]}))
    agent._ARTIFACTS["ta1"] = []
    r = run_python.invoke({"code": "fig = px.bar(df, x='g', y='v')"},
                          {"configurable": {"thread_id": "ta1"}})
    assert "chart was produced" in r
    assert agent._ARTIFACTS["ta1"][0][0] == "figure"


def test_run_python_reports_missing_dataset():
    r = run_python.invoke({"code": "df.head()"},
                          {"configurable": {"thread_id": "no-data-here"}})
    assert "No dataset" in r


def test_rehydrate_figure_and_table():
    fig = go.Figure(go.Bar(x=[1], y=[2]))
    assert isinstance(_rehydrate("figure", fig.to_json()), go.Figure)
    tbl = _rehydrate("table", {"records": [{"a": 1}], "shape": [1, 1]})
    assert isinstance(tbl, pd.DataFrame) and tbl.iloc[0]["a"] == 1


def test_build_graph_is_a_compiled_graph():
    g = build_graph()
    assert hasattr(g, "get_graph") and hasattr(g, "astream")


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
def test_live_analyst_plots_from_upload():
    import asyncio

    df = pd.DataFrame({"city": ["NYC", "LA", "NYC", "SF"], "sales": [10, 20, 30, 40]})
    url = "data:text/csv;base64," + base64.b64encode(df.to_csv(index=False).encode()).decode()

    class Ctx:
        thread_id = "live-analyst"
        inputs = {"dataset_file": url, "dataset_url": ""}
        input_specs = []

    async def run():
        analyst = agent.make_analyst()
        arts = []
        async for f in analyst("Plot total sales by city.", Ctx()):
            if isinstance(f, dict) and f.get("type") == "artifact":
                arts.append(f["content"])
        return arts

    arts = asyncio.run(run())
    assert any(isinstance(a, go.Figure) for a in arts)   # it actually plotted
