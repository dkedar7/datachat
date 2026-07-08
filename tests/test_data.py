"""Dataset loading, per-session cache, and summaries."""

import base64

import pandas as pd

from analyst import data


def _csv_url(df):
    return "data:text/csv;base64," + base64.b64encode(df.to_csv(index=False).encode()).decode()


def test_load_from_contents_csv():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    got = data.load_from_contents(_csv_url(df))
    assert list(got.columns) == ["a", "b"] and len(got) == 2


def test_get_dataframe_caches_and_reparses_on_change():
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"b": [2]})
    data.get_dataframe("s", contents=_csv_url(df1))
    assert list(data.current_dataframe("s").columns) == ["a"]
    data.get_dataframe("s", contents=_csv_url(df2))            # new source -> reparse
    assert list(data.current_dataframe("s").columns) == ["b"]


def test_no_source_returns_none():
    assert data.get_dataframe("s2") is None
    assert data.current_dataframe("never") is None


def test_schema_and_summary():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    assert "3 rows" in data.schema_text(df).replace(",", "")
    md = data.summary_markdown(df)
    assert "Loaded" in md and "| x |" in md


# --- default overview chart ----------------------------------------------- #
def _is_figure(fig):
    import plotly.graph_objects as go
    return isinstance(fig, go.Figure)


def test_empty_chart_is_a_figure_inviting_a_load():
    fig = data.empty_chart()
    assert _is_figure(fig)
    d = fig.to_plotly_json()                          # renderable as a plotly dict
    assert "data" in d and "layout" in d


def test_overview_chart_histograms_a_numeric_column():
    df = pd.DataFrame({"n": [1, 2, 2, 3, 3, 3], "g": ["a", "b", "a", "b", "a", "b"]})
    fig = data.overview_chart(df)
    assert _is_figure(fig)
    assert fig.data and fig.data[0].type == "histogram"


def test_overview_chart_bars_a_categorical_when_no_numeric():
    df = pd.DataFrame({"g": ["a", "b", "a", "c", "a"]})
    fig = data.overview_chart(df)
    assert _is_figure(fig)
    assert fig.data and fig.data[0].type == "bar"


def test_overview_chart_is_safe_on_all_nan_and_single_column():
    import numpy as np
    fig_nan = data.overview_chart(pd.DataFrame({"x": [np.nan, np.nan]}))
    assert _is_figure(fig_nan)                         # never raises, always a figure
    fig_one = data.overview_chart(pd.DataFrame({"only": [1, 2, 3]}))
    assert _is_figure(fig_one)


def test_overview_chart_is_safe_on_empty_frame():
    fig = data.overview_chart(pd.DataFrame())
    assert _is_figure(fig)
