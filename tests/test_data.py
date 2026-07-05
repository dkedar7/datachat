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
