"""The sandbox: captures figures/tables/output/errors, and stays locked down."""

import pandas as pd

from analyst.sandbox import run_code


def _df():
    return pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 2, 3]})


def test_produces_figure_from_fig_var():
    r = run_code("fig = px.bar(df, x='g', y='v')", _df())
    assert r["figure"] and r["error"] is None


def test_bare_expression_figure_is_captured():
    r = run_code("px.histogram(df, x='v')", _df())
    assert r["figure"]


def test_table_and_stdout():
    r = run_code("print('n', len(df)); result = df.groupby('g')['v'].sum().reset_index()", _df())
    assert r["table"]["shape"][0] == 2
    assert "n 3" in r["stdout"]


def test_error_is_captured_not_raised():
    r = run_code("df.this_does_not_exist()", _df())
    assert r["error"] and "this_does_not_exist" in r["error"]


def test_network_is_blocked():
    r = run_code("import urllib.request; urllib.request.urlopen('http://example.com')", _df())
    assert r["error"] and ("disabled" in r["error"].lower() or "OSError" in r["error"])


def test_secrets_are_scrubbed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-SECRET-value")
    r = run_code("import os; print(repr(os.environ.get('OPENROUTER_API_KEY')))", _df())
    assert "sk-SECRET" not in (r["stdout"] or "")


def test_timeout_is_enforced():
    r = run_code("while True:\n    pass", _df(), timeout=3)
    assert r["error"] and "longer than" in r["error"]
