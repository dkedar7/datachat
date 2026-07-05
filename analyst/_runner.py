"""Runs one snippet of analysis code in an isolated subprocess.

Invoked as ``python _runner.py <workdir>`` by ``sandbox.run_code``. Reads
``df.parquet`` + ``code.py`` from the workdir, executes the code against a
DataFrame ``df`` (with ``pd`` / ``np`` / ``px`` / ``go`` available), and writes
``result.json`` with any produced figure, table, printed output, or error.

Hardening (defense in depth — the parent also scrubs secrets from the env):
network is disabled, and CPU/address-space limits are set where supported.
"""

import ast
import io
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path


def _lock_down():
    # Disable network from within the sandbox (analysis runs on the provided df).
    # Block the connection functions + the connect() method rather than replacing
    # the socket class itself, which breaks stdlib imports (http.client etc.).
    import socket

    def _blocked(*a, **k):
        raise OSError("Network access is disabled in the analysis sandbox.")

    socket.getaddrinfo = _blocked          # no DNS -> no hostname connections
    socket.create_connection = _blocked
    try:
        socket.socket.connect = lambda self, *a, **k: _blocked()
        socket.socket.connect_ex = lambda self, *a, **k: _blocked()
    except (TypeError, AttributeError):
        pass
    # CPU + memory ceilings (POSIX only; a no-op on Windows dev).
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
        two_gb = 2 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (two_gb, two_gb))
    except Exception:
        pass


def _run(workdir: Path) -> dict:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    df = pd.read_parquet(workdir / "df.parquet")
    code = (workdir / "code.py").read_text(encoding="utf-8")
    ns = {"df": df, "pd": pd, "np": np, "px": px, "go": go, "__name__": "__sandbox__"}

    out = {"stdout": "", "figure": None, "table": None, "error": None}
    buf = io.StringIO()
    last_val = None
    try:
        tree = ast.parse(code)
        with redirect_stdout(buf):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                # Run everything but the trailing expression, then eval it so a
                # bare `px.bar(...)` last line is captured like a notebook cell.
                exec(compile(ast.Module(tree.body[:-1], []), "<code>", "exec"), ns)
                last_val = eval(
                    compile(ast.Expression(tree.body[-1].value), "<code>", "eval"), ns
                )
            else:
                exec(compile(tree, "<code>", "exec"), ns)
    except Exception:
        out["error"] = traceback.format_exc(limit=3)[-1500:]
    out["stdout"] = buf.getvalue()[-4000:]

    import plotly.graph_objects as go  # noqa: F811

    fig = ns.get("fig")
    if not isinstance(fig, go.Figure):
        fig = last_val if isinstance(last_val, go.Figure) else None
    if isinstance(fig, go.Figure):
        out["figure"] = fig.to_json()

    import pandas as pd  # noqa: F811

    tbl = ns.get("result")
    if not isinstance(tbl, (pd.DataFrame, pd.Series)):
        tbl = last_val if isinstance(last_val, (pd.DataFrame, pd.Series)) else None
    if isinstance(tbl, pd.Series):
        tbl = tbl.rename(tbl.name or "value").reset_index()
    if isinstance(tbl, pd.DataFrame):
        out["table"] = {
            "records": tbl.head(200).to_dict("records"),
            "shape": list(tbl.shape),
        }
    return out


def main():
    workdir = Path(sys.argv[1])
    _lock_down()
    try:
        result = _run(workdir)
    except Exception:
        result = {"stdout": "", "figure": None, "table": None,
                  "error": traceback.format_exc(limit=2)[-1000:]}
    (workdir / "result.json").write_text(
        json.dumps(result, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
