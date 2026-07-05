"""Execute LLM-written analysis code against a DataFrame, safely.

Each call runs in a fresh subprocess with a **scrubbed environment** (no API
keys or tokens), a wall-clock timeout, and — inside the child — a network block
and CPU/memory limits (see ``_runner.py``). The DataFrame is handed over as
parquet; the child returns any figure (as JSON), table, printed output, or error.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

_RUNNER = Path(__file__).with_name("_runner.py")
_SENSITIVE = ("KEY", "TOKEN", "SECRET", "PASSWORD", "OPENROUTER", "OPENAI")


def _scrubbed_env() -> dict:
    import os

    return {k: v for k, v in os.environ.items()
            if not any(s in k.upper() for s in _SENSITIVE)}


def run_code(code: str, df: pd.DataFrame, timeout: int = 25) -> dict:
    """Run ``code`` with ``df`` in scope. Returns a dict with keys:

    ``figure`` (plotly JSON str | None), ``table`` ({records, shape} | None),
    ``stdout`` (str), ``error`` (str | None).
    """
    with tempfile.TemporaryDirectory(prefix="datachat_") as tmp:
        work = Path(tmp)
        df.to_parquet(work / "df.parquet")
        (work / "code.py").write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(_RUNNER), str(work)],
                capture_output=True, text=True, timeout=timeout,
                env=_scrubbed_env(), cwd=str(work),
            )
        except subprocess.TimeoutExpired:
            return {"figure": None, "table": None, "stdout": "",
                    "error": f"The code took longer than {timeout}s and was stopped."}
        out_file = work / "result.json"
        if out_file.exists():
            return json.loads(out_file.read_text(encoding="utf-8"))
        return {"figure": None, "table": None, "stdout": (proc.stdout or "")[-2000:],
                "error": (proc.stderr or "").strip()[-1200:] or "No result was produced."}
