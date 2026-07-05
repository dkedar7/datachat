"""Dataset loading + a per-session cache the chat agent reads.

Turns an uploaded file (a ``dcc.Upload`` base64 data-URL) or a URL into a
DataFrame, and caches it per chat session so it's parsed once and reused across
turns (Fast Dash has no built-in per-session data store).
"""

from __future__ import annotations

import base64
import hashlib
import io

import pandas as pd

from .config import MAX_ROWS


# --- loading -------------------------------------------------------------- #
def load_from_contents(contents: str) -> pd.DataFrame:
    """Decode a ``dcc.Upload`` data-URL (``data:<mime>;base64,<...>``)."""
    header, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    bio = io.BytesIO(raw)
    is_excel = "excel" in header or "spreadsheet" in header or "officedocument" in header
    if is_excel:
        return pd.read_excel(bio)
    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0)
        return pd.read_csv(bio, sep=None, engine="python")  # sniff delimiter


def load_from_url(url: str) -> pd.DataFrame:
    u = url.strip()
    base = u.lower().split("?")[0]
    if base.endswith((".xlsx", ".xls")):
        return pd.read_excel(u)
    if base.endswith(".json"):
        return pd.read_json(u)
    if base.endswith((".parquet", ".pq")):
        return pd.read_parquet(u)
    return pd.read_csv(u)


def load(contents: str | None = None, url: str | None = None) -> pd.DataFrame | None:
    df = None
    if contents:
        df = load_from_contents(contents)
    elif url and url.strip():
        df = load_from_url(url)
    if df is not None and len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)
    return df


# --- per-session cache ---------------------------------------------------- #
_CACHE: dict[str, tuple[str, pd.DataFrame]] = {}


def _key(contents: str | None, url: str | None) -> str:
    src = f"{contents or ''}|{url or ''}"
    return hashlib.sha256(src.encode("utf-8", "ignore")).hexdigest()[:16]


def get_dataframe(thread_id: str, contents=None, url=None) -> pd.DataFrame | None:
    """Load (or return cached) DataFrame for a session, re-parsing on new input."""
    if not (contents or (url and url.strip())):
        return None
    h = _key(contents, url)
    hit = _CACHE.get(thread_id)
    if hit and hit[0] == h:
        return hit[1]
    df = load(contents=contents, url=url)
    if df is not None:
        _CACHE[thread_id] = (h, df)
    return df


def current_dataframe(thread_id: str) -> pd.DataFrame | None:
    hit = _CACHE.get(thread_id)
    return hit[1] if hit else None


# --- summaries ------------------------------------------------------------ #
def schema_text(df: pd.DataFrame) -> str:
    """A compact schema the agent sees so it can write correct code up front."""
    cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns[:60])
    extra = "" if df.shape[1] <= 60 else f" (+{df.shape[1] - 60} more)"
    return f"{df.shape[0]:,} rows x {df.shape[1]} columns. Columns: {cols}{extra}."


def summary_markdown(df: pd.DataFrame) -> str:
    lines = [
        f"### Loaded {df.shape[0]:,} rows x {df.shape[1]} columns",
        "",
        "| Column | Type | Non-null | Sample |",
        "| --- | --- | --- | --- |",
    ]
    for c in df.columns[:40]:
        sample = df[c].dropna()
        s = repr(sample.iloc[0]) if len(sample) else ""
        s = (s[:32] + "...") if len(s) > 35 else s
        lines.append(f"| {c} | {df[c].dtype} | {df[c].notna().sum():,} | {s} |")
    if df.shape[1] > 40:
        lines.append(f"| ... | | | *(+{df.shape[1] - 40} more columns)* |")
    lines.append("\n*Ask the **Data Analyst** in the left sidebar for any chart or insight.*")
    return "\n".join(lines)
