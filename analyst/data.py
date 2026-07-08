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
import plotly.express as px
import plotly.graph_objects as go

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


# --- default overview chart ----------------------------------------------- #
_OVERVIEW_TITLE = "Overview (ask the analyst to plot anything)"


def empty_chart() -> go.Figure:
    """A placeholder figure shown in the Chart panel before any data is loaded.

    Invites the user to load a dataset and ask the analyst for a plot; the
    analyst renders real figures into this same slot via ``set_output``.
    """
    fig = go.Figure()
    fig.update_layout(
        title="No dataset yet -- load one, or ask the analyst to plot anything",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": ("Upload a CSV/Excel file or paste a link on the left, then ask "
                     "the Data Analyst<br>for any chart -- it renders right here."),
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
            "showarrow": False, "align": "center",
        }],
    )
    return fig


def overview_chart(df: pd.DataFrame) -> go.Figure:
    """A lightweight default chart summarizing ``df`` for the Chart panel.

    Picks the first usable column: a histogram of the first numeric column, else
    a bar of the first categorical column's top value counts. Guards empty /
    all-NaN / single-column frames, always returning a titled figure (never
    raising) so the app's Chart slot is never blank.
    """
    try:
        if df is None or df.shape[0] == 0 or df.shape[1] == 0:
            return _overview_message("The dataset is empty.")

        # Prefer the first numeric column that has at least one real value.
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce") if df[col].dtype == object \
                else df[col]
            if pd.api.types.is_numeric_dtype(s) and s.notna().any():
                fig = px.histogram(df, x=col, title=_OVERVIEW_TITLE)
                fig.update_layout(xaxis_title=str(col), yaxis_title="Count")
                return fig

        # No numeric column: bar the first column that has any non-null values.
        for col in df.columns:
            counts = df[col].dropna().astype(str).value_counts().head(20)
            if not counts.empty:
                fig = px.bar(x=counts.index, y=counts.values, title=_OVERVIEW_TITLE)
                fig.update_layout(xaxis_title=str(col), yaxis_title="Count")
                return fig

        return _overview_message("No plottable (non-empty) column found.")
    except Exception:  # noqa: BLE001 -- an overview must never break the app
        return _overview_message("Could not build an overview chart.")


def _overview_message(text: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=_OVERVIEW_TITLE,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": text + "<br>Ask the Data Analyst to plot anything.",
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
            "showarrow": False, "align": "center",
        }],
    )
    return fig
