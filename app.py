"""DataChat -- upload or link any dataset, then chat with a data-analyst agent.

The app itself is a small upload/preview surface; the star is the **Data
Analyst** in the left sidebar: a LangGraph + OpenRouter agent that runs
pandas/plotly code in fast_dash's built-in sandbox and plots the answers inline
in the chat via fast_dash's inline typed rendering.
"""

from __future__ import annotations

import pandas as pd

from fast_dash import FastDash, Markdown, Table, Upload

from analyst import data
from analyst.agent import make_analyst


def explore(dataset_file: Upload, dataset_url: str = ""):
    """Load a dataset and show a preview + summary.

    Upload a CSV/Excel file, or paste a link to one. Then ask the Data Analyst
    in the left sidebar for any chart or insight.

    Args:
        dataset_file: A CSV or Excel file to analyze.
        dataset_url: ...or a link to a CSV / Excel / JSON / Parquet file.
    """
    df = data.load(contents=dataset_file, url=dataset_url or None)
    if df is None:
        placeholder = pd.DataFrame({"Getting started": [
            "1. Upload a CSV/Excel file or paste a data link on the left.",
            "2. Ask the Data Analyst in the left sidebar a question.",
        ]})
        return placeholder, (
            "### No dataset loaded yet\n\nUpload a file or paste a link, then click "
            "**Run** to preview it — or just open the **Data Analyst** and ask."
        )
    return df.head(20), data.summary_markdown(df)


def build_app() -> FastDash:
    return FastDash(
        callback_fn=explore,
        outputs=[Table, Markdown],
        output_labels=["Data preview", "Summary"],
        title="DataChat",
        subheader="Upload or link any dataset, then chat to explore and plot it",
        accent="violet",
        github_url="https://github.com/dkedar7/datachat",
        chat=make_analyst(),
        chat_title="Data Analyst",
        chat_tools=(),  # the analyst reads the data; it operates nothing structural
        chat_placeholder="Ask me anything, or load a dataset and I'll analyze it.",
    )


app = build_app()
server = app.app.server  # WSGI entry point for gunicorn

if __name__ == "__main__":
    app.run()
