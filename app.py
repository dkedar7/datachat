"""DataChat — upload or link any dataset, then chat with a data-analyst agent.

The app itself is a small upload/preview surface; the star is the **Data
Analyst** sidecar: a LangGraph + OpenRouter agent that runs sandboxed
pandas/plotly code and plots the answers inline in the chat.
"""

from __future__ import annotations

import pandas as pd

from fast_dash import FastDash, Markdown, Table, Upload

from analyst import data
from analyst.agent import make_analyst


def explore(dataset_file: Upload, dataset_url: str = ""):
    """Load a dataset and show a preview + summary.

    Upload a CSV/Excel file, or paste a link to one. Then open the Data Analyst
    (bottom-right) and ask for any chart or insight.

    Args:
        dataset_file: A CSV or Excel file to analyze.
        dataset_url: ...or a link to a CSV / Excel / JSON / Parquet file.
    """
    df = data.load(contents=dataset_file, url=dataset_url or None)
    if df is None:
        placeholder = pd.DataFrame({"Getting started": [
            "1. Upload a CSV/Excel file or paste a data link on the left.",
            "2. Open the Data Analyst (bottom-right) and ask a question.",
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
        chat_agent=make_analyst(),
        chat_agent_title="Data Analyst",
        chat_agent_drive=False,  # the analyst reads the data; it doesn't drive inputs
    )


app = build_app()
server = app.app.server  # WSGI entry point for gunicorn

if __name__ == "__main__":
    app.run()
