"""DataChat -- upload or link any dataset, then chat with a data-analyst agent.

The app itself is a small upload/preview surface; the star is the **Data
Analyst** in the left sidebar: a LangGraph + OpenRouter agent that is a full
complement (or replacement) for the input form. It can drive the form itself --
set ``dataset_url`` and run the app -- and it can plot *anything* into the app's
main-area **Chart** panel (not just inline in the chat), by running pandas/plotly
code in fast_dash's built-in sandbox and pushing the figure through fast_dash's
output pipeline.
"""

from __future__ import annotations

import pandas as pd

from fast_dash import FastDash, Graph, Markdown, Table, Upload

from analyst import data
from analyst.agent import make_analyst


def explore(dataset_file: Upload, dataset_url: str = ""):
    """Load a dataset and show a preview, a chart, and a summary.

    Upload a CSV/Excel file, or paste a link to one, then click **Run**. Or just
    open the **Data Analyst** in the left sidebar: it can set ``dataset_url`` and
    run the app for you, and plot any figure into the Chart panel below.

    Args:
        dataset_file: A CSV or Excel file to analyze (use the upload button).
        dataset_url: ...or a link to a CSV / Excel / JSON / Parquet file.
    """
    df = data.load(contents=dataset_file, url=dataset_url or None)
    if df is None:
        placeholder = pd.DataFrame({"Getting started": [
            "1. Upload a CSV/Excel file or paste a data link on the left.",
            "2. Ask the Data Analyst in the left sidebar a question.",
            "3. Or tell it to load a URL and plot anything -- it drives this app.",
        ]})
        summary = (
            "### No dataset loaded yet\n\nUpload a file or paste a link, then click "
            "**Run** to preview it -- or just open the **Data Analyst** and ask it to "
            "load data and plot anything."
        )
        return placeholder, data.empty_chart(), summary
    return df.head(20), data.overview_chart(df), data.summary_markdown(df)


def build_app() -> FastDash:
    return FastDash(
        callback_fn=explore,
        outputs=[Table, Graph, Markdown],
        output_labels=["Data preview", "Chart", "Summary"],
        # Preview (A) and summary (C) on top, the chart (B) full-width below, so
        # the analyst's figures land in a panel with real room. Slot letters are
        # assigned by output order: A=Table, B=Graph, C=Markdown.
        mosaic="AC\nBB",
        title="DataChat",
        subheader="Upload or link any dataset, then chat to explore and plot it",
        accent="violet",
        github_url="https://github.com/dkedar7/datachat",
        chat=make_analyst(),
        chat_title="Data Analyst",
        # The analyst is a full complement to the form: it can set inputs and run
        # the app (set_input/run_app), render a figure into the Chart panel
        # (set_output), and feature it full-screen on request (set_layout).
        chat_tools=("read_app", "set_input", "run_app", "set_output", "set_layout"),
        chat_placeholder="Ask me to load data, run the app, or plot anything.",
    )


app = build_app()
server = app.app.server  # WSGI entry point for gunicorn

if __name__ == "__main__":
    app.run()
