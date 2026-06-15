"""
Streamlit UI for the Twitter/TikTok sentiment analyser.

Run with:
    streamlit run streamlit_app.py
"""

import logging
from enum import Enum
from typing import List, Tuple

import pandas as pd
import streamlit as st

from predictor import SentimentPredictor, SentimentResult

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SENTIMENT_COLOURS = {
    "Positive": "#4ade80",
    "Neutral":  "#94a3b8",
    "Negative": "#f87171",
}


class AnalysisMode(str, Enum):
    SINGLE   = "Quick Message Analysis"
    BATCH    = "Conversation Analysis"
    TWO_PARTY = "Two-Party Conversation Analysis"


# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_resource
def load_predictor() -> SentimentPredictor:
    """Load the sentiment model once and cache it across Streamlit reruns."""
    return SentimentPredictor()


# ── Helper functions ───────────────────────────────────────────────────────────

def decode_uploaded_lines(uploaded_file) -> List[str]:
    """Read and decode non-empty lines from a Streamlit UploadedFile."""
    raw_lines = uploaded_file.readlines()
    return [line.decode("utf-8").strip() for line in raw_lines if line.strip()]


def build_sentiment_dataframe(
    texts: List[str],
    results: List[SentimentResult],
) -> pd.DataFrame:
    """Combine texts and prediction results into a display DataFrame."""
    return pd.DataFrame({
        "Text":       texts,
        "Sentiment":  [label for label, _ in results],
        "Confidence": [round(score, 3) for _, score in results],
    })


def render_sentiment_chart(df: pd.DataFrame) -> None:
    """Render a bar chart of sentiment counts from a results DataFrame."""
    counts = df["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    st.bar_chart(counts, x="Sentiment", y="Count")


def render_party_analysis(
    party_label: str,
    texts: List[str],
    results: List[SentimentResult],
) -> pd.DataFrame:
    """
    Render sentiment metrics and chart for one conversation party.
    Returns the constructed DataFrame for use in the comparison section.
    """
    df = build_sentiment_dataframe(texts, results)
    st.markdown(f"### {party_label}")
    st.metric("Average Confidence", f"{df['Confidence'].mean():.2f}")
    render_sentiment_chart(df)
    st.dataframe(df, use_container_width=True)
    return df


# ── Page modes ─────────────────────────────────────────────────────────────────

def render_single_message_mode(predictor: SentimentPredictor) -> None:
    """Render the single-message analysis UI."""
    text_input = st.text_area("Enter your text:", height=150)

    if st.button("Analyse Sentiment") and text_input.strip():
        label, score = predictor.predict(text_input)
        colour = SENTIMENT_COLOURS.get(label, "#94a3b8")
        st.markdown(
            f"<div style='padding:1rem;border-radius:8px;background:{colour}22;"
            f"border-left:4px solid {colour}'>"
            f"<strong>Sentiment:</strong> {label} &nbsp;|&nbsp; "
            f"<strong>Confidence:</strong> {score:.2f}</div>",
            unsafe_allow_html=True,
        )


def render_conversation_mode(predictor: SentimentPredictor) -> None:
    """Render the single-party conversation analysis UI."""
    uploaded_file = st.file_uploader(
        "Upload a .txt file (one sentence per line)", type="txt"
    )
    if uploaded_file is None:
        return

    texts = decode_uploaded_lines(uploaded_file)
    if not texts:
        st.warning("The uploaded file appears to be empty.")
        return

    results = predictor.predict_batch(texts)
    df = build_sentiment_dataframe(texts, results)

    st.subheader("Conversation Sentiment Breakdown")
    st.metric("Average Confidence", f"{df['Confidence'].mean():.2f}")
    render_sentiment_chart(df)
    st.dataframe(df, use_container_width=True)


def render_two_party_mode(predictor: SentimentPredictor) -> None:
    """Render the two-party conversation comparison UI."""
    col1, col2 = st.columns(2)
    with col1:
        party1_file = st.file_uploader("Party 1 messages (.txt)", type="txt", key="p1")
    with col2:
        party2_file = st.file_uploader("Party 2 messages (.txt)", type="txt", key="p2")

    if not (party1_file and party2_file):
        return

    texts1 = decode_uploaded_lines(party1_file)
    texts2 = decode_uploaded_lines(party2_file)

    results1 = predictor.predict_batch(texts1)
    results2 = predictor.predict_batch(texts2)

    st.subheader("Party-wise Analysis")
    col1, col2 = st.columns(2)

    with col1:
        df1 = render_party_analysis("Party 1", texts1, results1)
    with col2:
        df2 = render_party_analysis("Party 2", texts2, results2)

    _render_comparison_table(df1, df2)


def _render_comparison_table(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Render an overall side-by-side sentiment comparison between two parties."""
    st.subheader("Overall Sentiment Comparison")

    def count_sentiment(df: pd.DataFrame, label: str) -> int:
        return int(df["Sentiment"].value_counts().get(label, 0))

    comparison_df = pd.DataFrame({
        "Party":            ["Party 1", "Party 2"],
        "Avg Confidence":   [round(df1["Confidence"].mean(), 2), round(df2["Confidence"].mean(), 2)],
        "Positive":         [count_sentiment(df1, "Positive"), count_sentiment(df2, "Positive")],
        "Neutral":          [count_sentiment(df1, "Neutral"),  count_sentiment(df2, "Neutral")],
        "Negative":         [count_sentiment(df1, "Negative"), count_sentiment(df2, "Negative")],
    })
    st.dataframe(comparison_df, use_container_width=True)


# ── App entry point ────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Sentiment Analyser", layout="centered")
    st.title("Do they like me? 💬")

    predictor = load_predictor()

    mode = st.selectbox(
        "Choose analysis mode",
        options=[m.value for m in AnalysisMode],
    )

    if mode == AnalysisMode.SINGLE.value:
        render_single_message_mode(predictor)
    elif mode == AnalysisMode.BATCH.value:
        render_conversation_mode(predictor)
    elif mode == AnalysisMode.TWO_PARTY.value:
        render_two_party_mode(predictor)


if __name__ == "__main__":
    main()
