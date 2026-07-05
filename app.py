"""
Emotion Analysis - Streamlit Application
A polished 7-class emotion analyzer with a card-based UI,
custom progress bars, donut chart, and prediction history.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Analysis",
    page_icon="🎭",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# Emotion configuration: emoji, color, and background
# ──────────────────────────────────────────────────────────────
EMOTION_CONFIG = {
    "joy":      {"emoji": "😊", "color": "#28a745", "bg": "#d4edda"},
    "love":     {"emoji": "🥰", "color": "#e84393", "bg": "#fce4ec"},
    "surprise": {"emoji": "😲", "color": "#8e44ad", "bg": "#f3e5f5"},
    "neutral":  {"emoji": "😐", "color": "#f0ad4e", "bg": "#fff8e1"},
    "sadness":  {"emoji": "😢", "color": "#3498db", "bg": "#e3f2fd"},
    "anger":    {"emoji": "😠", "color": "#e74c3c", "bg": "#fdecea"},
    "fear":     {"emoji": "😨", "color": "#5d6d7e", "bg": "#ecf0f1"},
}

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8eaf6 50%, #f3e5f5 100%);
    }

    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 28px 32px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.07);
        margin-bottom: 20px;
    }

    .page-header {
        text-align: center;
        padding: 10px 0 4px 0;
    }
    .page-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        color: #2c3e50;
        margin: 0;
    }
    .page-header h1 span { color: #5b6abf; }
    .page-header p {
        color: #7f8c8d;
        font-size: 1.05rem;
        margin-top: 2px;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 12px;
    }

    .result-header {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #5b6abf;
        border-bottom: 3px solid #5b6abf;
        display: inline-block;
        padding-bottom: 4px;
        margin-bottom: 18px;
    }

    .emotion-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3.2rem;
        margin-bottom: 10px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.10);
    }

    .emotion-label-text {
        font-size: 0.95rem;
        font-weight: 600;
        color: #555;
        margin-bottom: 2px;
    }
    .emotion-name {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .confidence-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 14px;
        border-radius: 20px;
        margin-bottom: 6px;
    }
    .confidence-score {
        font-size: 2rem;
        font-weight: 800;
        color: #2c3e50;
    }

    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        gap: 10px;
    }
    .prob-emoji {
        font-size: 1.4rem;
        width: 32px;
        text-align: center;
    }
    .prob-label {
        font-weight: 600;
        font-size: 0.88rem;
        width: 80px;
        color: #333;
    }
    .prob-bar-bg {
        flex: 1;
        background: #e9ecef;
        border-radius: 10px;
        height: 14px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease;
    }
    .prob-pct {
        font-weight: 700;
        font-size: 0.88rem;
        width: 60px;
        text-align: right;
        color: #333;
    }

    .history-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.88rem;
    }
    .history-table thead th {
        background: #f8f9fa;
        color: #555;
        font-weight: 600;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 2px solid #dee2e6;
    }
    .history-table tbody tr {
        transition: background 0.2s;
    }
    .history-table tbody tr:hover {
        background: #f5f6ff;
    }
    .history-table tbody td {
        padding: 10px 14px;
        border-bottom: 1px solid #f0f0f0;
        color: #333;
    }

    .sentiment-badge {
        display: inline-block;
        padding: 3px 14px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        color: #fff;
    }

    .app-footer {
        text-align: center;
        color: #999;
        font-size: 0.9rem;
        padding: 18px 0 8px 0;
    }
    .app-footer span { color: #e74c3c; }

    div.stButton > button {
        background: linear-gradient(135deg, #5b6abf, #3b4ea0) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 32px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        cursor: pointer !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
        box-shadow: 0 4px 14px rgba(91, 106, 191, 0.35) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(91, 106, 191, 0.45) !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 14px !important;
        transition: border-color 0.2s !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #5b6abf !important;
        box-shadow: 0 0 0 2px rgba(91, 106, 191, 0.18) !important;
    }
    div[data-testid="stTextArea"] label { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Load model and vectorizer (cached so they load only once)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    """Load the pretrained model and TF-IDF vectorizer from disk."""
    m = joblib.load("emotion_model.pkl")
    v = joblib.load("vectorizer.pkl")
    return m, v


model, vectorizer = load_resources()

# ──────────────────────────────────────────────────────────────
# Prediction helper
# ──────────────────────────────────────────────────────────────
def predict_emotion(text):
    """Return a list of (label, probability) tuples sorted highest first."""
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    paired = list(zip(model.classes_, probs))
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired

# ──────────────────────────────────────────────────────────────
# Session-state for prediction history
# ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <h1>🎭 <span>Emotion</span> Analysis</h1>
        <p>Enter any text / sentence and find out the emotion</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# INPUT CARD
# ──────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">✏️ Enter Your Text</div>',
    unsafe_allow_html=True,
)
user_input = st.text_area(
    "input",
    height=100,
    placeholder="Type or paste your sentence here...",
    label_visibility="collapsed",
)
btn_cols = st.columns([5, 1])
with btn_cols[1]:
    analyze = st.button("🚀 Analyze")
st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PREDICTION RESULT
# ──────────────────────────────────────────────────────────────
if analyze:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        results = predict_emotion(user_input)
        top_label, top_prob = results[0]
        cfg = EMOTION_CONFIG.get(
            top_label,
            {"emoji": "❓", "color": "#7f8c8d", "bg": "#ecf0f1"},
        )

        # Section header
        st.markdown(
            '<div style="text-align:center; margin-top:8px;">'
            '<span class="result-header">📊 Prediction Result</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        # Three-column layout
        col_left, col_mid, col_right = st.columns([1, 1.6, 1])

        # LEFT — main emotion display
        with col_left:
            st.markdown(
                f"""
                <div class="card" style="text-align:center; padding:24px 18px;">
                    <div class="emotion-label-text">Sentiment :</div>
                    <div class="emotion-circle"
                         style="background:{cfg['bg']}; margin:8px auto;">
                        {cfg['emoji']}
                    </div>
                    <div class="emotion-name"
                         style="color:{cfg['color']};">
                        {top_label.upper()}
                    </div>
                    <div class="confidence-badge"
                         style="background:{cfg['bg']}; color:{cfg['color']};">
                        Confidence Score
                    </div>
                    <div class="confidence-score">{top_prob:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # MIDDLE — probability bars
        with col_mid:
            bars_html = ""
            for label, prob in results:
                c = EMOTION_CONFIG.get(
                    label, {"emoji": "❓", "color": "#bdc3c7"}
                )
                width = max(prob * 100, 1.5)
                bars_html += f"""
                <div class="prob-row">
                    <div class="prob-emoji">{c['emoji']}</div>
                    <div class="prob-label">{label.title()}</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill"
                             style="width:{width}%; background:{c['color']};"></div>
                    </div>
                    <div class="prob-pct">{prob:.2%}</div>
                </div>
                """
            st.markdown(
                f"""
                <div class="card" style="padding:24px 24px;">
                    <div class="section-title" style="margin-bottom:16px;">
                        Emotion Probability
                    </div>
                    {bars_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # RIGHT — donut chart
        with col_right:
            chart_labels = [r[0].title() for r in results]
            chart_values = [r[1] for r in results]
            chart_colors = [
                EMOTION_CONFIG.get(r[0], {"color": "#bdc3c7"})["color"]
                for r in results
            ]

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=chart_labels,
                        values=chart_values,
                        hole=0.55,
                        marker=dict(colors=chart_colors),
                        textinfo="none",
                        hovertemplate="%{label}: %{percent}<extra></extra>",
                        sort=False,
                    )
                ]
            )
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    font=dict(size=11, family="Poppins"),
                ),
                margin=dict(l=0, r=100, t=10, b=10),
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )

        # Save to history
        st.session_state.history.insert(
            0,
            {
                "text": user_input,
                "emotion": top_label,
                "confidence": top_prob,
                "time": datetime.now().strftime("%I:%M %p"),
            },
        )

# ──────────────────────────────────────────────────────────────
# RECENT PREDICTIONS TABLE
# ──────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown(
        '<div style="text-align:left; margin-top:4px;">'
        '<span class="section-title">🕑 Recent Predictions</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    rows_html = ""
    for entry in st.session_state.history:
        ecfg = EMOTION_CONFIG.get(
            entry["emotion"], {"color": "#7f8c8d"}
        )
        rows_html += f"""
        <tr>
            <td>{entry['text']}</td>
            <td>
                <span class="sentiment-badge"
                      style="background:{ecfg['color']};">
                    {entry['emotion'].title()}
                </span>
            </td>
            <td style="font-weight:600;">{entry['confidence']:.2%}</td>
            <td>{entry['time']}</td>
        </tr>
        """

    st.markdown(
        f"""
        <div class="card" style="padding:20px 28px;">
            <table class="history-table">
                <thead>
                    <tr>
                        <th style="width:45%;">Text</th>
                        <th>Sentiment</th>
                        <th>Confidence</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-footer">
        Built with <span>❤️</span> using Streamlit &amp; Python
    </div>
    """,
    unsafe_allow_html=True,
)