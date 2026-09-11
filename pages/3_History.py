"""
===================================================================================
STREAMLIT HISTORY PAGE — Depression Prediction for Cancer Patients
===================================================================================
This page displays session history with options to view past assessments,
export history data, and clear history.
"""

import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime

# Add parent directory to path to import from streamlit_app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app import (
    SEVERITY_LABELS, export_history_to_csv, init_session_state
)

# Severity index -> semantic tier, used to pick which .alert-card-<tier>
# class colors each history entry's summary.
SEVERITY_TIERS = {
    0: ("success", "🟢"),
    1: ("success", "🟢"),
    2: ("warning", "🟡"),
    3: ("elevated", "🟠"),
}


def severity_tier(severity_idx: int):
    return SEVERITY_TIERS.get(severity_idx, ("danger", "🔴"))


def kv_grid(pairs):
    """Render a compact label/value chip grid from a list of (label, value) tuples."""
    chips = "".join(
        f'<div class="kv-chip"><div class="kv-label">{label}</div>'
        f'<div class="kv-value">{value}</div></div>'
        for label, value in pairs
    )
    return f'<div class="kv-grid">{chips}</div>'


def history_page():
    st.set_page_config(
        page_title="History - Depression Screening",
        page_icon="📜",
        layout="wide"
    )

    # Same design system as the rest of the app, locked to a single light
    # theme regardless of the visitor's OS/browser preference or Streamlit's
    # own dark-mode toggle.
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    :root, html, html[data-theme="dark"], html[data-theme="light"] {
        color-scheme: light;
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --indigo: #4f46e5;
        --text: #172033;
        --muted: #64748b;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --surface-hover: #f1f5f9;
        --border: #e2e8f0;
        --input-bg: #ffffff;
        --radio-bg: #f8fafc;
        --sidebar-bg: #ffffff;
        --sidebar-text: #334155;
        --shadow: rgba(15, 23, 42, .06);

        /* Semantic alert tiers: bg / border / text */
        --success-bg: #ecfdf5;   --success-border: #22c55e;  --success-text: #166534;
        --info-bg:    #eff6ff;   --info-border:    #3b82f6;  --info-text:    #1e3a8a;
        --warning-bg: #fffbeb;   --warning-border: #f59e0b;  --warning-text: #92400e;
        --elevated-bg:#fff7ed;   --elevated-border:#f97316;  --elevated-text:#9a3412;
        --danger-bg:  #fef2f2;   --danger-border:  #ef4444;  --danger-text:  #991b1b;
        --stat-chip-bg: rgba(255,255,255,.6);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(37, 99, 235, 0.07), transparent 28%),
            radial-gradient(circle at 92% 10%, rgba(79, 70, 229, 0.06), transparent 25%),
            var(--surface-soft);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
    }

    [data-testid="stSidebar"] * {
        color: var(--sidebar-text);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        color: white;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 52%, #4f46e5 100%);
        box-shadow: 0 18px 45px rgba(37, 99, 235, .18);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -80px;
        top: -120px;
        border-radius: 50%;
        background: rgba(255,255,255,.10);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.25rem;
        line-height: 1.15;
        letter-spacing: -.03em;
    }

    .hero p {
        margin: .7rem 0 0;
        max-width: 680px;
        color: rgba(255,255,255,.84);
        font-size: 1rem;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        padding: .35rem .7rem;
        border-radius: 999px;
        background: rgba(255,255,255,.13);
        color: #dbeafe;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: .8rem;
    }

    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.35rem 1.5rem;
        margin: .85rem 0;
        box-shadow: 0 8px 25px var(--shadow);
    }

    .section-card .section-eyebrow {
        font-size: .78rem;
        color: var(--primary);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    .section-card h3 {
        margin: .3rem 0 .45rem;
        color: var(--text);
    }

    .section-card p {
        margin: 0;
        color: var(--muted);
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.3rem;
        box-shadow: 0 8px 25px var(--shadow);
    }

    .metric-label {
        color: var(--muted);
        font-size: .78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    .metric-value {
        color: var(--text);
        font-size: 2rem;
        font-weight: 700;
        margin-top: .25rem;
    }

    .metric-footnote {
        color: var(--muted);
        font-size: .85rem;
        margin-top: .3rem;
    }

    .alert-card {
        padding: 1.1rem 1.25rem;
        border-radius: 14px;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        color: var(--card-text);
        margin: 1rem 0;
    }

    .alert-card p, .alert-card h4 {
        color: var(--card-text);
        margin: 0;
    }

    .alert-card h4 {
        margin-bottom: .4rem;
        font-size: 1rem;
    }

    .alert-card--emphasis {
        border-width: 2px;
    }

    .alert-success { --card-bg: var(--success-bg); --card-border: var(--success-border); --card-text: var(--success-text); }
    .alert-info    { --card-bg: var(--info-bg);    --card-border: var(--info-border);    --card-text: var(--info-text); }
    .alert-warning { --card-bg: var(--warning-bg); --card-border: var(--warning-border); --card-text: var(--warning-text); }
    .alert-elevated{ --card-bg: var(--elevated-bg);--card-border: var(--elevated-border);--card-text: var(--elevated-text); }
    .alert-danger  { --card-bg: var(--danger-bg);  --card-border: var(--danger-border);  --card-text: var(--danger-text); }

    .result-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: .6rem;
    }

    .result-stat {
        padding: .8rem 1rem;
        border-radius: 12px;
        background: var(--stat-chip-bg);
    }

    .result-stat-label {
        font-size: .74rem;
        opacity: .8;
        margin-bottom: .15rem;
    }

    .result-stat-value {
        font-size: 1.05rem;
        font-weight: 700;
    }

    .kv-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: .6rem;
        margin-top: .5rem;
    }

    .kv-chip {
        background: var(--surface-hover);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: .55rem .75rem;
    }

    .kv-chip .kv-label {
        font-size: .7rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    .kv-chip .kv-value {
        font-size: .9rem;
        color: var(--text);
        font-weight: 600;
        margin-top: .15rem;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 11px !important;
        min-height: 2.8rem;
        font-weight: 600 !important;
        border: 1px solid var(--border) !important;
        transition: all .18s ease !important;
    }

    .stButton > button[kind="primary"], .stDownloadButton > button {
        border: none !important;
        background: linear-gradient(135deg, var(--primary), var(--indigo)) !important;
        color: white !important;
        box-shadow: 0 7px 18px rgba(37,99,235,.18);
    }

    .stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(37,99,235,.25);
    }

    .stButton > button[kind="secondary"] {
        background: var(--surface) !important;
        color: var(--text) !important;
    }

    .stButton > button:disabled {
        opacity: .5 !important;
        box-shadow: none !important;
        transform: none !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--surface);
        margin: .6rem 0;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    .stAlert {
        color: var(--text);
        border-radius: 12px;
    }

    /* Force every native Streamlit surface to light, even if the visitor's
       OS/browser prefers dark or they flip Streamlit's own theme toggle.
       Nothing else styles these app-chrome elements, so !important here
       can't clash with anything. */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stBottomBlockContainer"] {
        background-color: var(--surface-soft) !important;
        color: var(--text) !important;
    }

    /* Low-specificity base text color for plain markdown/captions/labels
       that aren't inside one of our styled cards — bare tag selectors so
       every card rule above still wins the cascade on its own color. */
    p, li, h2, h3, h4, label {
        color: var(--text);
    }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.6rem;
            border-radius: 18px;
        }

        .hero h1 {
            font-size: 1.75rem;
        }

        .result-grid, .kv-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Header
    st.markdown("""
    <div class="hero">
    <span class="eyebrow">Session history</span>
    <h1>📜 Assessment History</h1>
    <p>Review your past assessments and export session data.</p>
    </div>
    """, unsafe_allow_html=True)

    # Empty state
    if not st.session_state.history:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
        <h3>📭 No assessment history yet</h3>
        <p>Complete an assessment to see it show up here.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📋 Go to Assessment Page", type="primary", width='stretch'):
            st.switch_page("pages/1_Assessment.py")
        return

    # Summary metrics
    history = st.session_state.history
    latest = history[-1]
    latest_tier, latest_emoji = severity_tier(latest['prediction']['severity_idx'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-label">Total assessments</div>
        <div class="metric-value">{len(history)}</div>
        <div class="metric-footnote">Stored in this session</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-label">Latest result</div>
        <div class="metric-value" style="font-size:1.25rem;">{latest_emoji} {latest['prediction']['severity_label']}</div>
        <div class="metric-footnote">{latest['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-label">Latest confidence</div>
        <div class="metric-value">{latest['prediction']['confidence']:.1f}%</div>
        <div class="metric-footnote">{latest['mode'].capitalize()} mode</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Export history button
    csv_data = export_history_to_csv(history)
    if csv_data:
        st.download_button(
            label="📥 Export History to CSV",
            data=csv_data,
            file_name=f"assessment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width='stretch'
        )

    st.markdown("---")
    st.markdown("## 🗂️ All Assessments")

    # Display history as expandable, styled sections (most recent first)
    for i, assessment in enumerate(reversed(history), 1):
        pred = assessment['prediction']
        bdi_ref = assessment['bdi_reference']
        demographics = assessment['demographics']
        tier, emoji = severity_tier(pred['severity_idx'])

        with st.expander(f"{emoji} Assessment #{len(history) - i + 1} — {assessment['timestamp']} ({assessment['mode'].capitalize()} Mode)"):
            st.markdown(f"""
            <div class="alert-card alert-{tier}">
                <h4>{emoji} {pred['severity_label']}</h4>
                <div class="result-grid">
                    <div class="result-stat">
                        <div class="result-stat-label">Model confidence</div>
                        <div class="result-stat-value">{pred['confidence']:.1f}%</div>
                    </div>
                    <div class="result-stat">
                        <div class="result-stat-label">Depression probability</div>
                        <div class="result-stat-value">{pred['depression_prob']:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**BDI score:** {bdi_ref[0]}/{bdi_ref[1] * 3} (extrapolated: {bdi_ref[2]}/63, severity: {bdi_ref[3]})")

            st.markdown("##### 👤 Demographics")
            st.markdown(kv_grid([
                ("Gender", demographics.get('gender', '—')),
                ("Age group", demographics.get('age_group', '—')),
                ("Education", demographics.get('education_level', '—')),
                ("Cancer stage", demographics.get('cancer_stage', '—')),
                ("Cancer status", demographics.get('cancer_status', '—')),
            ]), unsafe_allow_html=True)

            # View detailed responses
            if st.button("🔍 View Detailed Responses", key=f"view_details_{i}", width='stretch'):
                resp_col1, resp_col2 = st.columns(2)
                with resp_col1:
                    st.markdown("**BDI Responses**")
                    bdi_df = pd.DataFrame(
                        [{"Item": k, "Score": v} for k, v in assessment['bdi_responses'].items()]
                    )
                    st.dataframe(bdi_df, width='stretch', hide_index=True)
                with resp_col2:
                    st.markdown("**FCRI Responses**")
                    fcri_df = pd.DataFrame(
                        [{"Item": k, "Score": v} for k, v in assessment['fcri_responses'].items()]
                    )
                    st.dataframe(fcri_df, width='stretch', hide_index=True)

    st.markdown("---")

    # Navigation buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 New Assessment", width='stretch'):
            st.session_state.assessment_step = 'mode_selection'
            st.session_state.demographics = {}
            st.session_state.bdi_responses = {}
            st.session_state.fcri_responses = {}
            st.session_state.current_assessment = None
            st.session_state.mode = None
            st.switch_page("pages/1_Assessment.py")

    with col2:
        if st.button("📊 View Latest Results", width='stretch'):
            if st.session_state.current_assessment:
                st.switch_page("pages/2_Results.py")
            else:
                st.warning("No current assessment available.")

    with col3:
        if st.button("🏠 Back to Home", width='stretch'):
            st.switch_page("streamlit_app.py")

if __name__ == '__main__':
    history_page()