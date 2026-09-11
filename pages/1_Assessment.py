"""
===================================================================================
STREAMLIT ASSESSMENT PAGE — Depression Prediction for Cancer Patients
===================================================================================
This page handles the interactive questionnaire interface for collecting
demographics, BDI responses, and FCRI responses.
"""

import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime

# Add parent directory to path to import from streamlit_app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app import (
    BDI_QUESTIONS, FCRI_KEY_ITEMS, FCRI_ALL_ITEMS,
    load_top_items, build_feature_vector, predict, bdi_reference,
    TOP_N_BDI, TOP_N_FCRI, CLINICAL_FOLLOW_UP_THRESHOLD,
    CRISIS_RESOURCES, init_session_state,
    SEVERITY_LABELS, SEVERITY_DESCRIPTIONS, export_results_to_csv, export_results_to_pdf
)

# Severity index -> semantic tier used for theme-aware styling.
# Each tier maps to an .alert-card.alert-<tier> class defined in the CSS below,
# so colors are resolved by the browser per active theme instead of being
# baked in as fixed hex values from Python.
SEVERITY_TIERS = {
    0: ("success", "🟢"),
    1: ("success", "🟢"),
    2: ("warning", "🟡"),
    3: ("elevated", "🟠"),
}


def severity_tier(severity_idx: int):
    return SEVERITY_TIERS.get(severity_idx, ("danger", "🔴"))


def display_assessment_results(assessment):
    """Display assessment results directly in the current page."""
    pred = assessment['prediction']
    bdi_ref = assessment['bdi_reference']
    demographics = assessment['demographics']
    mode = assessment['mode']

    # Success banner
    st.markdown("""
    <div class="banner-hero">
    <h2>✅ Assessment Completed Successfully</h2>
    <p>{timestamp} • {mode} Assessment</p>
    </div>
    """.format(timestamp=assessment['timestamp'], mode=mode.capitalize()),
    unsafe_allow_html=True)

    # Display demographics summary
    with st.expander("👤 View Demographics Information"):
        st.json(demographics)

    st.markdown("---")

    # Primary Results Card
    severity_idx = pred['severity_idx']
    severity_label = pred['severity_label']
    tier, emoji = severity_tier(severity_idx)

    st.markdown(f"""
    <div class="result-hero alert-card alert-{tier}">
        <div class="result-hero-eyebrow">ML screening result</div>
        <h2>{emoji} {severity_label}</h2>
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

    # Progress bar for confidence
    st.progress(pred['confidence'] / 100, text=f"Model Confidence: {pred['confidence']:.1f}%")

    st.markdown("---")

    # Severity probability breakdown
    st.markdown("## 📊 Severity Probability Breakdown")

    prob_data = []
    for i, sev in enumerate(SEVERITY_LABELS):
        prob = float(pred['multi_proba'][i]) * 100 if i < len(pred['multi_proba']) else 0.0
        prob_data.append({
            'Severity': sev,
            'Probability': prob
        })

    # Display as table with styling
    prob_df = pd.DataFrame(prob_data)
    prob_df = prob_df.sort_values('Probability', ascending=False)

    st.dataframe(
        prob_df.style.format({'Probability': '{:.1f}%'})
        .background_gradient(cmap='RdYlGn_r', subset=['Probability'])
        .set_properties(**{'text-align': 'center'}),
        width='stretch',
        hide_index=True
    )

    st.markdown("---")

    # BDI Reference Score
    st.markdown("## 📈 BDI Reference Score")

    bdi_raw, bdi_asked, bdi_extrap, bdi_rule_label = bdi_ref

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Raw BDI score</div>
            <div class="metric-value">{bdi_raw}/{bdi_asked * 3}</div>
            <div class="metric-footnote">Based on {bdi_asked} items answered</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Extrapolated score</div>
            <div class="metric-value">{bdi_extrap}/63</div>
            <div class="metric-footnote">Severity: {bdi_rule_label}</div>
        </div>
        """, unsafe_allow_html=True)

    if bdi_asked != 21:
        st.info("💡 **Note**: Quick-mode BDI score is extrapolated from fewer items. Run a Full Assessment for a validated 21-item BDI score.")

    st.caption("Rule-based score is informational; ML model output drives the assessment.")

    st.markdown("---")

    # Clinical description
    st.markdown("## 🏥 Clinical Assessment")
    desc = SEVERITY_DESCRIPTIONS.get(severity_label, '')
    if desc:
        st.markdown(f"""
        <div class="alert-card alert-info">
        <p><strong>Clinical Assessment:</strong> {desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Clinical follow-up flag
    if bdi_extrap >= CLINICAL_FOLLOW_UP_THRESHOLD or severity_idx >= 2:
        st.markdown("""
        <div class="alert-card alert-warning alert-card--emphasis">
        <h4>⚠️ Clinical follow-up recommended</h4>
        <p>Please discuss these results with your oncologist or a mental health professional at the earliest opportunity.</p>
        </div>
        """, unsafe_allow_html=True)

    # Crisis resources for severe cases
    if severity_idx >= 3:  # moderate or above
        st.markdown("---")
        st.markdown("## 🚨 Crisis & Support Resources (Pakistan)")
        st.error("If you are in immediate distress, please contact these resources:")

        crisis_cols = st.columns(2)
        for i, (name, number) in enumerate(CRISIS_RESOURCES.items()):
            with crisis_cols[i % 2]:
                st.markdown(f"""
                <div class="alert-card alert-danger crisis-card">
                <p class="crisis-name">{name}</p>
                <p class="crisis-number">📞 {number}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Export functionality
    st.markdown("## 📥 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = export_results_to_csv(
            demographics,
            assessment['bdi_responses'],
            assessment['fcri_responses'],
            pred,
            bdi_ref,
            assessment['timestamp']
        )
        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv_data,
            file_name=f"depression_assessment_{assessment['timestamp'].replace(' ', '_').replace(':', '-')}.csv",
            mime="text/csv",
            width='stretch'
        )

    with col2:
        pdf_data = export_results_to_pdf(
            demographics,
            assessment['bdi_responses'],
            assessment['fcri_responses'],
            pred,
            bdi_ref,
            assessment['timestamp']
        )
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_data,
            file_name=f"depression_assessment_{assessment['timestamp'].replace(' ', '_').replace(':', '-')}.pdf",
            mime="application/pdf",
            width='stretch'
        )

    st.markdown("---")

    # Disclaimer
    st.markdown("""
    <div class="alert-card alert-warning alert-card--emphasis">
    <h4>⚠️ Important disclaimer</h4>
    <p>
    This tool is for <strong>screening purposes only</strong> and is <strong>NOT a clinical diagnosis</strong>.
    Please consult a qualified healthcare professional for proper evaluation and treatment recommendations.
    </p>
    </div>
    """, unsafe_allow_html=True)

def assessment_page():
    st.set_page_config(
        page_title="Assessment - Depression Screening",
        page_icon="📋",
        layout="wide"
    )

    # Modern clinical dashboard styling — theme tokens are defined once and
    # every colored surface (alerts, results hero, crisis cards, metrics)
    # reads from these variables. The app is locked to a single light
    # theme regardless of the visitor's OS/browser preference or Streamlit's
    # own dark-mode toggle, so the design is never seen half-dark/half-light.
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

    .section-card, .question-card {
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

    .mode-card {
        min-height: 245px;
        padding: 1.6rem;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: var(--surface);
        box-shadow: 0 10px 30px var(--shadow);
        transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
        margin-bottom: 20px;
    }

    .mode-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 36px var(--shadow);
        border-color: var(--primary);
    }

    .mode-icon {
        width: 46px;
        height: 46px;
        display: grid;
        place-items: center;
        border-radius: 14px;
        background: color-mix(in srgb, var(--primary) 12%, var(--surface));
        font-size: 1.35rem;
        margin-bottom: 1rem;
    }

    .mode-card h3 {
        margin: 0 0 .55rem;
        color: var(--text);
        font-size: 1.25rem;
    }

    .mode-card p {
        color: var(--muted);
        margin: 0 0 1rem;
        font-size: .92rem;
        line-height: 1.6;
    }

    .mode-card ul {
        color: var(--text);
        margin: 0;
        padding-left: 1.2rem;
        line-height: 1.8;
        font-size: .9rem;
    }

    .stepper {
        display: flex;
        align-items: center;
        gap: .55rem;
        margin: 0 0 1.35rem;
        padding: .75rem 1rem;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: color-mix(in srgb, var(--surface) 88%, transparent);
        backdrop-filter: blur(8px);
    }

    .step {
        display: flex;
        align-items: center;
        gap: .45rem;
        color: var(--muted);
        font-size: .82rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .step.active {
        color: var(--primary);
    }

    .step.done {
        color: var(--success-border);
    }

    .step-dot {
        width: 25px;
        height: 25px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background: var(--border);
        color: var(--text);
        font-size: .72rem;
        font-weight: 700;
    }

    .step.active .step-dot {
        color: white;
        background: var(--primary);
        box-shadow: 0 0 0 4px color-mix(in srgb, var(--primary) 25%, transparent);
    }

    .step.done .step-dot {
        color: white;
        background: var(--success-border);
    }

    .step-line {
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    /* ---- Unified alert-card system used across notices, results hero
       and crisis cards. Each tier only sets the three color variables;
       structural styling stays shared so every notice looks consistent. */
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

    .instruction {
        padding: 1rem 1.15rem;
        border: 1px solid var(--info-border);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        background: var(--info-bg);
        color: var(--text);
        margin: 1rem 0 1.35rem;
    }

    .instruction.warning {
        border-color: var(--warning-border);
        border-left-color: var(--warning-border);
        background: var(--warning-bg);
        color: var(--text);
    }

    .question-heading {
        margin: 0 0 .9rem;
        color: var(--text);
        font-size: 1.02rem;
        font-weight: 700;
        line-height: 1.45;
    }

    .question-meta {
        color: var(--muted);
        font-size: .78rem;
        margin-top: -.5rem;
        margin-bottom: 1rem;
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

    .banner-hero {
        background: linear-gradient(135deg, var(--success-border) 0%, #20c997 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .banner-hero h2 { margin: 0; font-size: 1.8rem; }
    .banner-hero p { margin: .5rem 0 0 0; opacity: .92; }

    .result-hero {
        border-radius: 22px;
        padding: 1.6rem;
    }

    .result-hero h2 {
        margin: 0 0 .8rem;
        font-size: 1.65rem;
    }

    .result-hero-eyebrow {
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        opacity: .75;
    }

    .result-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
    }

    .result-stat {
        padding: 1rem;
        border-radius: 14px;
        background: var(--stat-chip-bg);
    }

    .result-stat-label {
        font-size: .78rem;
        opacity: .8;
        margin-bottom: .2rem;
    }

    .result-stat-value {
        font-size: 1.15rem;
        font-weight: 700;
    }

    .crisis-card {
        margin: .5rem 0;
    }

    .crisis-name {
        font-weight: 700;
    }

    .crisis-number {
        margin-top: .25rem !important;
        font-size: 1.2rem;
        font-weight: 700;
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

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--indigo));
    }

    .stProgress {
        margin: .7rem 0 1.2rem;
    }

    div[data-testid="stRadio"] > div {
        gap: .5rem;
    }

    div[data-testid="stRadio"] label {
        padding: .75rem .9rem !important;
        border: 1px solid var(--border);
        border-radius: 11px;
        background: var(--radio-bg);
        transition: all .15s ease;
    }

    div[data-testid="stRadio"] label:hover {
        border-color: var(--primary);
        background: color-mix(in srgb, var(--primary) 10%, var(--surface));
    }

    /* Theme-safe Streamlit form controls */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background: var(--input-bg) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    [role="option"] {
        color: var(--text) !important;
        background: var(--surface) !important;
    }

    [role="option"]:hover {
        background: var(--surface-hover) !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--surface);
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
       that aren't inside one of our styled cards. Deliberately bare tag
       selectors (no class/important) so every card rule above already
       wins the cascade and keeps its own color (e.g. white hero text,
       tinted alert-card text) without a specificity fight. */
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

        .stepper {
            overflow-x: auto;
        }

        .step-line {
            min-width: 22px;
        }

        .result-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Modern page header
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Clinical screening tool</div>
        <h1>Depression Assessment</h1>
        <p>Answer a short set of questions to generate an ML-assisted depression screening result for cancer patients.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for assessment (if not already done in init_session_state)
    if 'assessment_step' not in st.session_state:
        st.session_state.assessment_step = 'mode_selection'
    if 'demographics' not in st.session_state:
        st.session_state.demographics = {}
    if 'bdi_responses' not in st.session_state:
        st.session_state.bdi_responses = {}
    if 'fcri_responses' not in st.session_state:
        st.session_state.fcri_responses = {}

    # Assessment progress
    step_names = {
        'mode_selection': ('1', 'Mode'),
        'demographics': ('2', 'Profile'),
        'bdi_assessment': ('3', 'BDI'),
        'fcri_assessment': ('4', 'FCRI'),
        'results': ('5', 'Results')
    }
    current_step = st.session_state.get('assessment_step', 'mode_selection')
    current_num = int(step_names.get(current_step, ('1', 'Mode'))[0])

    step_html = '<div class="stepper">'
    labels = [('1', 'Mode'), ('2', 'Profile'), ('3', 'BDI'), ('4', 'FCRI'), ('5', 'Results')]
    for num, label in labels:
        n = int(num)
        cls = 'done' if n < current_num else ('active' if n == current_num else '')
        icon = '✓' if n < current_num else num
        step_html += f'<div class="step {cls}"><span class="step-dot">{icon}</span><span>{label}</span></div>'
        if n < 5:
            step_html += '<div class="step-line"></div>'
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)

    # Mode Selection
    if st.session_state.assessment_step == 'mode_selection':
        st.markdown("## Choose your assessment")
        st.markdown("Select the screening depth that fits your situation. You can start with a quick screening or complete the full assessment.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="mode-card">
                <div class="mode-icon">⚡</div>
                <h3>Quick Screening</h3>
                <p>A focused assessment using the questions identified as most predictive by the ML model.</p>
                <ul>
                    <li>About 7 minutes</li>
                    <li>Top-ranked BDI & FCRI items</li>
                    <li>Optimized for initial screening</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Start Quick Screening", width='stretch', type="primary"):
                st.session_state.mode = 'quick'
                top_bdi_nums, top_fcri_cols = load_top_items(TOP_N_BDI, TOP_N_FCRI)
                st.session_state.top_bdi_nums = top_bdi_nums
                st.session_state.top_fcri_cols = top_fcri_cols
                st.session_state.assessment_step = 'demographics'
                st.rerun()

        with col2:
            st.markdown("""
            <div class="mode-card">
                <div class="mode-icon">📋</div>
                <h3>Full Assessment</h3>
                <p>A more comprehensive assessment covering all BDI questions and key FCRI items.</p>
                <ul>
                    <li>About 15 minutes</li>
                    <li>Comprehensive clinical coverage</li>
                    <li>Best for detailed analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📋 Start Full Assessment", width='stretch', type="primary"):
                st.session_state.mode = 'full'
                st.session_state.top_bdi_nums = None
                st.session_state.top_fcri_cols = None
                st.session_state.assessment_step = 'demographics'
                st.rerun()

        st.info("💡 **Tip**: Quick mode uses machine learning feature importance to select the most predictive questions. Full mode provides a comprehensive clinical assessment.")

    # Demographics Collection
    elif st.session_state.assessment_step == 'demographics':
        st.markdown("""
        <div class="section-card">
            <div class="section-eyebrow">Step 2 · Profile</div>
            <h3>👤 About you</h3>
            <p>Please provide the basic information below. These details are used as model features to improve screening accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("demographics_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Gender**")
                gender = st.selectbox(
                    "Select your gender",
                    ['Male', 'Female', 'Other', 'Prefer not to say'],
                    index=0,
                    label_visibility="collapsed"
                )

            with col2:
                st.markdown("**Age Group**")
                age_group = st.selectbox(
                    "Select your age group",
                    ['Under 18', '18-30', '31-45', '46-60', 'Over 60'],
                    index=2,
                    label_visibility="collapsed"
                )

            with col3:
                st.markdown("**Education Level**")
                education_level = st.selectbox(
                    "Select your education level",
                    ['No formal education', 'Primary', 'Secondary / Matric',
                     'Intermediate / FSc', "Bachelor's", "Master's or higher"],
                    index=2,
                    label_visibility="collapsed"
                )

            col4, col5 = st.columns(2)

            with col4:
                st.markdown("**Cancer Stage**")
                cancer_stage = st.selectbox(
                    "Select your cancer stage",
                    ['Stage I', 'Stage II', 'Stage III', 'Stage IV',
                     'Metastatic / Advanced', 'Not sure / not told'],
                    index=0,
                    label_visibility="collapsed"
                )

            with col5:
                st.markdown("**Current Cancer Status**")
                cancer_status = st.selectbox(
                    "Select your current status",
                    ['Newly Diagnosed', 'Under Active Treatment', 'In Remission',
                     'Cancer Recurrence', 'Metastatic Disease', 'Palliative Care'],
                    index=0,
                    label_visibility="collapsed"
                )

            submitted = st.form_submit_button("Continue to BDI Assessment →", width='stretch', type="primary")
            if submitted:
                st.session_state.demographics = {
                    'gender': gender,
                    'age_group': age_group,
                    'education_level': education_level,
                    'cancer_stage': cancer_stage,
                    'cancer_status': cancer_status,
                }
                st.session_state.assessment_step = 'bdi_assessment'
                st.rerun()

    # BDI Assessment
    elif st.session_state.assessment_step == 'bdi_assessment':
        # Determine which BDI items to show
        if st.session_state.mode == 'quick':
            items_to_show = [q for q in BDI_QUESTIONS if q['num'] in st.session_state.top_bdi_nums]
            n_total = len(BDI_QUESTIONS)
            n_asked = len(items_to_show)
            st.markdown(f"""
            <div class='section-card'>
            <div class="section-eyebrow">Step 3 · BDI</div>
            <h3>🧠 Beck Depression Inventory — Quick Mode</h3>
            <p>Showing {n_asked} of {n_total} BDI items based on feature importance. Remaining items will be estimated from training averages.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            items_to_show = BDI_QUESTIONS
            n_total = len(BDI_QUESTIONS)
            n_asked = len(items_to_show)
            st.markdown(f"""
            <div class='section-card'>
            <div class="section-eyebrow">Step 3 · BDI</div>
            <h3>🧠 Beck Depression Inventory — Full Assessment</h3>
            <p>Showing all {n_total} BDI items.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="instruction">
            <strong>How to answer</strong><br>
            Choose the statement that best describes how you have been feeling <strong>during the past week, including today</strong>.
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"📝 {len(items_to_show)} questions in this section — all answers are required before continuing.")

        with st.form("bdi_form"):
            bdi_responses = {}

            for i, item in enumerate(items_to_show, 1):
                st.markdown(f"""
                <div class="question-card">
                    <div class="question-heading">{i}. {item['title']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Create user-friendly options without numbers
                user_friendly_options = [opt.split(' - ', 1)[1] if ' - ' in opt else opt for opt in item['options']]

                response = st.radio(
                    f"Select your response for {item['title']}",
                    user_friendly_options,
                    key=f"bdi_{item['num']}",
                    label_visibility="collapsed"
                )

                # Map back to original option to get score
                original_index = user_friendly_options.index(response)
                score = original_index  # 0-based index matches the score
                bdi_responses[f"{item['num']}."] = score

            submitted = st.form_submit_button("Continue to FCRI Assessment →", width='stretch', type="primary")
            if submitted:
                st.session_state.bdi_responses = bdi_responses

                # Check for suicidal ideation (BDI item 9, score >= 2)
                # Item 9 is always included in clinical defaults and typically ranks high in importance
                if bdi_responses.get('9.', 0) >= 2:
                    st.session_state.show_suicide_warning = True
                    st.error("⚠️ **IMPORTANT**: Your response to the question about suicidal thoughts indicates you may be in distress.")
                    st.error("Please contact a crisis helpline or healthcare provider now:")
                    for name, number in list(CRISIS_RESOURCES.items())[:3]:
                        st.error(f"**{name}**: {number}")
                    st.warning("You can continue with the assessment, but please reach out for support.")
                else:
                    st.session_state.show_suicide_warning = False

                st.session_state.assessment_step = 'fcri_assessment'
                st.rerun()

    # FCRI Assessment
    elif st.session_state.assessment_step == 'fcri_assessment':
        # Determine which FCRI items to show
        if st.session_state.mode == 'quick':
            items_to_show = [FCRI_ALL_ITEMS[c] for c in st.session_state.top_fcri_cols if c in FCRI_ALL_ITEMS]
            n_total = 42
            n_asked = len(items_to_show)
            st.markdown(f"""
            <div class='section-card'>
            <div class="section-eyebrow">Step 4 · FCRI</div>
            <h3>😰 Fear of Cancer Recurrence Inventory — Quick Mode</h3>
            <p>Showing {n_asked} of {n_total} FCRI items based on feature importance. Remaining items will be estimated from training averages.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            items_to_show = FCRI_KEY_ITEMS
            n_total = 42
            n_asked = len(items_to_show)
            st.markdown(f"""
            <div class='section-card'>
            <div class="section-eyebrow">Step 4 · FCRI</div>
            <h3>😰 Fear of Cancer Recurrence Inventory — Full Assessment</h3>
            <p>Showing {n_asked} key FCRI items covering all subscales.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="instruction warning">
            <strong>How to answer</strong><br>
            Indicate to what degree each statement applied to you <strong>during the past month</strong>.
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"📝 {len(items_to_show)} questions in this section — all answers are required before submitting.")

        with st.form("fcri_form"):
            fcri_responses = {}

            for i, item in enumerate(items_to_show, 1):
                st.markdown(f"""
                <div class="question-card">
                    <div class="question-heading">{i}. {item['label']}</div>
                    <div class="question-meta">Subscale: {item['subscale']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Create user-friendly options without numbers
                user_friendly_options = [opt.split(' - ', 1)[1] if ' - ' in opt else opt for opt in item['options']]

                response = st.radio(
                    f"Select your response for {item['label']}",
                    user_friendly_options,
                    key=f"fcri_{i}",
                    label_visibility="collapsed"
                )

                # Map back to original option to get score
                original_index = user_friendly_options.index(response)
                score = original_index  # 0-based index matches the score
                fcri_responses[item['col']] = score

            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("📊 Submit Assessment", width='stretch', type="primary")
            with col2:
                back = st.form_submit_button("← Back to BDI", width='stretch', type="secondary")

            if submitted:
                st.session_state.fcri_responses = fcri_responses

                # Build feature vector and run prediction
                feature_vec = build_feature_vector(
                    st.session_state.bdi_responses,
                    st.session_state.fcri_responses,
                    st.session_state.demographics,
                    st.session_state.artefacts['feature_cols'],
                    st.session_state.artefacts['feature_means'],
                    st.session_state.artefacts['label_encoders'],
                )

                pred = predict(feature_vec, st.session_state.artefacts)
                bdi_ref = bdi_reference(st.session_state.bdi_responses)

                # Store current assessment
                st.session_state.current_assessment = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'mode': st.session_state.mode,
                    'demographics': st.session_state.demographics.copy(),
                    'bdi_responses': st.session_state.bdi_responses.copy(),
                    'fcri_responses': st.session_state.fcri_responses.copy(),
                    'prediction': pred,
                    'bdi_reference': bdi_ref,
                }

                # Add to history
                st.session_state.history.append(st.session_state.current_assessment.copy())

                st.session_state.assessment_step = 'results'
                st.rerun()

            if back:
                st.session_state.assessment_step = 'bdi_assessment'
                st.rerun()

    # Results Display (show directly in current page)
    elif st.session_state.assessment_step == 'results':
        display_assessment_results(st.session_state.current_assessment)

        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 New Assessment", width='stretch'):
                # Reset assessment state
                st.session_state.assessment_step = 'mode_selection'
                st.session_state.demographics = {}
                st.session_state.bdi_responses = {}
                st.session_state.fcri_responses = {}
                st.session_state.current_assessment = None
                st.session_state.mode = None
                st.rerun()

        with col2:
            if st.button("📜 View History", width='stretch'):
                st.switch_page("pages/3_History.py")

        with col3:
            if st.button("🏠 Back to Home", width='stretch'):
                st.switch_page("streamlit_app.py")

if __name__ == '__main__':
    assessment_page()