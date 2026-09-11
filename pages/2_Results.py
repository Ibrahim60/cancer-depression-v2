"""
===================================================================================
STREAMLIT RESULTS PAGE — Depression Prediction for Cancer Patients
===================================================================================
This page displays assessment results with visual formatting, severity probabilities,
clinical recommendations, and export functionality.
"""

import streamlit as st
import sys
import os
import pandas as pd

# Add parent directory to path to import from streamlit_app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app import (
    SEVERITY_LABELS, SEVERITY_DESCRIPTIONS, CRISIS_RESOURCES,
    CLINICAL_FOLLOW_UP_THRESHOLD, export_results_to_csv, export_results_to_pdf,
    init_session_state
)

# Severity index -> semantic tier, used to pick which .alert-card-<tier>
# class colors the results hero and crisis cards.
SEVERITY_TIERS = {
    0: ("success", "🟢"),
    1: ("success", "🟢"),
    2: ("warning", "🟡"),
    3: ("elevated", "🟠"),
}

def severity_tier(severity_idx: int):
    return SEVERITY_TIERS.get(severity_idx, ("danger", "🔴"))

def results_page():
    st.set_page_config(
        page_title="Results - Depression Screening",
        page_icon="📊",
        layout="wide"
    )

    # Comprehensive CSS design system, locked to a single light theme
    # regardless of the visitor's OS/browser preference or Streamlit's own
    # dark-mode toggle.
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

        .result-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Header with gradient
    st.markdown("""
    <div class="hero">
    <span class="eyebrow">Assessment Results</span>
    <h1>📊 Your Assessment Results</h1>
    <p>View your detailed assessment results and export reports</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if there's a current assessment
    if 'current_assessment' not in st.session_state or st.session_state.current_assessment is None:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
        <h3>📭 No results yet</h3>
        <p>Complete an assessment first to see your results here.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📋 Go to Assessment Page", type="primary", width='stretch'):
            st.switch_page("pages/1_Assessment.py")
        return

    assessment = st.session_state.current_assessment
    pred = assessment['prediction']
    bdi_ref = assessment['bdi_reference']
    demographics = assessment['demographics']
    mode = assessment['mode']

    # Display assessment info
    st.markdown(f"""
    <div class="section-card" style="display:flex; gap:2rem; flex-wrap:wrap;">
    <div><span class="section-eyebrow">Completed</span><h3 style="margin:.2rem 0 0;font-size:1.05rem;">{assessment['timestamp']}</h3></div>
    <div><span class="section-eyebrow">Mode</span><h3 style="margin:.2rem 0 0;font-size:1.05rem;">{mode.capitalize()} Assessment</h3></div>
    </div>
    """, unsafe_allow_html=True)

    # Display demographics summary
    with st.expander("👤 Demographics Information"):
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

    prob_df = pd.DataFrame(prob_data)
    prob_df = prob_df.sort_values('Probability', ascending=False)

    # Display as bar chart
    st.bar_chart(prob_df.set_index('Severity')['Probability'])

    # Display as a styled table (matches the assessment-flow results view)
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
        <h4>Clinical Assessment</h4>
        <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Clinical follow-up flag
    if bdi_extrap >= CLINICAL_FOLLOW_UP_THRESHOLD or severity_idx >= 2:
        st.markdown(f"""
        <div class="alert-card alert-warning alert-card--emphasis">
        <h4>⚠️ CLINICAL FOLLOW-UP RECOMMENDED</h4>
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
    This tool supports clinical screening only and is <strong>NOT a diagnostic instrument</strong>.
    Always consult a qualified healthcare professional for diagnosis and treatment.
    </p>
    <p style="margin-top:.6rem;">
    If you are experiencing thoughts of self-harm, please contact emergency services or a crisis helpline immediately.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation buttons
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
            st.switch_page("pages/1_Assessment.py")

    with col2:
        if st.button("📜 View History", width='stretch'):
            st.switch_page("pages/3_History.py")

    with col3:
        if st.button("🏠 Back to Home", width='stretch'):
            st.switch_page("streamlit_app.py")

if __name__ == '__main__':
    results_page()