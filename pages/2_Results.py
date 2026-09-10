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

def results_page():
    st.set_page_config(
        page_title="Results - Depression Screening",
        page_icon="📊",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #357abd;
    }
    /* Fix sidebar colors in dark mode */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff;
    }
    /* Consistent card spacing */
    div[data-testid="stVerticalBlock"] > div {
        gap: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Header with gradient
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 2rem; color: white;'>
    <h1 style='margin: 0; font-size: 2rem;'>📊 Assessment Results</h1>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
    View your detailed assessment results and export reports
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Check if there's a current assessment
    if 'current_assessment' not in st.session_state or st.session_state.current_assessment is None:
        st.warning("No assessment results available. Please complete an assessment first.")
        if st.button("Go to Assessment Page"):
            st.switch_page("pages/1_Assessment.py")
        return

    assessment = st.session_state.current_assessment
    pred = assessment['prediction']
    bdi_ref = assessment['bdi_reference']
    demographics = assessment['demographics']
    mode = assessment['mode']

    # Display assessment info
    st.markdown(f"**Assessment completed:** {assessment['timestamp']}")
    st.markdown(f"**Mode:** {mode.capitalize()} Assessment")

    # Display demographics summary
    with st.expander("👤 Demographics Information"):
        st.json(demographics)

    st.markdown("---")

    # Primary Results
    st.markdown("## Primary Results")

    # Severity classification with color coding
    severity_idx = pred['severity_idx']
    severity_label = pred['severity_label']

    # Color coding based on severity
    if severity_idx <= 1:  # Normal or Mild
        severity_color = "🟢"
    elif severity_idx == 2:  # Borderline
        severity_color = "🟡"
    elif severity_idx == 3:  # Moderate
        severity_color = "🟠"
    else:  # Severe or Extreme
        severity_color = "🔴"

    st.markdown(f"### {severity_color} ML Model Severity: {severity_label}")
    st.markdown(f"**Model Confidence:** {pred['confidence']:.1f}%")
    st.markdown(f"**Depression Probability:** {pred['depression_prob']:.1f}%")

    # Progress bar for confidence
    st.progress(pred['confidence'] / 100, text=f"Model Confidence: {pred['confidence']:.1f}%")

    st.markdown("---")

    # Severity probability breakdown
    st.markdown("## Severity Probability Breakdown")

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

    # Display as table
    st.table(prob_df.style.format({'Probability': '{:.1f}%'}))

    st.markdown("---")

    # BDI Reference Score
    st.markdown("## BDI Reference Score")

    bdi_raw, bdi_asked, bdi_extrap, bdi_rule_label = bdi_ref

    if bdi_asked == 21:
        st.markdown(f"**BDI Score (all 21 items):** {bdi_raw}/63 → {bdi_rule_label}")
    else:
        st.markdown(f"**BDI Score ({bdi_asked}/{21} items asked):** {bdi_raw}/{bdi_asked * 3}")
        st.markdown(f"**Extrapolated BDI Score:** {bdi_extrap}/63 → {bdi_rule_label}")
        st.info("💡 **Note**: Quick-mode BDI score is extrapolated from fewer items. Run a Full Assessment for a validated 21-item BDI score.")

    st.markdown("*Rule-based score is informational; ML model output drives the assessment.*")

    st.markdown("---")

    # Clinical description
    st.markdown("## Clinical Assessment")
    desc = SEVERITY_DESCRIPTIONS.get(severity_label, '')
    if desc:
        st.info(desc)

    # Clinical follow-up flag
    if bdi_extrap >= CLINICAL_FOLLOW_UP_THRESHOLD or severity_idx >= 2:
        st.warning("⚠️ **CLINICAL FOLLOW-UP RECOMMENDED**")
        st.warning("Please discuss these results with your oncologist or a mental health professional at the earliest opportunity.")

    # Crisis resources for severe cases
    if severity_idx >= 3:  # moderate or above
        st.markdown("---")
        st.markdown("## 🚨 Crisis & Support Resources (Pakistan)")
        st.error("If you are in immediate distress, please contact these resources:")

        crisis_cols = st.columns(2)
        for i, (name, number) in enumerate(CRISIS_RESOURCES.items()):
            with crisis_cols[i % 2]:
                st.error(f"**{name}**")
                st.error(f"📞 {number}")

    st.markdown("---")

    # Export functionality
    st.markdown("## Export Results")

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
            label="📥 Download CSV Report",
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
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name=f"depression_assessment_{assessment['timestamp'].replace(' ', '_').replace(':', '-')}.pdf",
            mime="application/pdf",
            width='stretch'
        )

    st.markdown("---")

    # Disclaimer
    st.markdown("## Disclaimer")
    st.warning("""
    ⚠️ **DISCLAIMER**: This tool supports clinical screening only.
    It is **NOT** a diagnostic instrument. Always consult a qualified
    healthcare professional for diagnosis and treatment.

    If you are experiencing thoughts of self-harm, please contact emergency
    services or a crisis helpline immediately.
    """)

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
