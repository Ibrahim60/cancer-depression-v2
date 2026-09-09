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

def display_assessment_results(assessment):
    """Display assessment results directly in the current page."""
    pred = assessment['prediction']
    bdi_ref = assessment['bdi_reference']
    demographics = assessment['demographics']
    mode = assessment['mode']

    st.markdown("---")
    st.markdown("## 📊 Assessment Results")

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

    # Display as table
    prob_df = pd.DataFrame(prob_data)
    prob_df = prob_df.sort_values('Probability', ascending=False)
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
            use_container_width=True
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
            use_container_width=True
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

def assessment_page():
    st.set_page_config(
        page_title="Assessment - Depression Screening",
        page_icon="📋",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    st.title("📋 Depression Assessment")
    st.markdown("Complete the questionnaire below to assess depression levels.")

    # Initialize session state for assessment (if not already done in init_session_state)
    if 'assessment_step' not in st.session_state:
        st.session_state.assessment_step = 'mode_selection'
    if 'demographics' not in st.session_state:
        st.session_state.demographics = {}
    if 'bdi_responses' not in st.session_state:
        st.session_state.bdi_responses = {}
    if 'fcri_responses' not in st.session_state:
        st.session_state.fcri_responses = {}

    # Mode Selection
    if st.session_state.assessment_step == 'mode_selection':
        st.markdown("### Select Assessment Mode")
        st.markdown("""
        Choose the assessment mode that best fits your needs:

        - **Quick Screening**: Top-ranked items only (~7 minutes, 15 questions)
        - **Full Assessment**: All BDI + key FCRI items (~15 minutes, 30 questions)
        """)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Quick Screening", use_container_width=True):
                st.session_state.mode = 'quick'
                top_bdi_nums, top_fcri_cols = load_top_items(TOP_N_BDI, TOP_N_FCRI)
                st.session_state.top_bdi_nums = top_bdi_nums
                st.session_state.top_fcri_cols = top_fcri_cols
                st.session_state.assessment_step = 'demographics'
                st.rerun()

        with col2:
            if st.button("📋 Full Assessment", use_container_width=True):
                st.session_state.mode = 'full'
                st.session_state.top_bdi_nums = None
                st.session_state.top_fcri_cols = None
                st.session_state.assessment_step = 'demographics'
                st.rerun()

        st.info("💡 **Tip**: Quick mode uses machine learning feature importance to select the most predictive questions. Full mode provides a comprehensive clinical assessment.")

    # Demographics Collection
    elif st.session_state.assessment_step == 'demographics':
        st.markdown("### Demographics")
        st.markdown("Please provide some basic demographic information. This helps improve prediction accuracy.")

        with st.form("demographics_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox(
                    "Gender",
                    ['Male', 'Female', 'Other', 'Prefer not to say'],
                    index=0
                )

            with col2:
                age_group = st.selectbox(
                    "Age Group",
                    ['Under 18', '18-30', '31-45', '46-60', 'Over 60'],
                    index=2
                )

            with col3:
                education_level = st.selectbox(
                    "Education Level",
                    ['No formal education', 'Primary', 'Secondary / Matric',
                     'Intermediate / FSc', "Bachelor's", "Master's or higher"],
                    index=2
                )

            col4, col5 = st.columns(2)

            with col4:
                cancer_stage = st.selectbox(
                    "Cancer Stage",
                    ['Stage I', 'Stage II', 'Stage III', 'Stage IV',
                     'Metastatic / Advanced', 'Not sure / not told'],
                    index=0
                )

            with col5:
                cancer_status = st.selectbox(
                    "Current Cancer Status",
                    ['Newly Diagnosed', 'Under Active Treatment', 'In Remission',
                     'Cancer Recurrence', 'Metastatic Disease', 'Palliative Care'],
                    index=0
                )

            submitted = st.form_submit_button("Continue to BDI Assessment")
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
            st.markdown(f"### Beck Depression Inventory (BDI) — Quick Mode")
            st.info(f"Showing {n_asked} of {n_total} BDI items based on feature importance. Remaining items will be estimated from training averages.")
        else:
            items_to_show = BDI_QUESTIONS
            n_total = len(BDI_QUESTIONS)
            n_asked = len(items_to_show)
            st.markdown(f"### Beck Depression Inventory (BDI) — Full Assessment")
            st.info(f"Showing all {n_total} BDI items.")

        st.markdown("Choose the statement that best describes how you have been feeling **DURING THE PAST WEEK**, including today.")

        with st.form("bdi_form"):
            bdi_responses = {}

            for i, item in enumerate(items_to_show, 1):
                st.markdown(f"**{i}. {item['title']}**")
                response = st.radio(
                    f"Select your response for {item['title']}",
                    item['options'],
                    key=f"bdi_{item['num']}",
                    label_visibility="collapsed"
                )
                # Extract the numeric score (first character of the option)
                score = int(response.split(' - ')[0]) if ' - ' in response else int(response[0])
                bdi_responses[f"{item['num']}."] = score
                st.markdown("---")

            submitted = st.form_submit_button("Continue to FCRI Assessment")
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
            st.markdown(f"### Fear of Cancer Recurrence Inventory (FCRI) — Quick Mode")
            st.info(f"Showing {n_asked} of {n_total} FCRI items based on feature importance. Remaining items will be estimated from training averages.")
        else:
            items_to_show = FCRI_KEY_ITEMS
            n_total = 42
            n_asked = len(items_to_show)
            st.markdown(f"### Fear of Cancer Recurrence Inventory (FCRI) — Full Assessment")
            st.info(f"Showing {n_asked} key FCRI items covering all subscales.")

        st.markdown("Indicate to what degree each statement applied to you **DURING THE PAST MONTH**.")

        with st.form("fcri_form"):
            fcri_responses = {}

            for i, item in enumerate(items_to_show, 1):
                st.markdown(f"**{i}. {item['label']}** ({item['subscale']})")
                response = st.radio(
                    f"Select your response for {item['label']}",
                    item['options'],
                    key=f"fcri_{i}",
                    label_visibility="collapsed"
                )
                # Extract the numeric score (first character of the option)
                score = int(response.split(' - ')[0]) if ' - ' in response else int(response[0])
                fcri_responses[item['col']] = score
                st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Submit Assessment")
            with col2:
                back = st.form_submit_button("Back to BDI")

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
        st.success("✅ Assessment completed successfully!")
        display_assessment_results(st.session_state.current_assessment)

        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 New Assessment", use_container_width=True):
                # Reset assessment state
                st.session_state.assessment_step = 'mode_selection'
                st.session_state.demographics = {}
                st.session_state.bdi_responses = {}
                st.session_state.fcri_responses = {}
                st.session_state.current_assessment = None
                st.session_state.mode = None
                st.rerun()

        with col2:
            if st.button("📜 View History", use_container_width=True):
                st.switch_page("pages/3_History.py")

        with col3:
            if st.button("🏠 Back to Home", use_container_width=True):
                st.switch_page("streamlit_app.py")

if __name__ == '__main__':
    assessment_page()
