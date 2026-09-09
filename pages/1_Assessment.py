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
from datetime import datetime

# Add parent directory to path to import from streamlit_app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app import (
    BDI_QUESTIONS, FCRI_KEY_ITEMS, FCRI_ALL_ITEMS,
    load_top_items, build_feature_vector, predict, bdi_reference,
    TOP_N_BDI, TOP_N_FCRI, CLINICAL_FOLLOW_UP_THRESHOLD,
    CRISIS_RESOURCES, init_session_state
)

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

    # Results Display (redirect to results page)
    elif st.session_state.assessment_step == 'results':
        st.success("Assessment completed successfully!")
        st.info("Navigate to the **Results** page to view your assessment results.")
        if st.button("Go to Results Page"):
            st.switch_page("pages/2_Results.py")

    # Reset button
    st.markdown("---")
    if st.button("🔄 Start New Assessment"):
        # Reset assessment state
        st.session_state.assessment_step = 'mode_selection'
        st.session_state.demographics = {}
        st.session_state.bdi_responses = {}
        st.session_state.fcri_responses = {}
        st.session_state.current_assessment = None
        st.session_state.mode = None
        st.rerun()

if __name__ == '__main__':
    assessment_page()
