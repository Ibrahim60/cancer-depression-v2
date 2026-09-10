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

    # Success banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; color: white; text-align: center;'>
    <h2 style='margin: 0; font-size: 1.8rem;'>✅ Assessment Completed Successfully</h2>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
    {timestamp} • {mode} Assessment
    </p>
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

    # Color coding based on severity
    if severity_idx <= 1:  # Normal or Mild
        bg_color = "#d4edda"
        border_color = "#28a745"
        text_color = "#155724"
        emoji = "🟢"
    elif severity_idx == 2:  # Borderline
        bg_color = "#fff3cd"
        border_color = "#ffc107"
        text_color = "#856404"
        emoji = "🟡"
    elif severity_idx == 3:  # Moderate
        bg_color = "#ffeeba"
        border_color = "#fd7e14"
        text_color = "#856404"
        emoji = "🟠"
    else:  # Severe or Extreme
        bg_color = "#f8d7da"
        border_color = "#dc3545"
        text_color = "#721c24"
        emoji = "🔴"

    st.markdown(f"""
    <div style='background: {bg_color}; border: 2px solid {border_color}; padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
    <h2 style='margin: 0 0 1rem 0; color: {text_color};'>{emoji} ML Model Severity: {severity_label}</h2>
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
    <div>
    <p style='margin: 0; color: {text_color};'><strong>Model Confidence:</strong> {pred['confidence']:.1f}%</p>
    </div>
    <div>
    <p style='margin: 0; color: {text_color};'><strong>Depression Probability:</strong> {pred['depression_prob']:.1f}%</p>
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
        <div style='background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: none;'>
        <h4 style='margin: 0 0 0.5rem 0; color: #667eea; font-size: 1.1rem;'>Raw BDI Score</h4>
        <p style='margin: 0; font-size: 2rem; font-weight: bold; color: #764ba2;'>{bdi_raw}/{bdi_asked * 3}</p>
        <p style='margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.9rem;'>Based on {bdi_asked} items answered</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: none;'>
        <h4 style='margin: 0 0 0.5rem 0; color: #667eea; font-size: 1.1rem;'>Extrapolated Score</h4>
        <p style='margin: 0; font-size: 2rem; font-weight: bold; color: #764ba2;'>{bdi_extrap}/63</p>
        <p style='margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.9rem;'>Severity: {bdi_rule_label}</p>
        </div>
        """, unsafe_allow_html=True)

    if bdi_asked != 21:
        st.info("💡 **Note**: Quick-mode BDI score is extrapolated from fewer items. Run a Full Assessment for a validated 21-item BDI score.")

    st.markdown("*Rule-based score is informational; ML model output drives the assessment.*")

    st.markdown("---")

    # Clinical description
    st.markdown("## 🏥 Clinical Assessment")
    desc = SEVERITY_DESCRIPTIONS.get(severity_label, '')
    if desc:
        st.markdown(f"""
        <div style='background: #e7f3ff; border-left: 4px solid #667eea; padding: 1.5rem; border-radius: 8px;'>
        <p style='margin: 0; color: #004085;'><strong>Clinical Assessment:</strong> {desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # Clinical follow-up flag
    if bdi_extrap >= CLINICAL_FOLLOW_UP_THRESHOLD or severity_idx >= 2:
        st.markdown(f"""
        <div style='background: #fff3cd; border: 2px solid #ffc107; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
        <h4 style='margin: 0 0 0.5rem 0; color: #856404;'>⚠️ CLINICAL FOLLOW-UP RECOMMENDED</h4>
        <p style='margin: 0; color: #856404;'>
        Please discuss these results with your oncologist or a mental health professional at the earliest opportunity.
        </p>
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
                <div style='background: #f8d7da; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                <p style='margin: 0; color: #721c24;'><strong>{name}</strong></p>
                <p style='margin: 0.25rem 0 0 0; color: #721c24; font-size: 1.2rem;'>📞 {number}</p>
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
            label="� Download CSV Report",
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
            label="� Download PDF Report",
            data=pdf_data,
            file_name=f"depression_assessment_{assessment['timestamp'].replace(' ', '_').replace(':', '-')}.pdf",
            mime="application/pdf",
            width='stretch'
        )

    st.markdown("---")

    # Disclaimer with better styling
    st.markdown("""
    <div style='background: #fff3cd; border: 2px solid #ffc107; border-radius: 12px; padding: 1.5rem;'>
    <h4 style='margin: 0 0 0.5rem 0; color: #856404;'>⚠️ Important Disclaimer</h4>
    <p style='margin: 0; color: #856404;'>
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
    .stButton>button[type="secondary"] {
        background-color: #6c757d;
    }
    .stButton>button[type="secondary"]:hover {
        background-color: #5a6268;
    }
    .stProgress>div>div>div>div {
        background-color: #667eea;
    }
    .question-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: none;
    }
    /* Better radio button styling */
    .stRadio > div {
        background: transparent;
        padding: 0.5rem 0;
    }
    .stRadio > div > label {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        transition: all 0.2s;
    }
    .stRadio > div > label:hover {
        background: #e9ecef;
        border-color: #dee2e6;
    }
    .stRadio > div > label[data-testid="stMarkdown"] {
        background: #667eea;
        color: white;
        border-color: #667eea;
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
    <h1 style='margin: 0; font-size: 2rem;'>📋 Depression Assessment</h1>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
    Complete the questionnaire below to assess depression levels
    </p>
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

    # Mode Selection
    if st.session_state.assessment_step == 'mode_selection':
        st.markdown("### Select Assessment Mode")
        st.markdown("Choose the assessment mode that best fits your needs:")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class='question-card' style='border: none;'>
            <h3 style='margin: 0 0 0.5rem 0; color: #667eea; font-size: 1.2rem;'>⚡ Quick Screening</h3>
            <ul style='margin: 0.5rem 0; padding-left: 1.5rem; color: #495057;'>
            <li>Top-ranked items only</li>
            <li>~7 minutes, 15 questions</li>
            <li>ML-optimized selection</li>
            <li>Best for initial screening</li>
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
            <div class='question-card' style='border: none;'>
            <h3 style='margin: 0 0 0.5rem 0; color: #764ba2; font-size: 1.2rem;'>📋 Full Assessment</h3>
            <ul style='margin: 0.5rem 0; padding-left: 1.5rem; color: #495057;'>
            <li>All BDI + key FCRI items</li>
            <li>~15 minutes, 30 questions</li>
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
        <div class='question-card' style='border: none;'>
        <h3 style='margin: 0 0 0.5rem 0; color: #667eea; font-size: 1.2rem;'>👤 Demographics</h3>
        <p style='margin: 0; color: #495057;'>
        Please provide some basic demographic information. This helps improve prediction accuracy.
        </p>
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
            <div class='question-card' style='border: none;'>
            <h3 style='margin: 0 0 0.5rem 0; color: #667eea; font-size: 1.2rem;'>🧠 Beck Depression Inventory (BDI) — Quick Mode</h3>
            <p style='margin: 0; color: #495057;'>
            Showing {n_asked} of {n_total} BDI items based on feature importance. Remaining items will be estimated from training averages.
            </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            items_to_show = BDI_QUESTIONS
            n_total = len(BDI_QUESTIONS)
            n_asked = len(items_to_show)
            st.markdown(f"""
            <div class='question-card' style='border: none;'>
            <h3 style='margin: 0 0 0.5rem 0; color: #667eea; font-size: 1.2rem;'>🧠 Beck Depression Inventory (BDI) — Full Assessment</h3>
            <p style='margin: 0; color: #495057;'>
            Showing all {n_total} BDI items.
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background: #e7f3ff; border-left: 4px solid #667eea; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
        <p style='margin: 0; color: #004085;'>
        <strong>Instructions:</strong> Choose the statement that best describes how you have been feeling <strong>DURING THE PAST WEEK</strong>, including today.
        </p>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        progress = 0
        if 'current_question' not in st.session_state:
            st.session_state.current_question = 0

        st.progress(progress / len(items_to_show), text=f"Progress: {progress}/{len(items_to_show)} questions")

        with st.form("bdi_form"):
            bdi_responses = {}

            for i, item in enumerate(items_to_show, 1):
                st.markdown(f"""
                <div class='question-card' style='border: none;'>
                <h4 style='margin: 0 0 1rem 0; color: #667eea; font-size: 1.1rem;'>{i}. {item['title']}</h4>
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
            <div class='question-card' style='border: none;'>
            <h3 style='margin: 0 0 0.5rem 0; color: #764ba2; font-size: 1.2rem;'>😰 Fear of Cancer Recurrence Inventory (FCRI) — Quick Mode</h3>
            <p style='margin: 0; color: #495057;'>
            Showing {n_asked} of {n_total} FCRI items based on feature importance. Remaining items will be estimated from training averages.
            </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            items_to_show = FCRI_KEY_ITEMS
            n_total = 42
            n_asked = len(items_to_show)
            st.markdown(f"""
            <div class='question-card' style='border: none;'>
            <h3 style='margin: 0 0 0.5rem 0; color: #764ba2; font-size: 1.2rem;'>😰 Fear of Cancer Recurrence Inventory (FCRI) — Full Assessment</h3>
            <p style='margin: 0; color: #495057;'>
            Showing {n_asked} key FCRI items covering all subscales.
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
        <p style='margin: 0; color: #856404;'>
        <strong>Instructions:</strong> Indicate to what degree each statement applied to you <strong>DURING THE PAST MONTH</strong>.
        </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("fcri_form"):
            fcri_responses = {}

            for i, item in enumerate(items_to_show, 1):
                st.markdown(f"""
                <div class='question-card' style='border: none; background: white;'>
                <h4 style='margin: 0 0 0.5rem 0; color: #764ba2; font-size: 1.1rem;'>{i}. {item['label']}</h4>
                <p style='margin: 0 0 1rem 0; color: #6c757d; font-size: 0.9rem;'>Subscale: {item['subscale']}</p>
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
        st.success("✅ Assessment completed successfully!")
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
