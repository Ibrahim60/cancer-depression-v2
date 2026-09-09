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

def history_page():
    st.set_page_config(
        page_title="History - Depression Screening",
        page_icon="📜",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    st.title("📜 Assessment History")

    # Check if there's any history
    if not st.session_state.history:
        st.info("No assessment history available. Complete an assessment to see it here.")
        if st.button("Go to Assessment Page"):
            st.switch_page("pages/1_Assessment.py")
        return

    # Display history summary
    st.markdown(f"**Total assessments:** {len(st.session_state.history)}")

    # Export history button
    csv_data = export_history_to_csv(st.session_state.history)
    if csv_data:
        st.download_button(
            label="📥 Export History to CSV",
            data=csv_data,
            file_name=f"assessment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")

    # Display history as expandable sections
    for i, assessment in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Assessment #{i} - {assessment['timestamp']} ({assessment['mode'].capitalize()} Mode)"):
            # Demographics
            st.markdown("### 👤 Demographics")
            st.json(assessment['demographics'])

            # Results summary
            pred = assessment['prediction']
            bdi_ref = assessment['bdi_reference']

            st.markdown("### 📊 Results Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Severity", pred['severity_label'])
            with col2:
                st.metric("Confidence", f"{pred['confidence']:.1f}%")
            with col3:
                st.metric("Depression Prob", f"{pred['depression_prob']:.1f}%")

            st.markdown(f"**BDI Score:** {bdi_ref[0]}/{bdi_ref[1] * 3} (extrapolated: {bdi_ref[2]}/63)")
            st.markdown(f"**BDI Severity:** {bdi_ref[3]}")

            # View detailed responses
            if st.button(f"View Detailed Responses - Assessment #{i}", key=f"view_details_{i}"):
                st.markdown("### BDI Responses")
                for item_num, score in assessment['bdi_responses'].items():
                    st.markdown(f"**Item {item_num}:** Score = {score}")

                st.markdown("### FCRI Responses")
                for item_col, score in assessment['fcri_responses'].items():
                    st.markdown(f"**{item_col}:** Score = {score}")

            st.markdown("---")

    # Clear history button
    st.markdown("---")
    st.warning("⚠️ **Warning**: Clearing history will permanently delete all assessment records.")
    if st.button("🗑️ Clear All History", type="secondary"):
        st.session_state.history = []
        st.success("History cleared successfully!")
        st.rerun()

    st.markdown("---")

    # Navigation buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 New Assessment", use_container_width=True):
            st.session_state.assessment_step = 'mode_selection'
            st.session_state.demographics = {}
            st.session_state.bdi_responses = {}
            st.session_state.fcri_responses = {}
            st.session_state.current_assessment = None
            st.session_state.mode = None
            st.switch_page("pages/1_Assessment.py")

    with col2:
        if st.button("📊 View Latest Results", use_container_width=True):
            if st.session_state.current_assessment:
                st.switch_page("pages/2_Results.py")
            else:
                st.warning("No current assessment available.")

    with col3:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("streamlit_app.py")

if __name__ == '__main__':
    history_page()
