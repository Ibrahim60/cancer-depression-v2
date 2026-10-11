"""
===================================================================================
SHARED RESULTS VIEW — Depression Prediction for Cancer Patients
===================================================================================
Renders a completed assessment in three steps, used by both the Assessment
page's results step and the Results page:

  1. BDI results   — every answered item, raw/extrapolated score, severity
  2. FCRI results  — every answered item, subscale scores, Severity subscale
  3. Model result  — the ML model's depression detection from BDI + FCRI,
                     giving the final cancer-related depression verdict

Relies on the CSS classes (.alert-card, .metric-card, .result-hero, ...) that
each page injects.
"""

import pandas as pd
import streamlit as st

from streamlit_app import (
    BDI_QUESTIONS, FCRI_QUESTIONS, SEVERITY_LABELS, SEVERITY_DESCRIPTIONS,
    CRISIS_RESOURCES, CLINICAL_FOLLOW_UP_THRESHOLD,
    FCRI_SEVERITY_SCREEN_CUTOFF, FCRI_SEVERITY_CLINICAL_CUTOFF,
    fcri_reference, final_verdict, export_results_to_csv, export_results_to_pdf,
    suicide_risk, needs_follow_up, needs_crisis_resources,
)

VERDICT_EMOJI = {'YES': '🔴', 'POSSIBLE': '🟡', 'NO': '🟢'}


def _answer_text(options, score):
    """Option text for a 0-based score, without its leading 'N - ' prefix."""
    idx = int(score)
    if not 0 <= idx < len(options):
        return str(score)
    opt = options[idx]
    return opt.split(' - ', 1)[1] if ' - ' in opt else opt


def _metric_card(label, value, footnote):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-footnote">{footnote}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_bdi_section(bdi_responses, bdi_ref):
    st.markdown("## 📈 Step 1 — BDI Results (Depressive Symptoms)")

    bdi_raw, bdi_asked, bdi_extrap, bdi_rule_label = bdi_ref

    col1, col2, col3 = st.columns(3)
    with col1:
        _metric_card("Raw BDI score", f"{bdi_raw}/{bdi_asked * 3}",
                     f"Based on {bdi_asked} of 21 items answered")
    with col2:
        _metric_card("Total BDI score", f"{bdi_extrap}/63",
                     "Extrapolated to 21 items" if bdi_asked != 21 else "Full 21-item score")
    with col3:
        _metric_card("BDI severity", bdi_rule_label,
                     f"≥{CLINICAL_FOLLOW_UP_THRESHOLD} = clinically relevant depression")

    rows = []
    for q in BDI_QUESTIONS:
        key = f"{q['num']}."
        if key not in bdi_responses:
            continue
        score = int(bdi_responses[key])
        rows.append({
            '#': q['num'],
            'Symptom': q['title'],
            'Response': _answer_text(q['options'], score),
            'Score': f"{score}/3",
        })
    st.markdown("#### All BDI answers")
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    if bdi_asked != 21:
        st.info("💡 **Note**: Quick-mode BDI score is extrapolated from fewer items. "
                "Run a Full Assessment for a validated 21-item BDI score.")
    st.caption("BDI ranges: 0–10 Normal · 11–16 Mild mood disturbance · 17–20 Borderline clinical · "
               "21–30 Moderate · 31–40 Severe · 41–63 Extreme depression.")


def _render_fcri_section(fcri_responses, fcri_ref):
    st.markdown("## 🎗️ Step 2 — FCRI Results (Fear of Cancer Recurrence)")

    if not fcri_ref['total_asked']:
        st.info("No FCRI items were answered in this assessment.")
        return

    sev_score = fcri_ref['severity_score']
    sev_value = 'N/A' if sev_score is None else f"{sev_score}/36"

    col1, col2, col3 = st.columns(3)
    with col1:
        _metric_card("Severity subscale", sev_value,
                     "Extrapolated to 9 items" if fcri_ref['severity_extrapolated'] else "Items 9–17")
    with col2:
        _metric_card("Total FCRI score", f"{fcri_ref['total_raw']}/{fcri_ref['total_asked'] * 4}",
                     f"Based on {fcri_ref['total_asked']} of 42 items answered")
    with col3:
        _metric_card("FCRI severity", fcri_ref['severity_label'],
                     f"≥{FCRI_SEVERITY_SCREEN_CUTOFF} = clinically significant fear")

    sub_rows = [
        {'Subscale': name,
         'Score': f"{sub['raw']}/{sub['n_asked'] * 4}",
         'Items answered': f"{sub['n_asked']} of {sub['n_total']}"}
        for name, sub in fcri_ref['subscales'].items() if sub['n_asked']
    ]
    st.markdown("#### Subscale scores")
    st.dataframe(pd.DataFrame(sub_rows), width='stretch', hide_index=True)

    item_rows = []
    for q in FCRI_QUESTIONS:
        if q['col'] not in fcri_responses:
            continue
        score = int(fcri_responses[q['col']])
        item_rows.append({
            '#': q['num'],
            'Subscale': q['subscale'],
            'Item': q['label'],
            'Response': _answer_text(q['options'], score),
            # Item 13 is reverse-scored in the subscale total
            'Score': f"{4 - score}/4 (reversed)" if q['num'] == 13 else f"{score}/4",
        })
    st.markdown("#### All FCRI answers")
    st.dataframe(pd.DataFrame(item_rows), width='stretch', hide_index=True)

    if fcri_ref['severity_extrapolated']:
        st.info("💡 **Note**: Quick-mode FCRI severity score is extrapolated from fewer items. "
                "Run a Full Assessment for a validated score.")
    st.caption(
        f"Severity subscale (items 9–17): ≥{FCRI_SEVERITY_SCREEN_CUTOFF} suggests clinically significant "
        f"fear of recurrence; ≥{FCRI_SEVERITY_CLINICAL_CUTOFF} suggests clinical-level fear of recurrence."
    )


def _render_final_section(pred, bdi_ref, fcri_ref, verdict):
    st.markdown("## 🤖 Step 3 — Model Result: Depression Detection (BDI + FCRI)")
    st.caption("The ML ensemble was trained on all BDI and FCRI answers together; "
               "its prediction below is the final depression-detection result.")

    bdi_extrap, bdi_label = bdi_ref[2], bdi_ref[3]
    sev_score = fcri_ref['severity_score']
    fcri_value = 'N/A' if sev_score is None else f"{sev_score}/36"

    col1, col2 = st.columns(2)
    with col1:
        _metric_card("Model: depression probability", f"{pred['depression_prob']:.1f}%",
                     "Detected" if pred['binary_pred'] else "Not detected")
    with col2:
        _metric_card("Model: predicted severity", pred['severity_label'],
                     f"Confidence {pred['confidence']:.1f}%")

    def flag(positive):
        return '⚠️ Positive' if positive else '✅ Negative'

    st.markdown(f"""
    <div class="result-hero alert-card alert-{verdict['tier']} alert-card--emphasis">
        <div class="result-hero-eyebrow">Final result — does the patient have cancer-related depression?</div>
        <h2>{VERDICT_EMOJI[verdict['decision']]} {verdict['decision']} — {verdict['headline']}</h2>
        <p style="margin-bottom:1rem;">{verdict['summary']}</p>
        <div class="result-grid">
            <div class="result-stat">
                <div class="result-stat-label">Model — clinically relevant depression</div>
                <div class="result-stat-value">{flag(verdict['model_clinical'])}</div>
            </div>
            <div class="result-stat">
                <div class="result-stat-label">FCRI — fear of recurrence ({fcri_value})</div>
                <div class="result-stat-value">{flag(verdict['fcri_positive'])}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not verdict['bdi_agrees']:
        st.warning(f"ℹ️ The model's result differs from the BDI questionnaire score "
                   f"({bdi_extrap}/63, {bdi_label}). A clinician should review this case.")

    with st.expander("🧮 How the final result is decided"):
        st.markdown(f"""
| Model predicted severity | FCRI severity (fear of recurrence) | Final result |
|---|---|---|
| Borderline clinical or worse | ≥ {FCRI_SEVERITY_SCREEN_CUTOFF} | 🔴 **YES** — cancer-related depression |
| Borderline clinical or worse | < {FCRI_SEVERITY_SCREEN_CUTOFF} | 🔴 **YES** — depression, not driven by fear of recurrence |
| Mild mood disturbance | ≥ {FCRI_SEVERITY_SCREEN_CUTOFF} | 🟡 **POSSIBLE** — at risk |
| anything else | any | 🟢 **NO** |

The BDI total (≥ {CLINICAL_FOLLOW_UP_THRESHOLD} = clinically relevant) is compared with the model as a sanity check.
""")

    with st.expander("📊 Model severity probability breakdown"):
        prob_df = pd.DataFrame([
            {'Severity': sev,
             'Probability': float(pred['multi_proba'][i]) * 100 if i < len(pred['multi_proba']) else 0.0}
            for i, sev in enumerate(SEVERITY_LABELS)
        ]).sort_values('Probability', ascending=False)
        st.dataframe(
            prob_df.style.format({'Probability': '{:.1f}%'})
            .background_gradient(cmap='RdYlGn_r', subset=['Probability'])
            .set_properties(**{'text-align': 'center'}),
            width='stretch',
            hide_index=True
        )


def render_suicide_warning():
    """Persistent warning shown when BDI item 9 (Suicidal Thoughts) is scored 2+."""
    lines = "".join(f"<p><strong>{name}</strong>: {number}</p>"
                    for name, number in list(CRISIS_RESOURCES.items())[:3])
    st.markdown(f"""
    <div class="alert-card alert-danger alert-card--emphasis">
    <h4>⚠️ Your answer about suicidal thoughts indicates you may be in distress</h4>
    <p style="margin-bottom:.5rem;">Please contact a crisis helpline or healthcare provider now:</p>
    {lines}
    </div>
    """, unsafe_allow_html=True)


def render_assessment_results(assessment):
    """Render the full BDI → FCRI → final-result report for one assessment."""
    pred = assessment['prediction']
    bdi_ref = assessment['bdi_reference']
    demographics = assessment['demographics']
    bdi_responses = assessment.get('bdi_responses', {})
    fcri_responses = assessment.get('fcri_responses', {})

    fcri_ref = fcri_reference(fcri_responses)
    verdict = final_verdict(pred, bdi_ref, fcri_ref)

    if suicide_risk(bdi_responses):
        render_suicide_warning()

    with st.expander("👤 Demographics Information"):
        st.json(demographics)

    st.markdown("---")
    _render_bdi_section(bdi_responses, bdi_ref)
    st.markdown("---")
    _render_fcri_section(fcri_responses, fcri_ref)
    st.markdown("---")
    _render_final_section(pred, bdi_ref, fcri_ref, verdict)

    # Clinical guidance follows the model's severity, which is the final result
    desc = SEVERITY_DESCRIPTIONS.get(pred['severity_label'], '')
    if desc:
        st.markdown(f"""
        <div class="alert-card alert-info">
        <h4>Clinical Assessment</h4>
        <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    if needs_follow_up(pred, bdi_ref, verdict):
        st.markdown("""
        <div class="alert-card alert-warning alert-card--emphasis">
        <h4>⚠️ Clinical follow-up recommended</h4>
        <p>Please discuss these results with your oncologist or a mental health professional at the earliest opportunity.</p>
        </div>
        """, unsafe_allow_html=True)

    # Crisis resources for moderate depression or above, or suicidal-ideation risk
    if needs_crisis_resources(pred, bdi_ref, bdi_responses):
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
    file_stem = f"depression_assessment_{assessment['timestamp'].replace(' ', '_').replace(':', '-')}"
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Download CSV Report",
            data=export_results_to_csv(demographics, bdi_responses, fcri_responses,
                                       pred, bdi_ref, assessment['timestamp']),
            file_name=f"{file_stem}.csv",
            mime="text/csv",
            width='stretch'
        )
    with col2:
        st.download_button(
            label="⬇️ Download PDF Report",
            data=export_results_to_pdf(demographics, bdi_responses, fcri_responses,
                                       pred, bdi_ref, assessment['timestamp']),
            file_name=f"{file_stem}.pdf",
            mime="application/pdf",
            width='stretch'
        )

    st.markdown("---")

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
