"""
===================================================================================
STREAMLIT WEB APPLICATION — Depression Prediction for Cancer Patients
===================================================================================
Version: 1.0

This is the main entry point for the Streamlit web application that wraps the
CLI-based depression prediction tool. It provides a multi-page interface with
assessment, results, and history pages.

Usage
-----
    streamlit run streamlit_app.py

Artefacts required (produced by depression_detection_pipeline.py)
---------
    models/binary_voting_ensemble_model.joblib
    models/multiclass_voting_ensemble_model.joblib
    models/binary_scaler.joblib
    models/label_encoders.joblib
    models/feature_metadata.json
"""

import os
import sys
import json
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = 'models'

SEVERITY_LABELS = [
    'Normal',
    'Mild mood disturbance',
    'Borderline clinical depression',
    'Moderate depression',
    'Severe depression',
    'Extreme depression',
]

SEVERITY_DESCRIPTIONS = {
    'Normal':                          'Your score is within the normal range. '
                                       'Monitor your mood and seek support if needed.',
    'Mild mood disturbance':           'You are experiencing a mild mood disturbance. '
                                       'Consider speaking with a counsellor.',
    'Borderline clinical depression':  'Your score is in the borderline clinical range. '
                                       'A consultation with a mental health professional is recommended.',
    'Moderate depression':             'Your responses indicate moderate depression. '
                                       'Please seek professional mental health support promptly.',
    'Severe depression':               'Your responses indicate severe depression. '
                                       'Immediate professional intervention is strongly recommended.',
    'Extreme depression':              'Your responses indicate extreme depression. '
                                       'Please seek immediate professional support.',
}

CRISIS_RESOURCES = {
    'Umang Helpline (Pakistan)':                 '0317-4288665',
    'Rozan Counselling Helpline':                '051-2890505',
    'Edhi Foundation 24/7 Helpline':             '115',
    'Shaukat Khanum Psycho-Oncology Department': '042-35945100',
    'Pakistan Association for Mental Health':    '021-34389555',
}

CLINICAL_FOLLOW_UP_THRESHOLD = 17
TOP_N_BDI = 10
TOP_N_FCRI = 5

# =============================================================================
# BDI QUESTIONS (imported from predict_depression.py)
# =============================================================================

BDI_QUESTIONS = [
    {
        'num': 1, 'title': 'Sadness',
        'options': [
            '0 - I do not feel sad.',
            '1 - I feel sad.',
            '2 - I am sad all the time and I cannot snap out of it.',
            '3 - I am so sad and unhappy that I cannot stand it.',
        ]
    },
    {
        'num': 2, 'title': 'Pessimism',
        'options': [
            '0 - I am not particularly discouraged about the future.',
            '1 - I feel discouraged about the future.',
            '2 - I feel I have nothing to look forward to.',
            '3 - I feel the future is hopeless and that things cannot improve.',
        ]
    },
    {
        'num': 3, 'title': 'Past Failure',
        'options': [
            '0 - I do not feel like a failure.',
            '1 - I feel I have failed more than the average person.',
            '2 - As I look back on my life, all I can see is a lot of failures.',
            '3 - I feel I am a complete failure as a person.',
        ]
    },
    {
        'num': 4, 'title': 'Loss of Pleasure',
        'options': [
            '0 - I get as much satisfaction out of things as I used to.',
            '1 - I do not enjoy things the way I used to.',
            '2 - I do not get real satisfaction out of anything anymore.',
            '3 - I am dissatisfied or bored with everything.',
        ]
    },
    {
        'num': 5, 'title': 'Guilty Feelings',
        'options': [
            '0 - I do not feel particularly guilty.',
            '1 - I feel guilty a good part of the time.',
            '2 - I feel quite guilty most of the time.',
            '3 - I feel guilty all of the time.',
        ]
    },
    {
        'num': 6, 'title': 'Punishment Feelings',
        'options': [
            '0 - I do not feel I am being punished.',
            '1 - I feel I may be punished.',
            '2 - I expect to be punished.',
            '3 - I feel I am being punished.',
        ]
    },
    {
        'num': 7, 'title': 'Self-Dislike',
        'options': [
            '0 - I do not feel disappointed in myself.',
            '1 - I am disappointed in myself.',
            '2 - I am disgusted with myself.',
            '3 - I hate myself.',
        ]
    },
    {
        'num': 8, 'title': 'Self-Criticalness',
        'options': [
            '0 - I do not feel I am any worse than anybody else.',
            '1 - I am critical of myself for my weaknesses or mistakes.',
            '2 - I blame myself all the time for my faults.',
            '3 - I blame myself for everything bad that happens.',
        ]
    },
    {
        'num': 9, 'title': 'Suicidal Thoughts',
        'options': [
            '0 - I do not have any thoughts of killing myself.',
            '1 - I have thoughts of killing myself, but I would not carry them out.',
            '2 - I would like to kill myself.',
            '3 - I would kill myself if I had the chance.',
        ]
    },
    {
        'num': 10, 'title': 'Crying',
        'options': [
            '0 - I do not cry any more than usual.',
            '1 - I cry more now than I used to.',
            '2 - I cry all the time now.',
            '3 - I used to be able to cry, but now I cannot cry even though I want to.',
        ]
    },
    {
        'num': 11, 'title': 'Agitation',
        'options': [
            '0 - I am no more irritated by things than I ever was.',
            '1 - I am slightly more irritated now than usual.',
            '2 - I am quite annoyed or irritated a good deal of the time.',
            '3 - I feel irritated all the time.',
        ]
    },
    {
        'num': 12, 'title': 'Loss of Interest',
        'options': [
            '0 - I have not lost interest in other people.',
            '1 - I am less interested in other people than I used to be.',
            '2 - I have lost most of my interest in other people.',
            '3 - I have lost all of my interest in other people.',
        ]
    },
    {
        'num': 13, 'title': 'Indecisiveness',
        'options': [
            '0 - I make decisions about as well as I ever could.',
            '1 - I put off making decisions more than I used to.',
            '2 - I have greater difficulty making decisions than I used to.',
            '3 - I cannot make decisions at all anymore.',
        ]
    },
    {
        'num': 14, 'title': 'Body Image',
        'options': [
            '0 - I do not feel that I look any worse than I used to.',
            '1 - I am worried that I am looking old or unattractive.',
            '2 - I feel there are permanent changes in my appearance that make me look unattractive.',
            '3 - I believe that I look ugly.',
        ]
    },
    {
        'num': 15, 'title': 'Work Inhibition',
        'options': [
            '0 - I can work about as well as before.',
            '1 - It takes an extra effort to get started at doing something.',
            '2 - I have to push myself very hard to do anything.',
            '3 - I cannot do any work at all.',
        ]
    },
    {
        'num': 16, 'title': 'Sleep Disturbance',
        'options': [
            '0 - I can sleep as well as usual.',
            '1 - I do not sleep as well as I used to.',
            '2 - I wake up 1-2 hours earlier than usual and find it hard to get back to sleep.',
            '3 - I wake up several hours earlier than I used to and cannot get back to sleep.',
        ]
    },
    {
        'num': 17, 'title': 'Fatigability',
        'options': [
            '0 - I do not get more tired than usual.',
            '1 - I get tired more easily than I used to.',
            '2 - I get tired from doing almost anything.',
            '3 - I am too tired to do anything.',
        ]
    },
    {
        'num': 18, 'title': 'Appetite Changes',
        'options': [
            '0 - My appetite is no worse than usual.',
            '1 - My appetite is not as good as it used to be.',
            '2 - My appetite is much worse now.',
            '3 - I have no appetite at all anymore.',
        ]
    },
    {
        'num': 19, 'title': 'Weight Loss',
        'options': [
            '0 - I have not lost much weight, if any, lately.',
            '1 - I have lost more than five pounds.',
            '2 - I have lost more than ten pounds.',
            '3 - I have lost more than fifteen pounds.',
        ]
    },
    {
        'num': 20, 'title': 'Somatic Preoccupation',
        'options': [
            '0 - I am no more worried about my health than usual.',
            '1 - I am worried about physical problems like aches, pains, upset stomach, or constipation.',
            '2 - I am very worried about physical problems and it is hard to think of much else.',
            '3 - I am so worried about my physical problems that I cannot think of anything else.',
        ]
    },
    {
        'num': 21, 'title': 'Loss of Interest in Sex',
        'options': [
            '0 - I have not noticed any recent change in my interest in sex.',
            '1 - I am less interested in sex than I used to be.',
            '2 - I have almost no interest in sex.',
            '3 - I have lost interest in sex completely.',
        ]
    },
]

# =============================================================================
# FCRI KEY ITEMS
# =============================================================================

FCRI_KEY_ITEMS = [
    {
        'col':     '9. I am worried or anxious about the possibility of cancer recurrence',
        'label':   'FCRI: Worry about recurrence',
        'subscale':'Severity',
        'options': ['0 - Not at all', '1 - A little', '2 - Somewhat',
                    '3 - A lot', '4 - A great deal'],
    },
    {
        'col':     '14. In your opinion, are you at risk of having a cancer recurrence?',
        'label':   'FCRI: Perceived risk of recurrence',
        'subscale':'Severity',
        'options': ['0 - Not at all at risk', '1 - A little at risk', '2 - Somewhat at risk',
                    '3 - A lot at risk', '4 - A great deal at risk'],
    },
    {
        'col':     '15. How often do you think about the possibility of cancer recurrence?',
        'label':   'FCRI: Frequency of recurrence thoughts',
        'subscale':'Severity',
        'options': ['0 - Never', '1 - A few times a month', '2 - A few times a week',
                    '3 - A few times a day', '4 - Several times a day'],
    },
    {
        'col':     '18. Worry, fear or anxiety',
        'label':   'FCRI: Emotional distress — worry or fear',
        'subscale':'Psychological Distress',
        'options': ['0 - Not at all', '1 - A little', '2 - Somewhat',
                    '3 - A lot', '4 - A great deal'],
    },
    {
        'col':     '22. My social or leisure activities (e.g. outings, sports, travel)',
        'label':   'FCRI: Impact on social activities',
        'subscale':'Functioning Impairments',
        'options': ['0 - Not at all', '1 - A little', '2 - Somewhat',
                    '3 - A lot', '4 - A great deal'],
    },
    {
        'col':     '26. My state of mind or my mood',
        'label':   'FCRI: Impact on state of mind',
        'subscale':'Functioning Impairments',
        'options': ['0 - Not at all', '1 - A little', '2 - Somewhat',
                    '3 - A lot', '4 - A great deal'],
    },
    {
        'col':     '28. I feel that I worry excessively about the possibility of cancer recurrence',
        'label':   'FCRI: Awareness of excessive worry',
        'subscale':'Insight',
        'options': ['0 - Not at all', '1 - A little', '2 - Somewhat',
                    '3 - A lot', '4 - A great deal'],
    },
    {
        'col':     '31. I call my doctor or other health professional',
        'label':   'FCRI: Seeking reassurance from doctor',
        'subscale':'Reassurance',
        'options': ['0 - Never', '1 - Rarely', '2 - Sometimes',
                    '3 - Most of the time', '4 - All the time'],
    },
    {
        'col':     '36. I pray, meditate or do relaxation',
        'label':   'FCRI: Use of relaxation or prayer',
        'subscale':'Coping Strategies',
        'options': ['0 - Never', '1 - Rarely', '2 - Sometimes',
                    '3 - Most of the time', '4 - All the time'],
    },
]

FCRI_EXTRA_ITEMS = [
    {
        'col':     '13. I believe that I am cured and that the cancer will not come back',
        'label':   'FCRI: Belief in cure (reverse-scored)',
        'subscale':'Severity',
        'options': ['0 - Not at all', '1 - A little', '2 - Somewhat',
                    '3 - A lot', '4 - A great deal'],
    },
    {
        'col':     '16. How much time per day do you spend thinking about the possibility of cancer recurrence?',
        'label':   'FCRI: Time spent per day thinking about recurrence',
        'subscale':'Severity',
        'options': ["0 - I don't think about it", '1 - A few seconds',
                    '2 - A few minutes', '3 - A few hours', '4 - Several hours'],
    },
    {
        'col':     '23. My work or everyday activities',
        'label':   'FCRI: Impact on work or everyday activities',
        'subscale':'Functioning Impairments',
        'options': ['0 - Not at all', '1 - A little', '2 - Somewhat',
                    '3 - A lot', '4 - A great deal'],
    },
    {
        'col':     '10. I am afraid of cancer recurrence',
        'label':   'FCRI: Fear of recurrence',
        'subscale':'Severity',
        'options': ['0 - Not at all', '1 - A little', '2 - Somewhat',
                    '3 - A lot', '4 - A great deal'],
    },
    {
        'col':     '27. My quality of life in general',
        'label':   'FCRI: Impact on quality of life',
        'subscale':'Functioning Impairments',
        'options': ['0 - Not at all', '1 - A little', '2 - Somewhat',
                    '3 - A lot', '4 - A great deal'],
    },
]

FCRI_ALL_ITEMS = {item['col']: item for item in FCRI_KEY_ITEMS + FCRI_EXTRA_ITEMS}

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

def load_artefacts():
    """Load all saved model artefacts from the models/ directory."""
    paths = {
        'binary_model':    os.path.join(MODELS_DIR, 'binary_voting_ensemble_model.joblib'),
        'multi_model':     os.path.join(MODELS_DIR, 'multiclass_voting_ensemble_model.joblib'),
        'scaler':          os.path.join(MODELS_DIR, 'binary_scaler.joblib'),
        'label_encoders':  os.path.join(MODELS_DIR, 'label_encoders.joblib'),
        'feature_metadata':os.path.join(MODELS_DIR, 'feature_metadata.json'),
    }

    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        st.error("Missing model artefacts. Please run depression_detection_pipeline.py first.")
        for m in missing:
            st.error(f"  {m}")
        st.stop()

    with open(paths['feature_metadata']) as f:
        meta = json.load(f)

    return {
        'binary_model':   joblib.load(paths['binary_model']),
        'multi_model':    joblib.load(paths['multi_model']),
        'scaler':         joblib.load(paths['scaler']),
        'label_encoders': joblib.load(paths['label_encoders']),
        'feature_cols':   meta['feature_cols'],
        'feature_means':  meta['feature_means'],
    }

def load_top_items(n_bdi=TOP_N_BDI, n_fcri=TOP_N_FCRI):
    """
    Read feature_importance.csv and return top-ranked BDI item numbers and FCRI column names.
    Falls back to clinical defaults if file doesn't exist.
    """
    imp_path = os.path.join(MODELS_DIR, 'feature_importance.csv')
    CLINICAL_BDI_DEFAULT = [1, 2, 4, 5, 9, 15, 16, 17, 18, 20]
    CLINICAL_FCRI_DEFAULT = [item['col'] for item in FCRI_KEY_ITEMS]

    if not os.path.exists(imp_path):
        return CLINICAL_BDI_DEFAULT[:n_bdi], CLINICAL_FCRI_DEFAULT[:n_fcri]

    try:
        imp = pd.read_csv(imp_path)
        bdi_mask  = imp['feature'].str.match(r'^\d+\.$')
        fcri_mask = imp['feature'].str.match(r'^\d+\. ')

        top_bdi  = imp[bdi_mask].head(n_bdi)['feature'].tolist()
        top_fcri = imp[fcri_mask].head(n_fcri)['feature'].tolist()

        bdi_nums  = [int(f.rstrip('.')) for f in top_bdi]
        fcri_cols = [c for c in top_fcri if c in FCRI_ALL_ITEMS]

        for item in FCRI_KEY_ITEMS:
            if len(fcri_cols) >= n_fcri:
                break
            if item['col'] not in fcri_cols:
                fcri_cols.append(item['col'])

        return bdi_nums, fcri_cols[:n_fcri]

    except Exception as e:
        return CLINICAL_BDI_DEFAULT[:n_bdi], CLINICAL_FCRI_DEFAULT[:n_fcri]

# =============================================================================
# FEATURE VECTOR CONSTRUCTION
# =============================================================================

def build_feature_vector(bdi_responses, fcri_responses, demographics,
                          feature_cols, feature_means, label_encoders):
    """Construct the feature vector matching the training schema exactly."""
    feat = {col: feature_means.get(col, 0.0) for col in feature_cols}

    for col, score in bdi_responses.items():
        if col in feat:
            feat[col] = float(score)

    for col, score in fcri_responses.items():
        if col in feat:
            feat[col] = float(score)

    demo_map = {
        'gender':          'gender_encoded',
        'age_group':       'age_group_encoded',
        'education_level': 'education_level_encoded',
        'cancer_stage':    'cancer_stage_encoded',
        'cancer_status':   'cancer_status_encoded',
    }

    for demo_key, feat_key in demo_map.items():
        if feat_key not in feat:
            continue
        le = label_encoders.get(demo_key)
        if le is None:
            continue
        value = demographics.get(demo_key, 'Unknown')
        try:
            feat[feat_key] = float(le.transform([value])[0])
        except ValueError:
            feat[feat_key] = 0.0

    return np.array([feat[col] for col in feature_cols], dtype=np.float32)

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict(feature_vec, artefacts):
    """Run binary and multiclass predictions."""
    X_df     = pd.DataFrame([feature_vec], columns=artefacts['feature_cols'])
    X_scaled = artefacts['scaler'].transform(X_df)

    binary_pred  = int(artefacts['binary_model'].predict(X_scaled)[0])
    binary_proba = artefacts['binary_model'].predict_proba(X_scaled)[0]

    multi_pred  = int(artefacts['multi_model'].predict(X_scaled)[0])
    multi_proba = artefacts['multi_model'].predict_proba(X_scaled)[0]

    severity_label = (SEVERITY_LABELS[multi_pred]
                      if 0 <= multi_pred < len(SEVERITY_LABELS)
                      else 'Unknown')

    confidence = float(multi_proba[multi_pred]) * 100

    return {
        'binary_pred':    binary_pred,
        'depression_prob':float(binary_proba[1]) * 100,
        'severity_idx':   multi_pred,
        'severity_label': severity_label,
        'confidence':     confidence,
        'multi_proba':    multi_proba,
    }

def bdi_reference(bdi_responses, n_total=21):
    """Compute BDI total and severity label from collected items."""
    n_asked      = len(bdi_responses)
    raw_total    = sum(bdi_responses.values())
    extrapolated = round(raw_total * n_total / n_asked) if n_asked else 0

    severity_table = [
        ('Normal',                          0, 10),
        ('Mild mood disturbance',          11, 16),
        ('Borderline clinical depression', 17, 20),
        ('Moderate depression',            21, 30),
        ('Severe depression',              31, 40),
        ('Extreme depression',             41, 63),
    ]

    for label, lo, hi in severity_table:
        if lo <= extrapolated <= hi:
            return raw_total, n_asked, extrapolated, label
    return raw_total, n_asked, extrapolated, 'Unknown'

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_results_to_csv(demographics, bdi_responses, fcri_responses, pred, bdi_ref, timestamp=None):
    """Export assessment results to CSV format."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    data = {
        'timestamp': timestamp,
        'demographics': str(demographics),
        'bdi_responses': str(bdi_responses),
        'fcri_responses': str(fcri_responses),
        'severity_label': str(pred['severity_label']),
        'confidence': float(pred['confidence']),
        'depression_probability': float(pred['depression_prob']),
        'bdi_raw_score': int(bdi_ref[0]),
        'bdi_extrapolated_score': int(bdi_ref[2]),
        'bdi_severity': str(bdi_ref[3]),
    }

    df = pd.DataFrame([data])
    return df.to_csv(index=False).encode('utf-8')

def export_results_to_pdf(demographics, bdi_responses, fcri_responses, pred, bdi_ref, timestamp=None):
    """Export assessment results to PDF format."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("Depression Assessment Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    timestamp_para = Paragraph(f"Generated: {timestamp}", styles['Normal'])
    story.append(timestamp_para)
    story.append(Spacer(1, 12))

    # Demographics
    demo_title = Paragraph("Demographics", styles['Heading2'])
    story.append(demo_title)
    demo_data = [[str(k), str(v)] for k, v in demographics.items()]
    demo_table = Table(demo_data)
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 12))

    # Results
    results_title = Paragraph("Assessment Results", styles['Heading2'])
    story.append(results_title)

    results_data = [
        ['ML Model Severity', str(pred['severity_label'])],
        ['Model Confidence', f"{pred['confidence']:.1f}%"],
        ['Depression Probability', f"{pred['depression_prob']:.1f}%"],
        ['BDI Score', f"{bdi_ref[0]}/{bdi_ref[1] * 3}"],
        ['BDI Extrapolated', f"{bdi_ref[2]}/63"],
        ['BDI Severity', str(bdi_ref[3])],
    ]

    results_table = Table(results_data)
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 12))

    # Clinical description
    desc = SEVERITY_DESCRIPTIONS.get(pred['severity_label'], '')
    if desc:
        desc_title = Paragraph("Clinical Assessment", styles['Heading2'])
        story.append(desc_title)
        desc_para = Paragraph(desc, styles['Normal'])
        story.append(desc_para)
        story.append(Spacer(1, 12))

    # Crisis resources if needed
    if pred['severity_idx'] >= 3:
        crisis_title = Paragraph("Crisis Resources", styles['Heading2'])
        story.append(crisis_title)
        crisis_data = [[str(name), str(number)] for name, number in CRISIS_RESOURCES.items()]
        crisis_table = Table(crisis_data)
        crisis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(crisis_table)

    # Disclaimer
    story.append(Spacer(1, 24))
    disclaimer = Paragraph("DISCLAIMER: This tool supports clinical screening only. It is NOT a diagnostic instrument. Always consult a qualified healthcare professional for diagnosis and treatment.", styles['Normal'])
    story.append(disclaimer)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def export_history_to_csv(history):
    """Export session history to CSV format."""
    if not history:
        return None

    # Flatten the complex data structures for CSV export
    export_data = []
    for assessment in history:
        pred = assessment['prediction']
        bdi_ref = assessment['bdi_reference']

        row = {
            'timestamp': assessment['timestamp'],
            'mode': assessment['mode'],
            'gender': assessment['demographics'].get('gender', ''),
            'age_group': assessment['demographics'].get('age_group', ''),
            'education_level': assessment['demographics'].get('education_level', ''),
            'cancer_stage': assessment['demographics'].get('cancer_stage', ''),
            'cancer_status': assessment['demographics'].get('cancer_status', ''),
            'severity_label': str(pred['severity_label']),
            'confidence': float(pred['confidence']),
            'depression_probability': float(pred['depression_prob']),
            'bdi_raw_score': int(bdi_ref[0]),
            'bdi_asked': int(bdi_ref[1]),
            'bdi_extrapolated_score': int(bdi_ref[2]),
            'bdi_severity': str(bdi_ref[3]),
        }
        export_data.append(row)

    df = pd.DataFrame(export_data)
    return df.to_csv(index=False).encode('utf-8')

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'artefacts' not in st.session_state:
        with st.spinner("Loading model artefacts..."):
            st.session_state.artefacts = load_artefacts()
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_assessment' not in st.session_state:
        st.session_state.current_assessment = None
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'assessment_step' not in st.session_state:
        st.session_state.assessment_step = 'mode_selection'
    if 'demographics' not in st.session_state:
        st.session_state.demographics = {}
    if 'bdi_responses' not in st.session_state:
        st.session_state.bdi_responses = {}
    if 'fcri_responses' not in st.session_state:
        st.session_state.fcri_responses = {}

# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    st.set_page_config(
        page_title="Depression Screening for Cancer Patients",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    try:
        init_session_state()
    except Exception as e:
        st.error(f"Error initializing session state: {e}")
        st.error("Please ensure all model artefacts are available in the models/ directory.")
        st.stop()

    st.title("🏥 Depression Screening for Cancer Patients")
    st.markdown("""
    **Beck Depression Inventory + Fear of Cancer Recurrence Inventory**

    This web application provides an interactive screening tool for detecting depression
    and its severity in cancer patients using validated psychological assessments.

    ---
    """)

    # Sidebar navigation info
    with st.sidebar:
        st.markdown("### 📋 Navigation")
        st.markdown("""
        Use the pages in the sidebar to navigate:
        - **🏠 Home**: Overview and getting started
        - **📋 Assessment**: Complete the questionnaires
        - **📊 Results**: View your assessment results
        - **📜 History**: Review past assessments
        """)
        st.markdown("---")
        st.markdown("### ℹ️ App Info")
        st.markdown(f"**Feature Dimensions:** {len(st.session_state.artefacts['feature_cols'])}")
        st.markdown(f"**History Items:** {len(st.session_state.history)}")

    st.markdown("### Quick Start")
    st.info("Click on **📋 Assessment** in the sidebar to begin the screening process.")

    st.markdown("---")
    st.markdown("### About This Tool")
    st.markdown("""
    This screening tool uses a hybrid machine learning framework that combines:
    - **Beck Depression Inventory (BDI-II)**: 21-item assessment of depression symptoms
    - **Fear of Cancer Recurrence Inventory (FCRI)**: 42-item assessment of cancer-related fears
    - **Multi-model Ensemble**: Logistic Regression, Random Forest, and XGBoost
    - **Binary Classification**: Detects presence/absence of depression
    - **Multi-class Classification**: Identifies severity levels (Normal to Extreme)

    **Assessment Modes:**
    - **Quick Screening**: Top-ranked items only (~7 minutes, 15 questions)
    - **Full Assessment**: All BDI + key FCRI items (~15 minutes, 30 questions)
    """)

    st.markdown("---")
    st.warning("⚠️ **DISCLAIMER**: This tool is for screening purposes only and is NOT a clinical diagnosis. Please consult a qualified healthcare professional for proper evaluation and treatment recommendations.")

if __name__ == '__main__':
    main()
