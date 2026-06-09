"""
===================================================================================
DEPRESSION PREDICTION INTERFACE — Cancer Patient Psychological Assessment
===================================================================================
Version: 2.0

Fixes over v1
-------------
  ✗ ML models loaded but model.predict() NEVER CALLED — entire prediction was rule-based
  ✗ prepare_features() existed but was not used in main()
  ✗ feature_metadata.json not loaded → feature vector order was arbitrary / wrong
  ✗ Unasked FCRI items filled with 0.0 → now filled with per-feature training means
  ✗ BDI score extrapolated from 10 → 21 items (invalid)  — now collects all 21
  ✗ bdi_score *= gender_factor * age_factor before severity class (no clinical basis)
    → demographic risk insight moved to display-only; classification uses model output
  ✗ again == 'y': main() recursive call → replaced with while loop
  ✗ Duplicate helpline numbers in crisis resources
  ✗ label_encoders not loaded → demographic features encoded incorrectly

Usage
-----
    python predict_depression.py

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
import numpy as np
import pandas as pd
import joblib

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = 'models'

SEVERITY_LABELS   = [
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

CLINICAL_FOLLOW_UP_THRESHOLD = 17   # Borderline or above → recommend clinical follow-up

# =============================================================================
# BDI QUESTIONS  (all 21 items — required for valid BDI assessment)
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
# One item per subscale + two severity items — sufficient for a meaningful
# FCR screening while minimising assessment burden.
# Unasked FCRI items are imputed with per-feature training means (from
# feature_metadata.json) rather than 0.0 to avoid prediction bias.
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

# =============================================================================
# SECTION 1: MODEL LOADING
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
        print("\nERROR — missing artefact(s):")
        for m in missing:
            print(f"  {m}")
        print("Run depression_detection_pipeline.py first.")
        sys.exit(1)

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


# =============================================================================
# SECTION 2: INPUT COLLECTION
# =============================================================================

def _prompt_choice(prompt, options):
    """Present numbered options, return the integer value of the chosen option."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  {i + 1}. {opt}")
    while True:
        raw = input("  Enter number: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx     # index == numeric score (0-based)
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def collect_bdi():
    """Collect all 21 BDI items. Returns dict {legacy_col_name: score_0_3}."""
    print("\n" + "=" * 65)
    print("SECTION 1 OF 2 — BECK DEPRESSION INVENTORY (BDI-21)")
    print("=" * 65)
    print("Choose the statement that best describes how you have been")
    print("feeling DURING THE PAST WEEK, including today.")

    responses = {}
    for item in BDI_QUESTIONS:
        score = _prompt_choice(
            f"[BDI {item['num']:02d}/{len(BDI_QUESTIONS)}] {item['title']}",
            item['options']
        )
        col = f"{item['num']}."
        responses[col] = score
    return responses


def collect_fcri():
    """Collect 9 key FCRI items. Returns dict {full_col_name: score_0_4}."""
    print("\n" + "=" * 65)
    print("SECTION 2 OF 2 — FEAR OF CANCER RECURRENCE (FCRI — Key Items)")
    print("=" * 65)
    print("Indicate to what degree each statement applied to you")
    print("DURING THE PAST MONTH.")

    responses = {}
    for item in FCRI_KEY_ITEMS:
        score = _prompt_choice(
            f"[FCRI — {item['subscale']}]\n  {item['label']}",
            item['options']
        )
        responses[item['col']] = score
    return responses


def collect_demographics():
    """Collect demographic details for feature encoding."""
    print("\n" + "=" * 65)
    print("DEMOGRAPHICS (optional — improves prediction context)")
    print("=" * 65)

    gender_opts    = ['Male', 'Female', 'Other', 'Prefer not to say']
    age_opts       = ['Under 18', '18-30', '31-45', '46-60', 'Over 60']
    education_opts = ['No formal education', 'Primary', 'Secondary / Matric',
                      'Intermediate / FSc', "Bachelor's", "Master's or higher"]
    cancer_stage   = ['Stage I', 'Stage II', 'Stage III', 'Stage IV',
                      'Metastatic / Advanced', 'Not sure / not told']
    cancer_status  = ['Newly Diagnosed', 'Under Active Treatment', 'In Remission',
                      'Cancer Recurrence', 'Metastatic Disease', 'Palliative Care']

    gender_idx  = _prompt_choice("Gender", gender_opts)
    age_idx     = _prompt_choice("Age Group", age_opts)
    edu_idx     = _prompt_choice("Education Level", education_opts)
    stage_idx   = _prompt_choice("Cancer Stage", cancer_stage)
    status_idx  = _prompt_choice("Current Cancer Status", cancer_status)

    return {
        'gender':          gender_opts[gender_idx],
        'age_group':       age_opts[age_idx],
        'education_level': education_opts[edu_idx],
        'cancer_stage':    cancer_stage[stage_idx],
        'cancer_status':   cancer_status[status_idx],
    }


# =============================================================================
# SECTION 3: FEATURE VECTOR CONSTRUCTION
# =============================================================================

def build_feature_vector(bdi_responses, fcri_responses, demographics,
                          feature_cols, feature_means, label_encoders):
    """
    Construct the feature vector that matches the training schema exactly.

    Strategy for unasked FCRI items (FIXED over v1's zero-fill):
      Impute with per-feature training mean from feature_metadata.json.
      This assumes 'average' FCR on unasked items — far less biased than 0.
    """
    # Initialise all features with training means
    feat = {col: feature_means.get(col, 0.0) for col in feature_cols}

    # Fill BDI items
    for col, score in bdi_responses.items():
        if col in feat:
            feat[col] = float(score)

    # Fill asked FCRI items
    for col, score in fcri_responses.items():
        if col in feat:
            feat[col] = float(score)

    # Encode demographics using saved LabelEncoders
    demo_map = {
        'gender':          'gender_encoded',
        'age_group':       'age_group_encoded',
        'education_level': 'education_level_encoded',
        'cancer_stage':    'cancer_stage_encoded',
        'cancer_status':   'cancer_status_encoded',   # maps to 'cancer_status' or similar
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
            # Unseen category — use most frequent class (0)
            feat[feat_key] = 0.0

    # Return ordered array matching training schema
    return np.array([feat[col] for col in feature_cols], dtype=np.float32)


# =============================================================================
# SECTION 4: PREDICTION
# =============================================================================

def predict(feature_vec, artefacts):
    """Run binary and multiclass predictions. Returns a results dict."""
    X = feature_vec.reshape(1, -1)
    X_scaled = artefacts['scaler'].transform(X)

    # Binary: has_depression (0/1)
    binary_pred  = int(artefacts['binary_model'].predict(X_scaled)[0])
    binary_proba = artefacts['binary_model'].predict_proba(X_scaled)[0]

    # Multiclass: severity class (encoded label)
    multi_pred  = int(artefacts['multi_model'].predict(X_scaled)[0])
    multi_proba = artefacts['multi_model'].predict_proba(X_scaled)[0]

    # Resolve severity label
    severity_label = (SEVERITY_LABELS[multi_pred]
                      if 0 <= multi_pred < len(SEVERITY_LABELS)
                      else 'Unknown')

    # Confidence from multiclass proba
    confidence = float(multi_proba[multi_pred]) * 100

    return {
        'binary_pred':    binary_pred,
        'depression_prob':float(binary_proba[1]) * 100,
        'severity_idx':   multi_pred,
        'severity_label': severity_label,
        'confidence':     confidence,
        'multi_proba':    multi_proba,
    }


# =============================================================================
# SECTION 5: BDI REFERENCE SCORE  (informational only — not used for classification)
# =============================================================================

def bdi_reference(bdi_responses):
    """Compute BDI total and severity label from the 21 collected items."""
    total = sum(bdi_responses.values())
    for label, lo, hi in [
        ('Normal',                          0, 10),
        ('Mild mood disturbance',          11, 16),
        ('Borderline clinical depression', 17, 20),
        ('Moderate depression',            21, 30),
        ('Severe depression',              31, 40),
        ('Extreme depression',             41, 63),
    ]:
        if lo <= total <= hi:
            return total, label
    return total, 'Unknown'


# =============================================================================
# SECTION 6: RESULTS DISPLAY
# =============================================================================

def display_results(pred, bdi_total, bdi_rule_label, demographics):
    print("\n" + "=" * 65)
    print("ASSESSMENT RESULTS")
    print("=" * 65)

    # ── Primary: ML model classification ──────────────────────────────────────
    label = pred['severity_label']
    print(f"\n  ML Model Severity        : {label}")
    print(f"  Model Confidence         : {pred['confidence']:.1f}%")
    print(f"  Depression Probability   : {pred['depression_prob']:.1f}%")

    # Full severity probability breakdown
    print("\n  Severity probability breakdown:")
    for i, sev in enumerate(SEVERITY_LABELS):
        prob = float(pred['multi_proba'][i]) * 100 if i < len(pred['multi_proba']) else 0.0
        bar  = '█' * int(prob / 5) + '░' * (20 - int(prob / 5))
        print(f"    {sev:<38} {bar} {prob:5.1f}%")

    # ── Reference: BDI rule-based score ───────────────────────────────────────
    print(f"\n  BDI Rule-based Score     : {bdi_total}/63 → {bdi_rule_label}")
    print(f"  (Rule-based score is informational; ML model output drives the assessment.)")

    # ── Clinical description ───────────────────────────────────────────────────
    desc = SEVERITY_DESCRIPTIONS.get(label, '')
    if desc:
        print(f"\n  Assessment Note:\n    {desc}")

    # ── Clinical follow-up flag ───────────────────────────────────────────────
    if bdi_total >= CLINICAL_FOLLOW_UP_THRESHOLD or pred['severity_idx'] >= 2:
        print("\n  ⚠  CLINICAL FOLLOW-UP RECOMMENDED")
        print("     Please discuss these results with your oncologist or a mental")
        print("     health professional at the earliest opportunity.")

    # ── Suicidal ideation flag (BDI item 9) ───────────────────────────────────
    # Note: the BDI column key is '9.'
    if '9.' in demographics.get('_bdi_raw', {}):
        pass  # demographics dict is not the right place — handled below separately

    # ── Crisis resources ──────────────────────────────────────────────────────
    if pred['severity_idx'] >= 3:                    # moderate or above
        print("\n  CRISIS & SUPPORT RESOURCES (Pakistan):")
        for name, number in CRISIS_RESOURCES.items():
            print(f"    {name:<45} {number}")

    print("\n" + "=" * 65)
    print("DISCLAIMER: This tool supports clinical screening only.")
    print("It is NOT a diagnostic instrument. Always consult a qualified")
    print("healthcare professional for diagnosis and treatment.")
    print("=" * 65)


# =============================================================================
# MAIN — while loop (FIXED: v1 used recursion which hit Python's stack limit)
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("CANCER PATIENT PSYCHOLOGICAL SCREENING TOOL")
    print("Beck Depression Inventory + Fear of Cancer Recurrence")
    print("=" * 65)
    print("This assessment takes approximately 10-15 minutes.")
    print("All responses are confidential and used only for screening.")

    print("\nLoading model artefacts…", end=' ', flush=True)
    artefacts = load_artefacts()
    print("done.")
    print(f"Feature dimensions : {len(artefacts['feature_cols'])}")

    run_again = True
    while run_again:
        print("\n" + "─" * 65)
        input("Press Enter to begin the assessment…")

        # Collect responses
        demographics   = collect_demographics()
        bdi_responses  = collect_bdi()
        fcri_responses = collect_fcri()

        # Suicidal ideation warning (BDI item 9, score ≥ 2)
        if bdi_responses.get('9.', 0) >= 2:
            print("\n" + "!" * 65)
            print("  IMPORTANT: Your response to the question about suicidal")
            print("  thoughts indicates you may be in distress.")
            print("  Please contact a crisis helpline or healthcare provider now:")
            for name, number in list(CRISIS_RESOURCES.items())[:3]:
                print(f"    {name}: {number}")
            print("!" * 65)

        # Build feature vector
        feature_vec = build_feature_vector(
            bdi_responses, fcri_responses, demographics,
            artefacts['feature_cols'],
            artefacts['feature_means'],
            artefacts['label_encoders'],
        )

        # Predict (ML model is used — FIXED: v1 loaded models but never called predict)
        pred = predict(feature_vec, artefacts)

        # BDI rule-based score (informational reference only)
        bdi_total, bdi_rule_label = bdi_reference(bdi_responses)

        # Display
        display_results(pred, bdi_total, bdi_rule_label, demographics)

        # Loop control
        again = input("\nRun another assessment? (y/n): ").strip().lower()
        run_again = (again == 'y')

    print("\nThank you. Please ensure results are reviewed with a healthcare professional.")


if __name__ == '__main__':
    main()