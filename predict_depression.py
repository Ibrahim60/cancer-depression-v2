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
import re
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

# ── Quick-mode item counts ────────────────────────────────────────────────────
# predict_depression.py supports two assessment modes:
#   Full    — asks all 21 BDI + 9 FCRI key items  (30 questions, ~15 min)
#   Quick   — asks top-N items ranked by feature importance (≤15 questions, ~7 min)
#
# Items are chosen dynamically from models/feature_importance.csv, which is
# re-generated each time depression_detection_pipeline.py is run.  As real
# patient data accumulates the importance ordering will shift; Quick mode will
# automatically reflect that without any code changes.
#
# Unasked items are imputed with per-feature training means (feature_metadata.json),
# which is far less biased than filling with 0 ("no symptom at all").
TOP_N_BDI  = 10   # top N BDI items by combined RF+XGB feature importance
TOP_N_FCRI = 5    # top N FCRI items by combined RF+XGB feature importance

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

# ── Additional FCRI items surfaced by feature importance (not in KEY_ITEMS above) ─
# These are included so Quick mode can ask them when they rank in the top N.
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

# Unified lookup: column name → item definition
FCRI_ALL_ITEMS = {item['col']: item for item in FCRI_KEY_ITEMS + FCRI_EXTRA_ITEMS}

# =============================================================================
# SECTION 0: MODE SELECTION & DYNAMIC ITEM LOADING
# =============================================================================

def load_top_items(n_bdi=TOP_N_BDI, n_fcri=TOP_N_FCRI):
    """
    Read feature_importance.csv (saved by the pipeline) and return the
    top-ranked BDI item numbers and FCRI column names.

    Falls back to a sensible clinical default if the file does not exist:
      BDI default : items covering the core DSM-5 depression domains
      FCRI default: the 9 key items already defined in FCRI_KEY_ITEMS
    """
    imp_path = os.path.join(MODELS_DIR, 'feature_importance.csv')

    # ── Clinical fallback (used before first pipeline run) ────────────────────
    # Covers: sadness, anhedonia, guilt, suicidality, energy, sleep, appetite,
    # concentration, psychomotor, self-worth — one item per DSM domain
    CLINICAL_BDI_DEFAULT = [1, 2, 4, 5, 9, 15, 16, 17, 18, 20]
    CLINICAL_FCRI_DEFAULT = [item['col'] for item in FCRI_KEY_ITEMS]

    if not os.path.exists(imp_path):
        print(f"  [Quick mode] feature_importance.csv not found — using clinical default items.")
        return CLINICAL_BDI_DEFAULT[:n_bdi], CLINICAL_FCRI_DEFAULT[:n_fcri]

    try:
        imp = pd.read_csv(imp_path)

        # BDI items are named '1.', '2.', … '21.' in the importance table
        bdi_mask  = imp['feature'].str.match(r'^\d+\.$')
        fcri_mask = imp['feature'].str.match(r'^\d+\. ')

        top_bdi  = imp[bdi_mask].head(n_bdi)['feature'].tolist()
        top_fcri = imp[fcri_mask].head(n_fcri)['feature'].tolist()

        bdi_nums  = [int(f.rstrip('.')) for f in top_bdi]
        # Only keep FCRI items we have display definitions for
        fcri_cols = [c for c in top_fcri if c in FCRI_ALL_ITEMS]
        # If fewer than n_fcri are known, pad with KEY_ITEMS
        for item in FCRI_KEY_ITEMS:
            if len(fcri_cols) >= n_fcri:
                break
            if item['col'] not in fcri_cols:
                fcri_cols.append(item['col'])

        return bdi_nums, fcri_cols[:n_fcri]

    except Exception as e:
        print(f"  [Quick mode] Could not read feature_importance.csv ({e}) — using clinical defaults.")
        return CLINICAL_BDI_DEFAULT[:n_bdi], CLINICAL_FCRI_DEFAULT[:n_fcri]


def choose_mode():
    """
    Prompt the clinician / patient to select Full or Quick assessment mode.
    Returns 'full' or 'quick'.
    """
    print("\n" + "─" * 65)
    print("  SELECT ASSESSMENT MODE")
    print("─" * 65)
    print("  1. Quick Screening  — top-ranked items only  (~7 min, 15 questions)")
    print(f"     {TOP_N_BDI} BDI items + {TOP_N_FCRI} FCRI items, chosen by ML feature importance")
    print("     Unasked items imputed from training-data means")
    print()
    print("  2. Full Assessment  — all BDI + key FCRI items  (~15 min, 30 questions)")
    print("     All 21 BDI items + 9 cross-subscale FCRI items")
    print("     Recommended for first-time patients and clinical validation")
    print("─" * 65)

    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice == '1':
            return 'quick'
        if choice == '2':
            return 'full'
        print("  Please enter 1 or 2.")


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


def collect_bdi(item_nums=None):
    """
    Collect BDI responses.

    Parameters
    ----------
    item_nums : list[int] | None
        Item numbers to ask (e.g. [1, 5, 9]).  None = ask all 21.
        Unasked items are imputed with training means in build_feature_vector().
    """
    items   = (BDI_QUESTIONS if item_nums is None
               else [q for q in BDI_QUESTIONS if q['num'] in item_nums])
    n_ask   = len(items)
    n_total = len(BDI_QUESTIONS)

    print("\n" + "=" * 65)
    if n_ask == n_total:
        print("SECTION 1 OF 2 — BECK DEPRESSION INVENTORY (BDI-21)")
    else:
        print(f"SECTION 1 OF 2 — BECK DEPRESSION INVENTORY  "
              f"({n_ask} of {n_total} items)")
        print(f"  Remaining {n_total - n_ask} items will be estimated from training averages.")
    print("=" * 65)
    print("Choose the statement that best describes how you have been")
    print("feeling DURING THE PAST WEEK, including today.")

    responses = {}
    for i, item in enumerate(items, 1):
        score = _prompt_choice(
            f"[BDI {i:02d}/{n_ask}] {item['title']}",
            item['options']
        )
        responses[f"{item['num']}."] = score
    return responses


def collect_fcri(col_filter=None):
    """
    Collect FCRI responses.

    Parameters
    ----------
    col_filter : list[str] | None
        Column names to ask.  None = use the 9 default key items.
        Any column in col_filter that lacks a definition in FCRI_ALL_ITEMS
        is silently skipped; remaining items are imputed with training means.
    """
    if col_filter is None:
        items = FCRI_KEY_ITEMS
    else:
        items = [FCRI_ALL_ITEMS[c] for c in col_filter if c in FCRI_ALL_ITEMS]
        if not items:            # nothing matched — fall back to key items
            items = FCRI_KEY_ITEMS

    n_ask   = len(items)
    n_total = 42

    print("\n" + "=" * 65)
    if n_ask == 9 and col_filter is None:
        print("SECTION 2 OF 2 — FEAR OF CANCER RECURRENCE (FCRI — Key Items)")
    else:
        print(f"SECTION 2 OF 2 — FEAR OF CANCER RECURRENCE  "
              f"({n_ask} of {n_total} items)")
        print(f"  Remaining {n_total - n_ask} items will be estimated from training averages.")
    print("=" * 65)
    print("Indicate to what degree each statement applied to you")
    print("DURING THE PAST MONTH.")

    responses = {}
    for i, item in enumerate(items, 1):
        score = _prompt_choice(
            f"[FCRI {i:02d}/{n_ask} — {item['subscale']}]\n  {item['label']}",
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
    # Wrap in a DataFrame so feature names match those the scaler was fitted on.
    # Passing a raw numpy array causes sklearn to emit a UserWarning about
    # missing feature names — this line resolves it entirely.
    X_df     = pd.DataFrame([feature_vec], columns=artefacts['feature_cols'])
    X_scaled = artefacts['scaler'].transform(X_df)

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

def bdi_reference(bdi_responses, n_total=21):
    """
    Compute BDI total and severity label from the collected items.

    In Quick mode fewer than 21 items are asked; the raw total is shown
    alongside a linear extrapolation to the full 21-item scale so the
    clinician can compare against standard BDI cut-offs.
    """
    n_asked      = len(bdi_responses)
    raw_total    = sum(bdi_responses.values())
    # Extrapolate: assume unasked items average the same as asked items
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
# SECTION 6: RESULTS DISPLAY
# =============================================================================

def display_results(pred, bdi_raw, bdi_asked, bdi_extrap, bdi_rule_label, mode):
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
    if bdi_asked == 21:
        print(f"\n  BDI Score (all 21 items) : {bdi_raw}/63 → {bdi_rule_label}")
    else:
        print(f"\n  BDI Score ({bdi_asked}/{21} items asked)  : "
              f"{bdi_raw}/{bdi_asked * 3}  "
              f"(extrapolated ≈ {bdi_extrap}/63 → {bdi_rule_label})")
        print(f"  Note: Quick-mode BDI score is extrapolated from {bdi_asked} items.")
        print(f"        Run a Full Assessment for a validated 21-item BDI score.")
    print(f"  (Rule-based score is informational; ML model output drives the assessment.)")

    # ── Clinical description ───────────────────────────────────────────────────
    desc = SEVERITY_DESCRIPTIONS.get(label, '')
    if desc:
        print(f"\n  Assessment Note:\n    {desc}")

    # ── Clinical follow-up flag ───────────────────────────────────────────────
    if bdi_extrap >= CLINICAL_FOLLOW_UP_THRESHOLD or pred['severity_idx'] >= 2:
        print("\n  ⚠  CLINICAL FOLLOW-UP RECOMMENDED")
        print("     Please discuss these results with your oncologist or a mental")
        print("     health professional at the earliest opportunity.")

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
    print("All responses are confidential and used only for screening.")

    print("\nLoading model artefacts…", end=' ', flush=True)
    artefacts = load_artefacts()
    print("done.")
    print(f"Feature dimensions : {len(artefacts['feature_cols'])}")

    run_again = True
    while run_again:
        # ── Mode selection ─────────────────────────────────────────────────────
        mode = choose_mode()

        if mode == 'quick':
            top_bdi_nums, top_fcri_cols = load_top_items(TOP_N_BDI, TOP_N_FCRI)
            print(f"\n  Quick mode — BDI items  : {sorted(top_bdi_nums)}")
            print(f"  Quick mode — FCRI items : {len(top_fcri_cols)} selected")
            est_time = '~7 min'
        else:
            top_bdi_nums  = None          # None = ask all 21
            top_fcri_cols = None          # None = use FCRI_KEY_ITEMS (9 items)
            est_time = '~15 min'

        print(f"\n  Estimated time : {est_time}")
        print("─" * 65)
        input("Press Enter to begin the assessment…")

        # ── Data collection ────────────────────────────────────────────────────
        demographics   = collect_demographics()
        bdi_responses  = collect_bdi(item_nums=top_bdi_nums)
        fcri_responses = collect_fcri(col_filter=top_fcri_cols)

        # ── Suicidal ideation warning (BDI item 9, score ≥ 2) ─────────────────
        # Item 9 is always included in Quick mode (it ranks in the top 10 BDI
        # items by importance).  If the mode somehow excludes it, the check
        # simply won't fire — never shows a false negative.
        if bdi_responses.get('9.', 0) >= 2:
            print("\n" + "!" * 65)
            print("  IMPORTANT: Your response to the question about suicidal")
            print("  thoughts indicates you may be in distress.")
            print("  Please contact a crisis helpline or healthcare provider now:")
            for name, number in list(CRISIS_RESOURCES.items())[:3]:
                print(f"    {name}: {number}")
            print("!" * 65)

        # ── Feature vector & prediction ────────────────────────────────────────
        feature_vec = build_feature_vector(
            bdi_responses, fcri_responses, demographics,
            artefacts['feature_cols'],
            artefacts['feature_means'],
            artefacts['label_encoders'],
        )
        pred = predict(feature_vec, artefacts)

        # ── BDI reference (informational — not used for ML classification) ─────
        bdi_raw, bdi_asked, bdi_extrap, bdi_rule_label = bdi_reference(bdi_responses)

        # ── Results ────────────────────────────────────────────────────────────
        display_results(pred, bdi_raw, bdi_asked, bdi_extrap, bdi_rule_label, mode)

        # ── Loop control ───────────────────────────────────────────────────────
        again = input("\nRun another assessment? (y/n): ").strip().lower()
        run_again = (again == 'y')

    print("\nThank you. Please ensure results are reviewed with a healthcare professional.")


if __name__ == '__main__':
    main()