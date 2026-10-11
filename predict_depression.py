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

# Force UTF-8 stdout/stderr. This CLI prints box-drawing separators (─),
# arrows (→/—) and a block-character progress bar (█/░); Windows' default
# console codepage (cp1252) can't encode those and crashes with
# UnicodeEncodeError as soon as output isn't a UTF-8-capable terminal (e.g.
# piped to a file or run non-interactively). reconfigure() is Python 3.7+;
# guarded in case a stream doesn't support it.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

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
BDI_MILD_CUTOFF              = 11   # Mild mood disturbance or above
CRISIS_BDI_THRESHOLD         = 21   # Moderate or above -> show crisis resources
MODERATE_SEVERITY_IDX        = 3    # index of 'Moderate depression' in SEVERITY_LABELS
BORDERLINE_SEVERITY_IDX      = 2    # index of 'Borderline clinical depression'

# BDI item 9 (Suicidal Thoughts) is a safety item: it is always asked, in both
# modes, regardless of its feature-importance rank, so the suicidal-ideation
# check can never be skipped. A score of 2+ triggers the crisis warning.
SAFETY_BDI_ITEM    = 9
SUICIDE_RISK_SCORE = 2

# FCRI Severity subscale (items 9-17, range 0-36) cutoffs from Simard & Savard
# (2009, 2015): >=13 flags clinically significant fear of cancer recurrence,
# >=16 indicates clinical-level FCR warranting specialist referral.
FCRI_SEVERITY_SCREEN_CUTOFF   = 13
FCRI_SEVERITY_CLINICAL_CUTOFF = 16
FCRI_SUBSCALE_ORDER = [
    'Triggers', 'Severity', 'Psychological Distress', 'Functioning Impairments',
    'Insight', 'Reassurance', 'Coping Strategies',
]

# ── Quick-mode item counts ────────────────────────────────────────────────────
# predict_depression.py supports two assessment modes:
#   Full    — asks all 21 BDI + all 42 FCRI items  (63 questions, ~25-30 min)
#   Quick   — asks top-N items ranked by feature importance, plus BDI item 9
#             as a safety item (≤16 questions, ~7 min)
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

# Response scales. Verified against the real survey data (Data/cancer_psychology.csv)
# rather than assumed — Triggers (1-8) turned out to use the frequency scale, not
# the intensity scale used by Severity/Distress/Functioning/Insight.
_FREQ      = ['0 - Never', '1 - Rarely', '2 - Sometimes', '3 - Most of the time', '4 - All the time']
_INTENSITY = ['0 - Not at all', '1 - A little', '2 - Somewhat', '3 - A lot', '4 - A great deal']

# All 42 FCRI items, in questionnaire order, grouped by subscale
# (FCRI_SUBSCALES item ranges in depression_detection_pipeline.py):
#   Triggers 1-8 (freq) · Severity 9-17 (mixed, item 13 reverse-scored) ·
#   Psychological Distress 18-21 (intensity) · Functioning Impairments 22-27 (intensity) ·
#   Insight 28-30 (intensity) · Reassurance 31-33 (freq) · Coping Strategies 34-42 (freq)
FCRI_QUESTIONS = [
    {'num': 1, 'col': '1. Television shows or newspaper articles about cancer or illness',
     'label': 'Triggered by: TV/news about cancer or illness', 'subscale': 'Triggers', 'options': _FREQ},
    {'num': 2, 'col': '2. An appointment with my doctor or other health professional',
     'label': 'Triggered by: a doctor appointment', 'subscale': 'Triggers', 'options': _FREQ},
    {'num': 3, 'col': '3. Medical examinations (e.g. annual check-up, blood tests, X-rays)',
     'label': 'Triggered by: medical examinations', 'subscale': 'Triggers', 'options': _FREQ},
    {'num': 4, 'col': '4. Conversations about cancer or illness in general',
     'label': 'Triggered by: conversations about cancer/illness', 'subscale': 'Triggers', 'options': _FREQ},
    {'num': 5, 'col': '5. Seeing or hearing about someone who is ill',
     'label': 'Triggered by: seeing/hearing about someone ill', 'subscale': 'Triggers', 'options': _FREQ},
    {'num': 6, 'col': '6. Going to a funeral or reading the obituary section of the paper',
     'label': 'Triggered by: funerals or obituaries', 'subscale': 'Triggers', 'options': _FREQ},
    {'num': 7, 'col': '7. When I feel unwell physically or when I am sick',
     'label': 'Triggered by: feeling physically unwell', 'subscale': 'Triggers', 'options': _FREQ},
    {'num': 8, 'col': '8. Generally, I avoid situations or things that make me think about the possibility of cancer\nrecurrence',
     'label': 'I avoid situations that make me think about recurrence', 'subscale': 'Triggers', 'options': _FREQ},
    {'num': 9, 'col': '9. I am worried or anxious about the possibility of cancer recurrence',
     'label': 'FCRI: Worry about recurrence', 'subscale': 'Severity', 'options': _INTENSITY},
    {'num': 10, 'col': '10. I am afraid of cancer recurrence',
     'label': 'FCRI: Fear of recurrence', 'subscale': 'Severity', 'options': _INTENSITY},
    {'num': 11, 'col': '11. I believe it is normal to be worried or anxious about the possibility of cancer recurrence',
     'label': 'I believe it is normal to worry about recurrence', 'subscale': 'Severity', 'options': _INTENSITY},
    {'num': 12, 'col': '12. When I think about the possibility of cancer recurrence, this triggers other unpleasant\nthoughts or images (such as death, suffering, the consequences for my family)',
     'label': 'Recurrence thoughts trigger other unpleasant thoughts/images', 'subscale': 'Severity', 'options': _INTENSITY},
    {'num': 13, 'col': '13. I believe that I am cured and that the cancer will not come back',
     'label': 'FCRI: Belief in cure (reverse-scored)', 'subscale': 'Severity', 'options': _INTENSITY},
    {'num': 14, 'col': '14. In your opinion, are you at risk of having a cancer recurrence?',
     'label': 'FCRI: Perceived risk of recurrence', 'subscale': 'Severity',
     'options': ['0 - Not at all at risk', '1 - A little at risk', '2 - Somewhat at risk',
                 '3 - A lot at risk', '4 - A great deal at risk']},
    {'num': 15, 'col': '15. How often do you think about the possibility of cancer recurrence?',
     'label': 'FCRI: Frequency of recurrence thoughts', 'subscale': 'Severity',
     'options': ['0 - Never', '1 - A few times a month', '2 - A few times a week',
                 '3 - A few times a day', '4 - Several times a day']},
    {'num': 16, 'col': '16. How much time per day do you spend thinking about the possibility of cancer recurrence?',
     'label': 'FCRI: Time spent per day thinking about recurrence', 'subscale': 'Severity',
     'options': ["0 - I don't think about it", '1 - A few seconds',
                 '2 - A few minutes', '3 - A few hours', '4 - Several hours']},
    {'num': 17, 'col': '17. How long have you been thinking about the possibility of cancer recurrence?',
     'label': 'FCRI: How long recurrence thoughts have persisted', 'subscale': 'Severity',
     'options': ["0 - I don't think about it", '1 - A few weeks',
                 '2 - A few months', '3 - A few years', '4 - Several years']},
    {'num': 18, 'col': '18. Worry, fear or anxiety',
     'label': 'FCRI: Emotional distress — worry or fear', 'subscale': 'Psychological Distress', 'options': _INTENSITY},
    {'num': 19, 'col': '19. Sadness, discouragement or disappointment',
     'label': 'FCRI: Emotional distress — sadness/disappointment', 'subscale': 'Psychological Distress', 'options': _INTENSITY},
    {'num': 20, 'col': '20. Frustration, anger or outrage',
     'label': 'FCRI: Emotional distress — frustration/anger', 'subscale': 'Psychological Distress', 'options': _INTENSITY},
    {'num': 21, 'col': '21. Helplessness or resignation',
     'label': 'FCRI: Emotional distress — helplessness/resignation', 'subscale': 'Psychological Distress', 'options': _INTENSITY},
    {'num': 22, 'col': '22. My social or leisure activities (e.g. outings, sports, travel)',
     'label': 'FCRI: Impact on social activities', 'subscale': 'Functioning Impairments', 'options': _INTENSITY},
    {'num': 23, 'col': '23. My work or everyday activities',
     'label': 'FCRI: Impact on work or everyday activities', 'subscale': 'Functioning Impairments', 'options': _INTENSITY},
    {'num': 24, 'col': '24. My relationships with my partner, my family, or those close to me',
     'label': 'FCRI: Impact on relationships', 'subscale': 'Functioning Impairments', 'options': _INTENSITY},
    {'num': 25, 'col': '25. My ability to make future plans or set life goals',
     'label': 'FCRI: Impact on future plans/life goals', 'subscale': 'Functioning Impairments', 'options': _INTENSITY},
    {'num': 26, 'col': '26. My state of mind or my mood',
     'label': 'FCRI: Impact on state of mind', 'subscale': 'Functioning Impairments', 'options': _INTENSITY},
    {'num': 27, 'col': '27. My quality of life in general',
     'label': 'FCRI: Impact on quality of life', 'subscale': 'Functioning Impairments', 'options': _INTENSITY},
    {'num': 28, 'col': '28. I feel that I worry excessively about the possibility of cancer recurrence',
     'label': 'FCRI: Awareness of excessive worry', 'subscale': 'Insight', 'options': _INTENSITY},
    {'num': 29, 'col': '29. Other people think that I worry excessively about the possibility of cancer recurrence',
     'label': 'Others think I worry excessively about recurrence', 'subscale': 'Insight', 'options': _INTENSITY},
    {'num': 30, 'col': '30. I think that I worry more about the possibility of cancer recurrence than other people who\nhave been diagnosed with cancer',
     'label': 'I worry more about recurrence than other cancer patients', 'subscale': 'Insight', 'options': _INTENSITY},
    {'num': 31, 'col': '31. I call my doctor or other health professional',
     'label': 'FCRI: Seeking reassurance from doctor', 'subscale': 'Reassurance', 'options': _FREQ},
    {'num': 32, 'col': '32. I go to the hospital or clinic for an examination',
     'label': 'I go to the hospital/clinic for an examination', 'subscale': 'Reassurance', 'options': _FREQ},
    {'num': 33, 'col': '33. I examine myself to see if I have any physical signs of cancer',
     'label': 'I examine myself for physical signs of cancer', 'subscale': 'Reassurance', 'options': _FREQ},
    {'num': 34, 'col': '34. I try to distract myself (e.g. do various activities, watch television, read, work)',
     'label': 'Coping: I try to distract myself', 'subscale': 'Coping Strategies', 'options': _FREQ},
    {'num': 35, 'col': '35. I try not to think about it, to get the idea out of my mind',
     'label': 'Coping: I try not to think about it', 'subscale': 'Coping Strategies', 'options': _FREQ},
    {'num': 36, 'col': '36. I pray, meditate or do relaxation',
     'label': 'FCRI: Use of relaxation or prayer', 'subscale': 'Coping Strategies', 'options': _FREQ},
    {'num': 37, 'col': '37. I try to convince myself that everything will be fine or I think positively',
     'label': 'Coping: I try to think positively', 'subscale': 'Coping Strategies', 'options': _FREQ},
    {'num': 38, 'col': '38. I talk to someone about it',
     'label': 'Coping: I talk to someone about it', 'subscale': 'Coping Strategies', 'options': _FREQ},
    {'num': 39, 'col': '39. I try to understand what is happening and deal with it',
     'label': 'Coping: I try to understand and deal with it', 'subscale': 'Coping Strategies', 'options': _FREQ},
    {'num': 40, 'col': '40. I try to find a solution',
     'label': 'Coping: I try to find a solution', 'subscale': 'Coping Strategies', 'options': _FREQ},
    {'num': 41, 'col': '41. I try to replace this thought with a more pleasant one',
     'label': 'Coping: I try to replace the thought with a pleasant one', 'subscale': 'Coping Strategies', 'options': _FREQ},
    {'num': 42, 'col': '42. I tell myself “stop it”',
     'label': 'Coping: I tell myself "stop it"', 'subscale': 'Coping Strategies', 'options': _FREQ},
]

# Full lookup: column name -> item definition (covers all 42, so any item the
# ML feature-importance ranking picks for Quick mode is guaranteed to resolve —
# previously only 14 of 42 items had display definitions here).
FCRI_ALL_ITEMS = {q['col']: q for q in FCRI_QUESTIONS}

# 9-item clinical default used by Quick mode when feature_importance.csv is
# absent (first run, before any training) — one item per subscale.
_fcri_by_num  = {q['num']: q for q in FCRI_QUESTIONS}
FCRI_KEY_ITEMS = [_fcri_by_num[n] for n in [9, 14, 15, 18, 22, 26, 28, 31, 36]]

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
        return with_safety_item(CLINICAL_BDI_DEFAULT[:n_bdi]), CLINICAL_FCRI_DEFAULT[:n_fcri]

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

        return with_safety_item(bdi_nums), fcri_cols[:n_fcri]

    except Exception as e:
        print(f"  [Quick mode] Could not read feature_importance.csv ({e}) — using clinical defaults.")
        return with_safety_item(CLINICAL_BDI_DEFAULT[:n_bdi]), CLINICAL_FCRI_DEFAULT[:n_fcri]


def with_safety_item(bdi_nums):
    """Add BDI item 9 to a Quick-mode item list if the ranking left it out."""
    return bdi_nums if SAFETY_BDI_ITEM in bdi_nums else bdi_nums + [SAFETY_BDI_ITEM]


def choose_mode():
    """
    Prompt the clinician / patient to select Full or Quick assessment mode.
    Returns 'full' or 'quick'.
    """
    print("\n" + "─" * 65)
    print("  SELECT ASSESSMENT MODE")
    print("─" * 65)
    print("  1. Quick Screening  — top-ranked items only  (~7 min, up to 16 questions)")
    print(f"     {TOP_N_BDI} BDI items + {TOP_N_FCRI} FCRI items, chosen by ML feature importance,")
    print(f"     plus BDI item {SAFETY_BDI_ITEM} (suicidal thoughts) as a safety question")
    print("     Unasked items imputed from training-data means")
    print()
    print("  2. Full Assessment  — all BDI + all FCRI items  (~25-30 min, 63 questions)")
    print("     All 21 BDI items + all 42 FCRI items — no answers are estimated")
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
        'binary_model':    joblib.load(paths['binary_model']),
        'multi_model':     joblib.load(paths['multi_model']),
        'scaler':          joblib.load(paths['scaler']),
        'label_encoders':  joblib.load(paths['label_encoders']),
        'feature_cols':    meta['feature_cols'],
        'feature_means':   meta['feature_means'],
        # CV-tuned decision threshold on P(has_depression) that maximizes
        # macro-F1 (see tune_binary_threshold() in the training pipeline).
        # Falls back to the sklearn default of 0.5 for older metadata files.
        'binary_threshold':meta.get('binary_threshold', 0.5),
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
    if n_ask == n_total:
        print(f"SECTION 2 OF 2 — FEAR OF CANCER RECURRENCE (FCRI — all {n_total} items)")
    elif col_filter is None:
        print("SECTION 2 OF 2 — FEAR OF CANCER RECURRENCE (FCRI — Key Items)")
        print(f"  Remaining {n_total - n_ask} items will be estimated from training averages.")
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
    # Option text must match the training data's categories exactly
    # (label_encoders.joblib), or the answer can't be encoded.
    age_opts       = ['Below 18', '18-30', '31-45', '46-60', 'Above 60']
    education_opts = ['No formal education', 'Primary', 'Secondary / Matric',
                      'Intermediate / FSc', "Bachelor's", "Master's or higher"]
    cancer_stage   = ['Stage I', 'Stage II', 'Stage III', 'Stage IV',
                      'Metastatic / Advanced', 'Not sure / not told']
    cancer_status  = ['Newly Diagnosed', 'Under Treatment', 'Recurrence',
                      'Recovered / Survivor']

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

    # Fill asked FCRI items. Item 13 ("I believe that I am cured…") is
    # reverse-scored (4 − raw) during training, so the raw answer must be
    # reversed here too or the model sees the opposite of what was answered.
    for col, score in fcri_responses.items():
        if col in feat:
            feat[col] = 4.0 - float(score) if col.startswith('13. ') else float(score)

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
        # A category the model never saw keeps its training mean (set above),
        # like an unasked item; code 0 would silently mean a real category.
        if value in le.classes_:
            feat[feat_key] = float(le.transform([value])[0])

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

    # Binary: has_depression (0/1), decided against the tuned threshold
    # rather than the model's default 0.5 cutoff.
    binary_proba = artefacts['binary_model'].predict_proba(X_scaled)[0]
    binary_pred  = int(binary_proba[1] >= artefacts.get('binary_threshold', 0.5))

    # The pipeline encodes severity with LabelEncoder, which sorts labels
    # alphabetically, so the model's class ids are NOT in severity order.
    # Map each class id back to its label, then reorder the probabilities
    # into SEVERITY_LABELS order (Normal -> Extreme).
    encoded_labels = sorted(SEVERITY_LABELS)
    model = artefacts['multi_model']
    raw_proba = model.predict_proba(X_scaled)[0]
    multi_proba = np.zeros(len(SEVERITY_LABELS))
    for cls, p in zip(model.classes_, raw_proba):
        multi_proba[SEVERITY_LABELS.index(encoded_labels[int(cls)])] = p

    severity_idx   = int(np.argmax(multi_proba))
    severity_label = SEVERITY_LABELS[severity_idx]
    confidence     = float(multi_proba[severity_idx]) * 100

    return {
        'binary_pred':    binary_pred,
        'depression_prob':float(binary_proba[1]) * 100,
        'severity_idx':   severity_idx,
        'severity_label': severity_label,
        'confidence':     confidence,
        'multi_proba':    multi_proba,
    }


# =============================================================================
# SECTION 5: BDI REFERENCE SCORE
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
# SECTION 6: FCRI REFERENCE SCORE & FINAL MODEL RESULT
# =============================================================================

def fcri_reference(fcri_responses):
    """
    Score collected FCRI items: per-subscale raw totals, the overall total, and
    the Severity subscale (extrapolated to 9 items when only some were asked)
    with its clinical interpretation. Item 13 is reverse-scored (4 - raw).
    """
    subscales = {}
    for name in FCRI_SUBSCALE_ORDER:
        items = [q for q in FCRI_QUESTIONS if q['subscale'] == name]
        answered = [q for q in items if q['col'] in fcri_responses]
        raw = 0
        for q in answered:
            score = float(fcri_responses[q['col']])
            raw += 4 - score if q['num'] == 13 else score
        subscales[name] = {
            'raw':     int(round(raw)),
            'n_asked': len(answered),
            'n_total': len(items),
        }

    total_raw   = sum(s['raw'] for s in subscales.values())
    total_asked = sum(s['n_asked'] for s in subscales.values())

    sev = subscales['Severity']
    if sev['n_asked']:
        sev_score = round(sev['raw'] * sev['n_total'] / sev['n_asked'])
        if sev_score >= FCRI_SEVERITY_CLINICAL_CUTOFF:
            sev_label = 'Clinical fear of recurrence'
        elif sev_score >= FCRI_SEVERITY_SCREEN_CUTOFF:
            sev_label = 'Clinically significant fear of recurrence'
        else:
            sev_label = 'Non-clinical fear of recurrence'
    else:
        sev_score, sev_label = None, 'Not assessed'

    return {
        'total_raw':      total_raw,
        'total_asked':    total_asked,
        'severity_score': sev_score,        # 0-36, extrapolated if partial
        'severity_label': sev_label,
        'severity_extrapolated': 0 < sev['n_asked'] < sev['n_total'],
        'subscales':      subscales,
    }


def final_verdict(pred, bdi_ref, fcri_ref):
    """
    Final depression-detection result. The ML ensemble (trained on all BDI +
    FCRI answers) decides whether depression is present; the FCRI Severity
    subscale decides whether it is linked to fear of cancer recurrence:

      model severity >= Borderline,  FCRI >= 13  -> YES       cancer-related depression
      model severity >= Borderline,  FCRI <  13  -> YES       depression, not driven by fear of recurrence
      model mild / binary positive,  FCRI >= 13  -> POSSIBLE  at risk
      otherwise                                  -> NO

    The rule-based BDI total is compared against the model as a sanity check.
    """
    model_clinical = pred['severity_idx'] >= 2
    model_mild = not model_clinical and (pred['severity_idx'] == 1 or bool(pred['binary_pred']))
    fcri_score = fcri_ref['severity_score']
    fcri_positive = fcri_score is not None and fcri_score >= FCRI_SEVERITY_SCREEN_CUTOFF
    bdi_clinical = bdi_ref[2] >= CLINICAL_FOLLOW_UP_THRESHOLD

    if model_clinical and fcri_positive:
        decision, tier = 'YES', 'danger'
        headline = 'Cancer-related depression detected'
        summary = ('The model detects clinically relevant depression, and the FCRI shows significant '
                   'fear of cancer recurrence, consistent with depression linked to the cancer experience.')
    elif model_clinical:
        decision, tier = 'YES', 'elevated'
        headline = 'Depression detected'
        summary = ('The model detects clinically relevant depression. Fear of cancer recurrence (FCRI) '
                   'is not elevated, so the depression does not appear to be driven mainly by cancer-related fear.')
    elif model_mild and fcri_positive:
        decision, tier = 'POSSIBLE', 'warning'
        headline = 'At risk of cancer-related depression'
        summary = ('The model finds only mild, sub-clinical mood symptoms, but the FCRI shows significant '
                   'fear of cancer recurrence. This should be monitored.')
    else:
        decision, tier = 'NO', 'success'
        headline = 'No cancer-related depression detected'
        summary = 'The model does not detect clinically relevant depression.'
        if fcri_positive:
            summary += (' Fear of cancer recurrence (FCRI) is elevated on its own and may still benefit '
                        'from psycho-oncology support.')

    return {
        'decision':      decision,        # 'YES' | 'POSSIBLE' | 'NO'
        'has_cancer_depression': decision == 'YES',
        'tier':          tier,
        'headline':      headline,
        'summary':       summary,
        'model_clinical':model_clinical,
        'model_mild':    model_mild,
        'fcri_positive': fcri_positive,
        'bdi_clinical':  bdi_clinical,
        'bdi_agrees':    bdi_clinical == model_clinical,
    }


# Safety and follow-up rules, shared by the CLI, the web results page and the
# PDF report so all three always show the same messages. Each errs towards
# showing the message: either the model or the BDI total can trigger it.

def suicide_risk(bdi_responses):
    """True if BDI item 9 (Suicidal Thoughts) was answered 2 or higher."""
    return bdi_responses.get(f'{SAFETY_BDI_ITEM}.', 0) >= SUICIDE_RISK_SCORE


def needs_follow_up(pred, bdi_ref, verdict):
    """Recommend follow-up: verdict YES/POSSIBLE, or Borderline+ by model or BDI."""
    return (verdict['decision'] != 'NO'
            or pred['severity_idx'] >= BORDERLINE_SEVERITY_IDX
            or bdi_ref[2] >= CLINICAL_FOLLOW_UP_THRESHOLD)


def needs_crisis_resources(pred, bdi_ref, bdi_responses):
    """Show crisis helplines: Moderate+ by model or BDI, or suicidal-ideation risk."""
    return (pred['severity_idx'] >= MODERATE_SEVERITY_IDX
            or bdi_ref[2] >= CRISIS_BDI_THRESHOLD
            or suicide_risk(bdi_responses))


# =============================================================================
# SECTION 7: RESULTS DISPLAY  (Step 1 BDI → Step 2 FCRI → Step 3 model result)
# =============================================================================

def _answer_text(options, score):
    """Option text for a 0-based score, without its leading 'N - ' prefix."""
    opt = options[int(score)]
    return opt.split(' - ', 1)[1] if ' - ' in opt else opt


def display_results(pred, bdi_responses, fcri_responses, bdi_ref, mode):
    bdi_raw, bdi_asked, bdi_extrap, bdi_rule_label = bdi_ref
    fcri_ref = fcri_reference(fcri_responses)
    verdict  = final_verdict(pred, bdi_ref, fcri_ref)

    print("\n" + "=" * 65)
    print(f"ASSESSMENT RESULTS  ({mode.capitalize()} mode)")
    print("=" * 65)

    # ── Step 1: BDI ───────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  STEP 1 — BDI RESULTS (Depressive Symptoms)")
    print("─" * 65)
    for q in BDI_QUESTIONS:
        key = f"{q['num']}."
        if key in bdi_responses:
            score = int(bdi_responses[key])
            print(f"    {q['num']:>2}. {q['title']:<24} {score}/3  {_answer_text(q['options'], score)}")
    print()
    if bdi_asked == 21:
        print(f"  BDI Total Score          : {bdi_raw}/63")
    else:
        print(f"  BDI Raw Score            : {bdi_raw}/{bdi_asked * 3}  ({bdi_asked} of 21 items asked)")
        print(f"  BDI Extrapolated Score   : {bdi_extrap}/63")
        print("  Note: Quick-mode BDI score is extrapolated. Run a Full Assessment for a validated score.")
    print(f"  BDI Severity             : {bdi_rule_label}")

    # ── Step 2: FCRI ──────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  STEP 2 — FCRI RESULTS (Fear of Cancer Recurrence)")
    print("─" * 65)
    if fcri_ref['total_asked']:
        for q in FCRI_QUESTIONS:
            if q['col'] in fcri_responses:
                score = int(fcri_responses[q['col']])
                rev = ' (reversed)' if q['num'] == 13 else ''
                print(f"    {q['num']:>2}. [{q['subscale']}] {q['label']}")
                print(f"        {score}/4{rev}  {_answer_text(q['options'], score)}")
        print("\n  Subscale scores:")
        for name, sub in fcri_ref['subscales'].items():
            if sub['n_asked']:
                print(f"    {name:<26} {sub['raw']:>3}/{sub['n_asked'] * 4:<3}  "
                      f"({sub['n_asked']} of {sub['n_total']} items)")
        sev_score = fcri_ref['severity_score']
        sev_text = 'N/A' if sev_score is None else f"{sev_score}/36"
        if fcri_ref['severity_extrapolated']:
            sev_text += '  (extrapolated)'
        print(f"\n  FCRI Total Score         : {fcri_ref['total_raw']}/{fcri_ref['total_asked'] * 4}")
        print(f"  FCRI Severity Score      : {sev_text}")
        print(f"  FCRI Severity            : {fcri_ref['severity_label']}")
        print(f"  (Severity >= {FCRI_SEVERITY_SCREEN_CUTOFF}: clinically significant fear; "
              f">= {FCRI_SEVERITY_CLINICAL_CUTOFF}: clinical-level fear)")
    else:
        print("  No FCRI items were answered.")

    # ── Step 3: model result ─────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  STEP 3 — MODEL RESULT: DEPRESSION DETECTION (BDI + FCRI)")
    print("─" * 65)
    print(f"  Depression Probability   : {pred['depression_prob']:.1f}%  "
          f"({'detected' if pred['binary_pred'] else 'not detected'})")
    print(f"  Predicted Severity       : {pred['severity_label']}  "
          f"(confidence {pred['confidence']:.1f}%)")
    print("\n  Severity probability breakdown:")
    for i, sev in enumerate(SEVERITY_LABELS):
        prob = float(pred['multi_proba'][i]) * 100 if i < len(pred['multi_proba']) else 0.0
        bar  = '█' * int(prob / 5) + '░' * (20 - int(prob / 5))
        print(f"    {sev:<38} {bar} {prob:5.1f}%")

    print("\n  " + "=" * 61)
    print(f"  FINAL RESULT : {verdict['decision']} — {verdict['headline']}")
    print("  " + "=" * 61)
    print(f"  {verdict['summary']}")
    if not verdict['bdi_agrees']:
        print(f"\n  Note: the model result differs from the BDI questionnaire score "
              f"({bdi_extrap}/63, {bdi_rule_label}).")
        print("        A clinician should review this case.")

    # ── Clinical follow-up flag ───────────────────────────────────────────────
    if needs_follow_up(pred, bdi_ref, verdict):
        print("\n  ⚠  CLINICAL FOLLOW-UP RECOMMENDED")
        print("     Please discuss these results with your oncologist or a mental")
        print("     health professional at the earliest opportunity.")

    # ── Crisis resources ──────────────────────────────────────────────────────
    if needs_crisis_resources(pred, bdi_ref, bdi_responses):
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
            top_bdi_nums  = None                                  # None = ask all 21
            top_fcri_cols = [q['col'] for q in FCRI_QUESTIONS]    # all 42 FCRI items
            est_time = '~25-30 min'

        print(f"\n  Estimated time : {est_time}")
        print("─" * 65)
        input("Press Enter to begin the assessment…")

        # ── Data collection ────────────────────────────────────────────────────
        demographics   = collect_demographics()
        bdi_responses  = collect_bdi(item_nums=top_bdi_nums)
        fcri_responses = collect_fcri(col_filter=top_fcri_cols)

        # ── Suicidal ideation warning (BDI item 9, score ≥ 2) ─────────────────
        # Item 9 is asked in both modes (with_safety_item), so this always runs.
        if suicide_risk(bdi_responses):
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

        # ── Results: BDI → FCRI → model result ─────────────────────────────────
        display_results(pred, bdi_responses, fcri_responses,
                        bdi_reference(bdi_responses), mode)

        # ── Optional SHAP explanation ────────────────────────────────────────────
        try:
            from shap_analyzer import SHAPAnalyzer
            show_explanation = input("\nShow model explanation? (y/n): ").strip().lower()
            if show_explanation == 'y':
                print("\nGenerating patient-level SHAP explanation...")
                
                # Create patient output directory
                patient_dir = os.path.join(MODELS_DIR, 'shap', 'patients')
                os.makedirs(patient_dir, exist_ok=True)
                
                shap_analyzer = SHAPAnalyzer(
                    models={'voting_ensemble': artefacts['binary_model']},
                    feature_names=artefacts['feature_cols'],
                    X_train=None,  # Not needed for patient-level
                    X_test=None,
                    scaler=artefacts['scaler'],
                    output_dir=patient_dir,
                    task='binary',
                    class_names=['No Depression', 'Has Depression'],
                    dpi=150  # Default to standard resolution
                )
                
                # Need to load training data for background dataset
                try:
                    from depression_detection_pipeline import ClinicalDataPreprocessor, TargetVariableEngineer
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import StandardScaler
                    
                    print("Loading training data for SHAP background...")
                    data_path = os.path.join('Data', 'cancer_psychology.csv')
                    if os.path.exists(data_path):
                        df_temp = pd.read_csv(data_path)
                        preprocessor_temp = ClinicalDataPreprocessor()
                        df_proc_temp, _ = preprocessor_temp.fit_transform(df_temp)
                        tve_temp = TargetVariableEngineer(preprocessor_temp.bdi_columns)
                        df_fin_temp = tve_temp.create_targets(df_proc_temp)
                        
                        feature_cols_temp = [c for c in
                                          preprocessor_temp.bdi_columns + 
                                          preprocessor_temp.fcri_columns + 
                                          [c for c in df_fin_temp.columns if c.endswith('_encoded')]
                                          if c in df_fin_temp.columns]
                        X_temp = df_fin_temp[feature_cols_temp].fillna(0)
                        y_temp = df_fin_temp['has_depression']
                        
                        X_tr, _, _, _ = train_test_split(
                            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
                        )
                        
                        scaler_temp = StandardScaler()
                        X_train_temp = scaler_temp.fit_transform(X_tr)
                        
                        # Update analyzer with training data
                        shap_analyzer.X_train = X_train_temp
                    else:
                        print("Warning: Training data not found, using minimal background")
                except Exception as e:
                    print(f"Warning: Could not load training data ({e}), using minimal background")
                
                explanation = shap_analyzer.explain_patient(
                    feature_vec=feature_vec,
                    prediction=pred,
                    bdi_responses=bdi_responses,
                    fcri_responses=fcri_responses,
                    save_plots=True
                )
                
                if explanation:
                    shap_analyzer.print_patient_explanation(explanation, hybrid_display=True)
                    print(f"\nSHAP visualizations saved to: {patient_dir}/")
                else:
                    print("Could not generate SHAP explanation.")
        except ImportError:
            print("SHAP not installed. Prediction continues without explanations.")
            print("Install SHAP with: pip install shap")
        except Exception as e:
            print(f"Error generating SHAP explanation: {e}")
            print("Prediction completed without explanations.")

        # ── Loop control ───────────────────────────────────────────────────────
        again = input("\nRun another assessment? (y/n): ").strip().lower()
        run_again = (again == 'y')

    print("\nThank you. Please ensure results are reviewed with a healthcare professional.")


if __name__ == '__main__':
    main()