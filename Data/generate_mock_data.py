"""
Mock Data Generator v2 — Cancer Psychology Assessment
=======================================================
Generates 20,000 training records that are:
  • Schema-compatible with the original source CSV (same BDI/FCRI column names and text format)
  • Enriched with all v2-form additions (Cancer Stage, Education Level, etc.)
  • Clinically plausible: FCRI subscale scores are correlated with BDI severity
  • Score-accurate: BDI items are sampled to hit exact target totals via constrained-sum algorithm
  • Fast: vectorised medication consistency, no row-by-row patching

Fixes over v1
-------------
  ✗ BDI responses sampled randomly from source text → score uncontrolled
  ✗ FCRI responses uncorrelated with BDI severity
  ✗ medication consistency patched with slow .loc loop
  ✗ Missing v2 columns (Cancer Stage, Education Level, Marital Status, etc.)
  ✗ No GAD-2 items
  ✗ "Name" field included (PII)
  ✗ No reproducibility seed
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ─── I/O paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "cancer_psychology.csv")
NUM_RECORDS = 20_000

# =============================================================================
# 1. RESPONSE TEXT LOOKUPS
#    All text values match the original source CSV exactly so the pipeline's
#    existing text→numeric conversion works unchanged.
# =============================================================================

# ── BDI items: score (0-3) → exact response text ─────────────────────────────
BDI_TEXT = {
    '1.':  {0: 'I do not feel sad.',
            1: 'I feel sad',
            2: "I am sad all the time and I can't snap out of it.",
            3: "I am so sad and unhappy that I can't stand it."},
    '2.':  {0: 'I am not particularly discouraged about the future.',
            1: 'I feel discouraged about the future.',
            2: 'I feel I have nothing to look forward to.',
            3: 'I feel the future is hopeless and that things cannot improve.'},
    '3.':  {0: 'I do not feel like a failure.',
            1: 'I feel I have failed more than the average person.',
            2: 'As I look back on my life, all I can see is a lot of failures.',
            3: 'I feel I am a complete failure as a person.'},
    '4.':  {0: 'I get as much satisfaction out of things as I used to.',
            1: "I don't enjoy things the way I used to.",
            2: "I don't get real satisfaction out of anything anymore.",
            3: 'I am dissatisfied or bored with everything.'},
    '5.':  {0: "I don't feel particularly guilty",
            1: 'I feel guilty a good part of the time.',
            2: 'I feel quite guilty most of the time.',
            3: 'I feel guilty all of the time.'},
    '6.':  {0: "I don't feel I am being punished.",
            1: 'I feel I may be punished.',
            2: 'I expect to be punished.',
            3: 'I feel I am being punished.'},
    '7.':  {0: "I don't feel disappointed in myself.",
            1: 'I am disappointed in myself.',
            2: 'I am disgusted with myself.',
            3: 'I hate myself.'},
    '8.':  {0: "I don't feel I am any worse than anybody else.",
            1: 'I am critical of myself for my weaknesses or mistakes.',
            2: 'I blame myself all the time for my faults.',
            3: 'I blame myself for everything bad that happens.'},
    '9.':  {0: "I don't have any thoughts of killing myself.",
            1: 'I have thoughts of killing myself, but I would not carry them out.',
            2: 'I would like to kill myself.',
            3: 'I would kill myself if I had the chance.'},
    '10.': {0: "I don't cry any more than usual.",
            1: 'I cry more now than I used to.',
            2: 'I cry all the time now.',
            3: "I used to be able to cry, but now I can't cry even though I want to."},
    '11.': {0: 'I am no more irritated by things than I ever was.',
            1: 'I am slightly more irritated now than usual.',
            2: 'I am quite annoyed or irritated a good deal of the time.',
            3: 'I feel irritated all the time.'},
    '12.': {0: 'I have not lost interest in other people.',
            1: 'I am less interested in other people than I used to be.',
            2: 'I have lost most of my interest in other people.',
            3: 'I have lost all of my interest in other people.'},
    '13.': {0: 'I make decisions about as well as I ever could.',
            1: 'I put off making decisions more than I used to.',
            2: 'I have greater difficulty in making decisions more than I used to.',
            3: "I can't make decisions at all anymore."},
    '14.': {0: "I don't feel that I look any worse than I used to.",
            1: 'I am worried that I am looking old or unattractive.',
            2: 'I feel there are permanent changes in my appearance that make me look unattractive',
            3: 'I believe that I look ugly.'},
    '15.': {0: 'I can work about as well as before.',
            1: 'It takes an extra effort to get started at doing something.',
            2: 'I have to push myself very hard to do anything.',
            3: "I can't do any work at all."},
    '16.': {0: 'I can sleep as well as usual.',
            1: "I don't sleep as well as I used to.",
            2: 'I wake up 1-2 hours earlier than usual and find it hard to get back to sleep.',
            3: 'I wake up several hours earlier than I used to and cannot get back to sleep.'},
    '17.': {0: "I don't get more tired than usual.",
            1: 'I get tired more easily than I used to.',
            2: 'I get tired from doing almost anything.',
            3: 'I am too tired to do anything.'},
    '18.': {0: 'My appetite is no worse than usual.',
            1: 'My appetite is not as good as it used to be.',
            2: 'My appetite is much worse now.',
            3: 'I have no appetite at all anymore.'},
    '19.': {0: "I haven't lost much weight, if any, lately.",
            1: 'I have lost more than five pounds.',
            2: 'I have lost more than ten pounds.',
            3: 'I have lost more than fifteen pounds.'},
    '20.': {0: 'I am no more worried about my health than usual.',
            1: 'I am worried about physical problems like aches, pains, upset stomach, or constipation.',
            2: "I am very worried about physical problems and it's hard to think of much else.",
            3: 'I am so worried about my physical problems that I cannot think of anything else.'},
    '21.': {0: 'I have not noticed any recent change in my interest in sex.',
            1: 'I am less interested in sex than I used to be.',
            2: 'I have almost no interest in sex.',
            3: 'I have lost interest in sex completely.'},
}

# ── FCRI response text maps (keyed by 0-4) ───────────────────────────────────
FREQ    = {0: 'Never', 1: 'Rarely', 2: 'Sometimes', 3: 'Most of the time', 4: 'All the time'}
INTENS  = {0: 'Not at all', 1: 'A little', 2: 'Somewhat', 3: 'A lot', 4: 'A great deal'}
F14     = {0: 'Not at all at risk', 1: 'A little at risk', 2: 'Somewhat at risk',
           3: 'A lot at risk', 4: 'A great deal at risk'}
F15     = {0: 'Never', 1: 'A few times a month', 2: 'A few times a week',
           3: 'A few times a day', 4: 'Several times a day'}
F16     = {0: "I don't think about it", 1: 'A few seconds', 2: 'A few minutes',
           3: 'A few hours', 4: 'Several hours'}
F17     = {0: "I don't think about it", 1: 'A few weeks', 2: 'A few months',
           3: 'A few years', 4: 'Several years'}

# ── FCRI column names — exact match with source CSV ───────────────────────────
FCRI_COLS = [
    '1. Television shows or newspaper articles about cancer or illness',
    '2. An appointment with my doctor or other health professional',
    '3. Medical examinations (e.g. annual check-up, blood tests, X-rays)',
    '4. Conversations about cancer or illness in general',
    '5. Seeing or hearing about someone who is ill',
    '6. Going to a funeral or reading the obituary section of the paper',
    '7. When I feel unwell physically or when I am sick',
    '8. Generally, I avoid situations or things that make me think about the possibility of cancer\nrecurrence',
    '9. I am worried or anxious about the possibility of cancer recurrence',
    '10. I am afraid of cancer recurrence',
    '11. I believe it is normal to be worried or anxious about the possibility of cancer recurrence',
    '12. When I think about the possibility of cancer recurrence, this triggers other unpleasant\nthoughts or images (such as death, suffering, the consequences for my family)',
    '13. I believe that I am cured and that the cancer will not come back',
    '14. In your opinion, are you at risk of having a cancer recurrence?',
    '15. How often do you think about the possibility of cancer recurrence?',
    '16. How much time per day do you spend thinking about the possibility of cancer recurrence?',
    '17. How long have you been thinking about the possibility of cancer recurrence?',
    '18. Worry, fear or anxiety',
    '19. Sadness, discouragement or disappointment',
    '20. Frustration, anger or outrage',
    '21. Helplessness or resignation',
    '22. My social or leisure activities (e.g. outings, sports, travel)',
    '23. My work or everyday activities',
    '24. My relationships with my partner, my family, or those close to me',
    '25. My ability to make future plans or set life goals',
    '26. My state of mind or my mood',
    '27. My quality of life in general',
    '28. I feel that I worry excessively about the possibility of cancer recurrence',
    '29. Other people think that I worry excessively about the possibility of cancer recurrence',
    '30. I think that I worry more about the possibility of cancer recurrence than other people who\nhave been diagnosed with cancer',
    '31. I call my doctor or other health professional',
    '32. I go to the hospital or clinic for an examination',
    '33. I examine myself to see if I have any physical signs of cancer',
    '34. I try to distract myself (e.g. do various activities, watch television, read, work)',
    '35. I try not to think about it, to get the idea out of my mind',
    '36. I pray, meditate or do relaxation',
    '37. I try to convince myself that everything will be fine or I think positively',
    '38. I talk to someone about it',
    '39. I try to understand what is happening and deal with it',
    '40. I try to find a solution',
    '41. I try to replace this thought with a more pleasant one',
    '42. I tell myself "stop it"',
    'Do you feel reassured when you use these strategies?',
]

# =============================================================================
# 2. DEMOGRAPHIC & CLINICAL OPTION LISTS
# =============================================================================
GENDERS          = ['Male', 'Female']
GENDER_PROBS     = [0.35, 0.65]            # breast cancer skews female

FEMALE_ONLY      = {'Breast Cancer', 'Cervical Cancer', 'Ovarian Cancer'}
MALE_ONLY        = {'Prostate Cancer'}
ALL_CANCERS      = ['Blood Cancer (Leukemia / Lymphoma)', 'Lung Cancer', 'Stomach Cancer',
                    'Breast Cancer', 'Ovarian Cancer', 'Colorectal Cancer', 'Brain Tumour',
                    'Prostate Cancer', 'Cervical Cancer', 'Liver Cancer']
FEMALE_CANCERS   = [c for c in ALL_CANCERS if c not in MALE_ONLY]
MALE_CANCERS     = [c for c in ALL_CANCERS if c not in FEMALE_ONLY]

PROVINCES        = ['Punjab', 'Sindh', 'Khyber Pakhtunkhwa', 'Balochistan',
                    'Islamabad Capital Territory', 'Gilgit-Baltistan', 'Azad Jammu & Kashmir']
PROV_PROBS       = [0.45, 0.20, 0.15, 0.07, 0.06, 0.04, 0.03]

AGE_GROUPS       = ['Under 18', '18-30', '31-45', '46-60', 'Over 60']
AGE_PROBS        = [0.03, 0.15, 0.30, 0.32, 0.20]

STATUSES         = ['Newly Diagnosed', 'Under Active Treatment', 'In Remission',
                    'Cancer Recurrence', 'Metastatic Disease', 'Palliative Care']
DURATIONS        = ['Less than 3 months', '3-6 months', '6-12 months', '1-2 years', 'More than 2 years']
TREATMENTS       = ['Chemotherapy', 'Radiotherapy', 'Surgery', 'Immunotherapy',
                    'Hormone Therapy', 'Targeted Therapy', 'Palliative Care', 'Follow-up Only']

MED_TYPES        = ['Antidepressants', 'Anti-anxiety medications', 'Mood stabilizers',
                    'Sleep medications', 'Other']
MED_DURATIONS    = ['Less than 1 month', '1-6 months', '6-12 months', 'More than 1 year']
MED_EFFECTS      = ['Significantly', 'Somewhat', 'No Change', 'Worse']

# ── v2 form additions ─────────────────────────────────────────────────────────
SITES            = ['Pakistan Institute of Medical Sciences (PIMS)',
                    'Shaukat Khanum Memorial Cancer Hospital & Research Centre (SKMCH&RC)']
STAGES           = ['Stage I', 'Stage II', 'Stage III', 'Stage IV', 'Metastatic / Advanced',
                    'Not sure / not told']
EDUCATION        = ['No formal education', 'Primary (up to Grade 5)',
                    'Secondary / Matric / O-Level', 'Intermediate / FSc / A-Level',
                    "Bachelor's degree", "Master's or higher"]
EDU_PROBS        = [0.08, 0.12, 0.28, 0.22, 0.20, 0.10]
MARITAL          = ['Single', 'Married', 'Divorced or Separated', 'Widowed']
MARITAL_PROBS    = [0.18, 0.65, 0.07, 0.10]
EMPLOYMENT       = ['Employed full-time', 'Employed part-time', 'Self-employed',
                    'Unemployed / looking for work', 'Homemaker', 'Student', 'Retired',
                    'Unable to work due to illness']
PRIOR_PSYCH      = ['Yes', 'No', 'Not sure']
PRIOR_PROBS      = [0.20, 0.70, 0.10]
OOP_BANDS        = ['Less than PKR 10,000', 'PKR 10,001-25,000', 'PKR 25,001-50,000',
                    'PKR 50,001-100,000', 'More than PKR 100,000']
DIST_FIN         = ['Never', 'Rarely', 'Sometimes', 'Often', 'Always']

# ── Severity distribution (target class balance) ──────────────────────────────
SEVERITY_NAMES   = ['normal', 'mild', 'borderline', 'moderate', 'severe', 'extreme']
SEVERITY_PROBS   = [0.20, 0.15, 0.15, 0.30, 0.15, 0.05]
SEVERITY_RANGES  = {'normal': (0, 10), 'mild': (11, 16), 'borderline': (17, 20),
                    'moderate': (21, 30), 'severe': (31, 40), 'extreme': (41, 50)}

# Medication likelihood per severity (higher severity → more likely on medication)
MED_PROB_BY_SEV  = {'normal': 0.10, 'mild': 0.20, 'borderline': 0.28,
                    'moderate': 0.40, 'severe': 0.55, 'extreme': 0.65}

# =============================================================================
# 3. CORE SAMPLING FUNCTIONS
# =============================================================================

def _sample_constrained_sum(target: int, n: int, max_val: int) -> np.ndarray:
    """
    Sample n non-negative integers, each ≤ max_val, that sum to exactly `target`.
    Uses sequential bounded sampling then shuffles to remove positional bias.
    """
    target = int(np.clip(target, 0, n * max_val))
    result = np.zeros(n, dtype=np.int32)
    remaining = target
    for i in range(n - 1):
        lo = max(0, remaining - (n - i - 1) * max_val)
        hi = min(max_val, remaining)
        result[i] = int(rng.integers(lo, hi + 1))
        remaining -= result[i]
    result[n - 1] = remaining  # always in [0, max_val] by construction
    rng.shuffle(result)
    return result


def _noisy_int(base: float, std: float, lo: int, hi: int) -> int:
    """Sample a noisy integer in [lo, hi] around base."""
    return int(np.clip(round(base + rng.normal(0.0, std)), lo, hi))


def _subscale_target(distress: float, n_items: int, max_item: int,
                     strength: float, floor: float = 0.0) -> int:
    """Derive a correlated FCRI subscale total from a normalised distress level."""
    max_total = n_items * max_item
    mean = (floor + distress * strength) * max_total
    raw  = mean + rng.normal(0.0, 0.12 * max_total)
    return int(np.clip(round(raw), 0, max_total))


# =============================================================================
# 4. MAIN GENERATION LOOP
# =============================================================================

def generate_record(severity: str) -> dict:
    row = {}

    # ── Target BDI and distress level ─────────────────────────────────────────
    lo, hi      = SEVERITY_RANGES[severity]
    target_bdi  = int(rng.integers(lo, hi + 1))
    distress    = target_bdi / 63.0          # normalised 0-1

    # ── BDI: constrained-sum item scores, converted to exact text ─────────────
    bdi_scores = _sample_constrained_sum(target_bdi, 21, 3)
    for j, col in enumerate([f'{i}.' for i in range(1, 22)]):
        row[col] = BDI_TEXT[col][bdi_scores[j]]

    # ── FCRI: subscale totals correlated with distress, then text-encoded ──────
    # Triggers (1-8): frequency scale — moderate correlation
    t_scores = _sample_constrained_sum(_subscale_target(distress, 8, 4, 0.55, 0.05), 8, 4)
    # Severity 9-12: intensity — high correlation
    s912  = _sample_constrained_sum(_subscale_target(distress, 4, 4, 0.85), 4, 4)
    # Item 13 (I believe I am cured — REVERSED by pipeline): inversely correlated
    s13_raw = _noisy_int((1.0 - distress) * 4.0, 0.7, 0, 4)
    # Items 14-17: individual special scales
    s14 = _noisy_int(distress * 4.0,        0.6, 0, 4)
    s15 = _noisy_int(distress * 4.0,        0.7, 0, 4)
    s16 = _noisy_int(distress * 4.0,        0.7, 0, 4)
    s17 = _noisy_int(distress * 3.0 + 0.5, 0.8, 0, 4)   # duration: less correlated
    # Distress (18-21): intensity — high
    d_scores = _sample_constrained_sum(_subscale_target(distress, 4, 4, 0.80), 4, 4)
    # Functioning (22-27): intensity — high
    f_scores = _sample_constrained_sum(_subscale_target(distress, 6, 4, 0.75), 6, 4)
    # Insight (28-30): intensity — high
    i_scores = _sample_constrained_sum(_subscale_target(distress, 3, 4, 0.70), 3, 4)
    # Reassurance (31-33): frequency — low correlation (variable coping)
    r_scores = _sample_constrained_sum(_subscale_target(distress, 3, 4, 0.35, 0.10), 3, 4)
    # Coping (34-42): frequency — low-moderate
    c_scores = _sample_constrained_sum(_subscale_target(distress, 9, 4, 0.40, 0.15), 9, 4)
    # Reassurance effectiveness: inversely correlated with distress
    reass_eff = _noisy_int((1.0 - distress) * 4.0, 0.7, 0, 4)

    fcri_scores = (
        [FREQ[s]   for s in t_scores]   +   # items 1-8
        [INTENS[s] for s in s912]        +   # items 9-12
        [INTENS[s13_raw]]                +   # item 13
        [F14[s14], F15[s15], F16[s16], F17[s17]] +  # items 14-17
        [INTENS[s] for s in d_scores]    +   # items 18-21
        [INTENS[s] for s in f_scores]    +   # items 22-27
        [INTENS[s] for s in i_scores]    +   # items 28-30
        [FREQ[s]   for s in r_scores]    +   # items 31-33
        [FREQ[s]   for s in c_scores]    +   # items 34-42
        [FREQ[reass_eff]]                    # reassurance effectiveness
    )
    for col, val in zip(FCRI_COLS, fcri_scores):
        row[col] = val

    # ── Demographics ──────────────────────────────────────────────────────────
    gender = rng.choice(GENDERS, p=GENDER_PROBS)
    row['Gender']     = gender
    row['Age Group']  = rng.choice(AGE_GROUPS, p=AGE_PROBS)
    row['Povince']    = rng.choice(PROVINCES,  p=PROV_PROBS)

    cancer_pool = FEMALE_CANCERS if gender == 'Female' else MALE_CANCERS
    row['  Cancer Type  ']               = rng.choice(cancer_pool)
    row['  Duration Since Diagnosis  ']  = rng.choice(DURATIONS)
    row['  Current Cancer Status  ']     = rng.choice(STATUSES)
    row['  Current Treatment Type  ']    = rng.choice(TREATMENTS)

    # ── v2 additions ──────────────────────────────────────────────────────────
    row['Data Collection Site'] = rng.choice(SITES)
    row['Cancer Stage']         = rng.choice(STAGES)
    row['Education Level']      = rng.choice(EDUCATION, p=EDU_PROBS)
    row['Marital Status']       = rng.choice(MARITAL,   p=MARITAL_PROBS)
    row['Employment Status']    = rng.choice(EMPLOYMENT)
    row['Prior Psych History']  = rng.choice(PRIOR_PSYCH, p=PRIOR_PROBS)

    # ── Socioeconomic ─────────────────────────────────────────────────────────
    row['OOP Expenditure']      = rng.choice(OOP_BANDS)
    row['Distress Financing']   = rng.choice(DIST_FIN)
    row['Social Support']       = _noisy_int((1.0 - distress) * 4.0 + 1.0, 0.9, 1, 5)

    # ── GAD-2 (correlated with distress) ─────────────────────────────────────
    row['GAD2_1'] = _noisy_int(distress * 3.0, 0.6, 0, 3)
    row['GAD2_2'] = _noisy_int(distress * 3.0, 0.6, 0, 3)

    # ── Medication (probability correlated with severity) ─────────────────────
    on_meds = rng.random() < MED_PROB_BY_SEV[severity]
    med_col = ('Are you currently taking any medication for depression, anxiety, stress, '
               'or other emotional/psychological problems?  ')
    row[med_col] = 'Yes' if on_meds else 'No'
    if on_meds:
        row['If yes, what type of medication are you taking?  ']      = rng.choice(MED_TYPES)
        row['How long have you been taking this medication?  ']        = rng.choice(MED_DURATIONS)
        row['Was this medication prescribed by a healthcare professional? '] = rng.choice(['Yes', 'No'], p=[0.85, 0.15])
        row['Do you feel the medication has improved your emotional well-being? '] = rng.choice(MED_EFFECTS)
    else:
        row['If yes, what type of medication are you taking?  ']       = 'Not Applicable'
        row['How long have you been taking this medication?  ']         = 'Not Applicable'
        row['Was this medication prescribed by a healthcare professional? '] = 'Not Applicable'
        row['Do you feel the medication has improved your emotional well-being? '] = 'Not Applicable'

    return row


# =============================================================================
# 5. RUN
# =============================================================================

def main():
    print(f"Generating {NUM_RECORDS:,} records (seed={RANDOM_STATE})…")

    # Assign severity classes up-front for exact count control
    severities = rng.choice(SEVERITY_NAMES, size=NUM_RECORDS, p=SEVERITY_PROBS)

    records = []
    start_dt = datetime(2025, 1, 1)
    end_dt   = datetime(2026, 6, 1)
    total_secs = int((end_dt - start_dt).total_seconds())

    for idx, severity in enumerate(severities):
        row = generate_record(severity)
        # Anonymised patient ID — no names
        row['Patient Code / ID'] = f'MOCK-{idx+1:05d}'
        row['Timestamp'] = (start_dt + timedelta(
            seconds=int(rng.integers(0, total_secs))
        )).strftime('%d/%m/%Y %H:%M:%S')

        records.append(row)
        if (idx + 1) % 5_000 == 0:
            print(f"  {idx+1:,} records generated…")

    df = pd.DataFrame(records)

    # ── Severity distribution check ────────────────────────────────────────────
    print("\nSeverity distribution:")
    sev_counts = pd.Series(severities).value_counts()
    for sev in SEVERITY_NAMES:
        print(f"  {sev:<12}: {sev_counts.get(sev, 0):>6,}  "
              f"({sev_counts.get(sev, 0)/NUM_RECORDS*100:.1f}%)")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. {len(df):,} records written → {OUTPUT_FILE}")


if __name__ == '__main__':
    main()