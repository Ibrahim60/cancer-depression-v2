"""
Mock Data Generator v3 — Cancer Psychology Assessment
====================================================
Generates training records that are:
  • Schema-compatible with auto_data_submitter/test.csv (same column names and format)
  • Clinically plausible: FCRI subscale scores are correlated with BDI severity
  • Score-accurate: BDI items are sampled to hit exact target totals via constrained-sum algorithm
  • Fast: vectorised medication consistency, no row-by-row patching

Schema
------
  • Name, Gender, Age Group, Povince (keeping typo to match test.csv)
  • Cancer Type, Duration Since Diagnosis, Current Cancer Status, Current Treatment Type
  • Taking Medication, Medication Type, Medication Duration, Prescribed by Professional, Improved Well-being
  • BDI_1 through BDI_21 (text responses)
  • FCRI_1 through FCRI_42, FCRI_Reassured (text responses)
"""

import os
import itertools
import random
import numpy as np
import pandas as pd

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ─── I/O paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "cancer_psychology.csv")
NUM_RECORDS = 3000

# =============================================================================
# 1. RESPONSE TEXT LOOKUPS
#    All text values match the Google Form exactly
# =============================================================================

# ── BDI items: score (0-3) → exact response text ─────────────────────────────
BDI_TEXT = {
    'BDI_1':  {0: 'I do not feel sad.',
            1: 'I feel sad',
            2: "I am sad all the time and I can't snap out of it.",
            3: "I am so sad and unhappy that I can't stand it."},
    'BDI_2':  {0: 'I am not particularly discouraged about the future.',
            1: 'I feel discouraged about the future.',
            2: 'I feel I have nothing to look forward to.',
            3: 'I feel the future is hopeless and that things cannot improve.'},
    'BDI_3':  {0: 'I do not feel like a failure.',
            1: 'I feel I have failed more than the average person.',
            2: 'As I look back on my life, all I can see is a lot of failures.',
            3: 'I feel I am a complete failure as a person.'},
    'BDI_4':  {0: 'I get as much satisfaction out of things as I used to.',
            1: "I don't enjoy things the way I used to.",
            2: "I don't get real satisfaction out of anything anymore.",
            3: 'I am dissatisfied or bored with everything.'},
    'BDI_5':  {0: "I don't feel particularly guilty",
            1: 'I feel guilty a good part of the time.',
            2: 'I feel quite guilty most of the time.',
            3: 'I feel guilty all of the time.'},
    'BDI_6':  {0: "I don't feel I am being punished.",
            1: 'I feel I may be punished.',
            2: 'I expect to be punished.',
            3: 'I feel I am being punished.'},
    'BDI_7':  {0: "I don't feel disappointed in myself.",
            1: 'I am disappointed in myself.',
            2: 'I am disgusted with myself.',
            3: 'I hate myself.'},
    'BDI_8':  {0: "I don't feel I am any worse than anybody else.",
            1: 'I am critical of myself for my weaknesses or mistakes.',
            2: 'I blame myself all the time for my faults.',
            3: 'I blame myself for everything bad that happens.'},
    'BDI_9':  {0: "I don't have any thoughts of killing myself.",
            1: 'I have thoughts of killing myself, but I would not carry them out.',
            2: 'I would like to kill myself.',
            3: 'I would kill myself if I had the chance.'},
    'BDI_10': {0: "I don't cry any more than usual.",
            1: 'I cry more now than I used to.',
            2: 'I cry all the time now.',
            3: "I used to be able to cry, but now I can't cry even though I want to."},
    'BDI_11': {0: 'I am no more irritated by things than I ever was.',
            1: 'I am slightly more irritated now than usual.',
            2: 'I am quite annoyed or irritated a good deal of the time.',
            3: 'I feel irritated all the time.'},
    'BDI_12': {0: 'I have not lost interest in other people.',
            1: 'I am less interested in other people than I used to be.',
            2: 'I have lost most of my interest in other people.',
            3: 'I have lost all of my interest in other people.'},
    'BDI_13': {0: 'I make decisions about as well as I ever could.',
            1: 'I put off making decisions more than I used to.',
            2: 'I have greater difficulty in making decisions more than I used to.',
            3: "I can't make decisions at all anymore."},
    'BDI_14': {0: "I don't feel that I look any worse than I used to.",
            1: 'I am worried that I am looking old or unattractive.',
            2: 'I feel there are permanent changes in my appearance that make me look unattractive',
            3: 'I believe that I look ugly.'},
    'BDI_15': {0: 'I can work about as well as before.',
            1: 'It takes an extra effort to get started at doing something.',
            2: 'I have to push myself very hard to do anything.',
            3: "I can't do any work at all."},
    'BDI_16': {0: 'I can sleep as well as usual.',
            1: "I don't sleep as well as I used to.",
            2: 'I wake up 1-2 hours earlier than usual and find it hard to get back to sleep.',
            3: 'I wake up several hours earlier than I used to and cannot get back to sleep.'},
    'BDI_17': {0: "I don't get more tired than usual.",
            1: 'I get tired more easily than I used to.',
            2: 'I get tired from doing almost anything.',
            3: 'I am too tired to do anything.'},
    'BDI_18': {0: 'My appetite is no worse than usual.',
            1: 'My appetite is not as good as it used to be.',
            2: 'My appetite is much worse now.',
            3: 'I have no appetite at all anymore.'},
    'BDI_19': {0: "I haven't lost much weight, if any, lately.",
            1: 'I have lost more than five pounds.',
            2: 'I have lost more than ten pounds.',
            3: 'I have lost more than fifteen pounds.'},
    'BDI_20': {0: 'I am no more worried about my health than usual.',
            1: 'I am worried about physical problems like aches, pains, upset stomach, or constipation.',
            2: "I am very worried about physical problems and it's hard to think of much else.",
            3: 'I am so worried about my physical problems that I cannot think of anything else.'},
    'BDI_21': {0: 'I have not noticed any recent change in my interest in sex.',
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

# ── FCRI column names — simple format to match test.csv ───────────────────────
FCRI_COLS = [f'FCRI_{i}' for i in range(1, 43)] + ['FCRI_Reassured']

# =============================================================================
# 2. DEMOGRAPHIC & CLINICAL OPTION LISTS
# =============================================================================
# ── Name generation ─────────────────────────────────────────────────────────
FIRST_NAMES_MALE = [
    'Ahmed', 'Ali', 'Hassan', 'Hussain', 'Omar', 'Usman', 'Bilal', 'Kashif', 'Imran', 'Junaid',
    'Saad', 'Farhan', 'Rizwan', 'Tariq', 'Nasir', 'Kamran', 'Arif', 'Sajid', 'Shahid', 'Yasir',
    'Waqas', 'Adeel', 'Faisal', 'Zeeshan', 'Asad', 'Shoaib', 'Naveed', 'Salman', 'Fahad', 'Danish',
    'Haris', 'Zaid', 'Umair', 'Abdullah', 'Talha', 'Mohsin', 'Waleed', 'Sohail', 'Aamir', 'Noman',
    'Ayaan', 'Hamza', 'Ibrahim', 'Yousaf', 'Shahzad', 'Amjad', 'Asim', 'Khalid', 'Majid', 'Adnan',
    'Gohar', 'Sarfaraz', 'Zubair', 'Naeem', 'Rashid', 'Jamil', 'Ehsan', 'Faraz', 'Qasim', 'Owais',
    'Sikandar', 'Zafar', 'Iftikhar', 'Mudassar', 'Shakeel', 'Adil', 'Basit', 'Daniyal', 'Ehtisham', 'Fawad',
    'Ghazanfar', 'Haseeb', 'Ismail', 'Jawad', 'Karam', 'Luqman', 'Moiz', 'Noor-ul-Amin', 'Osama', 'Parvez',
    'Qaiser', 'Raheel', 'Sameer', 'Taimoor', 'Uzair', 'Vaqar', 'Wahaj', 'Yahya', 'Zohaib', 'Abrar',
    'Bashir', 'Dawood', 'Ghulam', 'Habib', 'Israr', 'Jahangir', 'Kaleem', 'Laraib', 'Muzammil', 'Nauman',
]
FIRST_NAMES_FEMALE = [
    'Fatima', 'Ayesha', 'Zara', 'Hira', 'Sana', 'Maryam', 'Khadija', 'Aisha', 'Zainab', 'Hafsa',
    'Saima', 'Nadia', 'Sadia', 'Rubina', 'Shazia', 'Farah', 'Kiran', 'Iqra', 'Anum', 'Mehwish',
    'Amina', 'Areeba', 'Bushra', 'Sobia', 'Rabia', 'Uzma', 'Samina', 'Nida', 'Rida', 'Mahnoor',
    'Alishba', 'Laiba', 'Noor', 'Sundas', 'Tayyaba', 'Wajiha', 'Yusra', 'Zoya', 'Mariam', 'Asma',
    'Shabana', 'Yasmin', 'Naila', 'Farzana', 'Ghazala', 'Huma', 'Javeria', 'Kinza', 'Mahjabeen', 'Neelam',
    'Palwasha', 'Quratulain', 'Rukhsana', 'Sidra', 'Tehmina', 'Umaira', 'Warda', 'Amber', 'Beenish', 'Dua',
    'Erum', 'Fariha', 'Gulnaz', 'Hina', 'Ifra', 'Jaweria', 'Kausar', 'Lubna', 'Mavra', 'Nazia',
    'Ozma', 'Parveen', 'Qandeel', 'Rimsha', 'Sabahat', 'Tuba', 'Ushna', 'Wardah', 'Xoya', 'Yumna',
    'Zubaida', 'Amara', 'Bisma', 'Ceeza', 'Duaa', 'Esha', 'Fajar', 'Gulalai', 'Hoorain', 'Ifrah',
]
LAST_NAMES = [
    'Khan', 'Ahmed', 'Ali', 'Hussain', 'Malik', 'Siddiqui', 'Shaikh', 'Butt', 'Raza', 'Ahmad',
    'Farooq', 'Hassan', 'Mirza', 'Qureshi', 'Chaudhry', 'Iqbal', 'Aslam', 'Akbar', 'Rehman', 'Saeed',
    'Bhatti', 'Awan', 'Gill', 'Warraich', 'Cheema', 'Dar', 'Bhutto', 'Zardari', 'Leghari', 'Bugti',
    'Marwat', 'Yousafzai', 'Afridi', 'Wazir', 'Baloch', 'Soomro', 'Abbasi', 'Durrani', 'Ghaznavi',
    'Rana', 'Sheikh', 'Javed', 'Anwar', 'Sultan', 'Hashmi', 'Naqvi', 'Zaidi', 'Kayani', 'Suri',
    'Tahir', 'Bajwa', 'Randhawa', 'Sandhu', 'Virk', 'Joyia', 'Khokhar', 'Langah', 'Niazi', 'Orakzai',
    'Peracha', 'Qazi', 'Rajput', 'Satti', 'Tanoli', 'Utmanzai', 'Vardag', 'Watto', 'Yusufzai', 'Zaman',
    'Abro', 'Bughio', 'Channa', 'Dahri', 'Junejo', 'Khaskheli', 'Lund', 'Memon', 'Nizamani', 'Panhwar',
]


class NameProvider:
    """
    Hands out full names with NO repeats until every combo in the pool has
    been used, then reshuffles and starts over. Avoids the birthday-paradox
    collision rate you get from independently sampling first/last name each call
    (e.g. ~15% duplicate rate at 1,000 draws from a 3,185-combo space).

    Uses the module-level `rng` (np.random.default_rng) so output stays
    reproducible under RANDOM_STATE.
    """
    def __init__(self, rng: np.random.Generator):
        self._rng = rng
        self._pools = {
            'Male':   self._build_pool(FIRST_NAMES_MALE),
            'Female': self._build_pool(FIRST_NAMES_FEMALE),
        }
        self._cursors = {'Male': 0, 'Female': 0}

    def _build_pool(self, first_names: list) -> list:
        combos = [f"{f} {l}" for f, l in itertools.product(first_names, LAST_NAMES)]
        self._rng.shuffle(combos)  # in-place shuffle, works on python lists too
        return combos

    def next(self, gender: str) -> str:
        key = 'Female' if gender == 'Female' else 'Male'
        pool = self._pools[key]
        idx = self._cursors[key]
        if idx >= len(pool):
            # Exhausted the full name space — reshuffle and recycle.
            self._rng.shuffle(pool)
            idx = 0
        name = pool[idx]
        self._cursors[key] = idx + 1
        return name


# Single provider shared across the whole generation run so uniqueness holds
# across all records, not just within one call.
NAME_PROVIDER = NameProvider(rng)

GENDERS          = ['Male', 'Female']
GENDER_PROBS     = [0.35, 0.65]            # breast cancer skews female

FEMALE_ONLY      = {'Breast Cancer', 'Cervical Cancer', 'Ovarian Cancer'}
MALE_ONLY        = {'Prostate Cancer'}
ALL_CANCERS      = ['Breast Cancer', 'Lung Cancer', 'Colorectal Cancer',
                    'Blood Cancer (Leukemia / Lymphoma)', 'Ovarian Cancer',
                    'Cervical Cancer', 'Prostate Cancer', 'Stomach Cancer']
FEMALE_CANCERS   = [c for c in ALL_CANCERS if c not in MALE_ONLY]
MALE_CANCERS     = [c for c in ALL_CANCERS if c not in FEMALE_ONLY]

PROVINCES        = ['Punjab', 'Sindh', 'Khyber Pakhtunkhwa (KPK)', 'Balochistan',
                    'Islamabad Capital Territory', 'Gilgit Baltistan', 'Azad Jammu & Kashmir']
PROV_PROBS       = [0.45, 0.20, 0.15, 0.07, 0.06, 0.04, 0.03]

AGE_GROUPS       = ['Below 18', '18-30', '31-45', '46-60', 'Above 60']
AGE_PROBS        = [0.03, 0.15, 0.30, 0.32, 0.20]

STATUSES         = ['Newly Diagnosed', 'Under Treatment', 'Recovered / Survivor', 'Recurrence']
DURATIONS        = ['Less than 3 months', '3-6 months', '6-12 months', '1-2 years', 'More than 2 years']
TREATMENTS       = ['Chemotherapy', 'Radiotherapy', 'Surgery', 'Targeted Therapy',
                    'Immunotherapy', 'Not currently in treatment']

MED_TYPES        = ['Antidepressants', 'Sleeping medication']
MED_DURATIONS    = ['Less than 1 month', '1–6 months', '6–12 months', 'More than 1 year']
MED_EFFECTS      = ['Yes', 'No', 'Not sure']

# ── Severity distribution (target class balance) ──────────────────────────────
SEVERITY_NAMES   = ['normal', 'mild', 'borderline', 'moderate', 'severe', 'extreme']
SEVERITY_PROBS   = [0.20, 0.15, 0.15, 0.30, 0.15, 0.05]
SEVERITY_RANGES  = {'normal': (0, 10), 'mild': (11, 16), 'borderline': (17, 20),
                    'moderate': (21, 30), 'severe': (31, 40), 'extreme': (41, 50)}

# Medication likelihood per severity (higher severity → more likely on medication)
MED_PROB_BY_SEV  = {'normal': 0.10, 'mild': 0.20, 'borderline': 0.28,
                    'moderate': 0.40, 'severe': 0.55, 'extreme': 0.65}

# =============================================================================
# 2b. CROSS-FIELD REALISM WEIGHTS
#     These tie clinically/logically related fields together so records read
#     as coherent cases instead of independently-shuffled attributes.
# =============================================================================

# Psychological severity → likely clinical status. Higher distress skews
# toward "Newly Diagnosed"/"Recurrence"; low distress skews toward
# "Recovered / Survivor". Order matches STATUSES; each row sums to 1.0.
STATUS_WEIGHTS_BY_SEVERITY = {
    'normal':     [0.10, 0.15, 0.65, 0.10],
    'mild':       [0.15, 0.20, 0.50, 0.15],
    'borderline': [0.20, 0.25, 0.35, 0.20],
    'moderate':   [0.28, 0.30, 0.22, 0.20],
    'severe':     [0.32, 0.30, 0.12, 0.26],
    'extreme':    [0.35, 0.22, 0.05, 0.38],
}

# Clinical status → duration since diagnosis. "Newly Diagnosed" can't have a
# 2-year-old duration; "Recovered / Survivor" is unlikely to be 3 months out.
# Order matches DURATIONS; each row sums to 1.0.
DURATION_WEIGHTS_BY_STATUS = {
    'Newly Diagnosed':      [0.55, 0.30, 0.10, 0.04, 0.01],
    'Under Treatment':      [0.10, 0.25, 0.30, 0.25, 0.10],
    'Recovered / Survivor': [0.02, 0.03, 0.10, 0.35, 0.50],
    'Recurrence':           [0.02, 0.05, 0.13, 0.35, 0.45],
}

# Clinical status → current treatment. Survivors are overwhelmingly "Not
# currently in treatment"; active/newly-diagnosed/recurrence cases skew
# toward active modalities. Order matches TREATMENTS; each row sums to 1.0.
TREATMENT_WEIGHTS_BY_STATUS = {
    'Newly Diagnosed':      [0.28, 0.15, 0.30, 0.12, 0.05, 0.10],
    'Under Treatment':      [0.32, 0.20, 0.10, 0.18, 0.12, 0.08],
    'Recovered / Survivor': [0.03, 0.02, 0.03, 0.02, 0.02, 0.88],
    'Recurrence':           [0.30, 0.20, 0.08, 0.20, 0.15, 0.07],
}

# Duration since diagnosis → max plausible medication duration index, so a
# patient diagnosed 3 months ago can't be shown on meds for "More than 1 year".
_DURATION_INDEX = {name: i for i, name in enumerate(DURATIONS)}
MED_DURATION_CAP_BY_DIAGNOSIS_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}

# Cancer type incidence weight by age group (relative, not literal
# epidemiology — just enough shape that a 20-year-old isn't routinely drawn
# for prostate/lung cancer and a child isn't drawn for cervical cancer).
CANCER_AGE_WEIGHTS = {
    'Breast Cancer':                       {'Below 18': 0.5, '18-30': 3,   '31-45': 10, '46-60': 12, 'Above 60': 8},
    'Lung Cancer':                         {'Below 18': 0.1, '18-30': 0.5, '31-45': 2,  '46-60': 8,  'Above 60': 14},
    'Colorectal Cancer':                   {'Below 18': 0.2, '18-30': 1,   '31-45': 4,  '46-60': 9,  'Above 60': 12},
    'Blood Cancer (Leukemia / Lymphoma)':  {'Below 18': 10,  '18-30': 6,   '31-45': 5,  '46-60': 6,  'Above 60': 9},
    'Ovarian Cancer':                      {'Below 18': 0.3, '18-30': 3,   '31-45': 8,  '46-60': 9,  'Above 60': 6},
    'Cervical Cancer':                     {'Below 18': 0.2, '18-30': 4,   '31-45': 9,  '46-60': 7,  'Above 60': 4},
    'Prostate Cancer':                     {'Below 18': 0.05,'18-30': 0.3, '31-45': 2,  '46-60': 9,  'Above 60': 15},
    'Stomach Cancer':                      {'Below 18': 0.3, '18-30': 1,   '31-45': 4,  '46-60': 8,  'Above 60': 10},
}


def _weighted_cancer_type(cancer_pool: list, age_group: str) -> str:
    weights = np.array([CANCER_AGE_WEIGHTS[c][age_group] for c in cancer_pool], dtype=float)
    weights = weights / weights.sum()
    return rng.choice(cancer_pool, p=weights)


def _weighted_med_duration(diagnosis_duration: str) -> str:
    cap_idx = MED_DURATION_CAP_BY_DIAGNOSIS_IDX[_DURATION_INDEX[diagnosis_duration]]
    options = MED_DURATIONS[:cap_idx + 1]
    # Weight toward the longer end of the allowed range — a patient rarely
    # starts and stops medication within days of a long-standing diagnosis.
    weights = np.arange(1, len(options) + 1, dtype=float)
    weights = weights / weights.sum()
    return rng.choice(options, p=weights)

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
    for j, col in enumerate([f'BDI_{i}' for i in range(1, 22)]):
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
    age_group = rng.choice(AGE_GROUPS, p=AGE_PROBS)
    row['Name']        = NAME_PROVIDER.next(gender)
    row['Gender']      = gender
    row['Age Group']   = age_group
    row['Povince']     = rng.choice(PROVINCES,  p=PROV_PROBS)

    # Cancer type is weighted by age group (e.g. blood cancer skews younger,
    # prostate/lung skew older) rather than uniform across the gender pool.
    cancer_pool = FEMALE_CANCERS if gender == 'Female' else MALE_CANCERS
    row['Cancer Type'] = _weighted_cancer_type(cancer_pool, age_group)

    # Clinical status is informed by psychological severity (higher distress
    # skews toward newly-diagnosed/recurrence, lower toward survivorship),
    # then duration and treatment are each conditioned on that status so the
    # three fields tell one coherent clinical story instead of clashing.
    status = rng.choice(STATUSES, p=STATUS_WEIGHTS_BY_SEVERITY[severity])
    duration = rng.choice(DURATIONS, p=DURATION_WEIGHTS_BY_STATUS[status])
    treatment = rng.choice(TREATMENTS, p=TREATMENT_WEIGHTS_BY_STATUS[status])
    row['Current Cancer Status']     = status
    row['Duration Since Diagnosis']  = duration
    row['Current Treatment Type']    = treatment

    # ── Medication (probability correlated with severity) ─────────────────────
    on_meds = rng.random() < MED_PROB_BY_SEV[severity]
    row['Taking Medication'] = 'Yes' if on_meds else 'No'
    if on_meds:
        row['Medication Type'] = rng.choice(MED_TYPES)
        # Capped so a patient diagnosed 3 months ago can't show "More than 1
        # year" on medication.
        row['Medication Duration'] = _weighted_med_duration(duration)
        row['Prescribed by Professional'] = rng.choice(['Yes', 'No'], p=[0.85, 0.15])
        row['Improved Well-being'] = rng.choice(MED_EFFECTS)
    else:
        # When not on medication, these fields are not asked in the form
        # But we'll still include them with empty values for compatibility
        row['Medication Type'] = ''
        row['Medication Duration'] = ''
        row['Prescribed by Professional'] = ''
        row['Improved Well-being'] = ''

    return row


# =============================================================================
# 5. RUN
# =============================================================================

def main():
    print(f"Generating {NUM_RECORDS:,} records (seed={RANDOM_STATE})…")

    # Assign severity classes up-front for exact count control
    severities = rng.choice(SEVERITY_NAMES, size=NUM_RECORDS, p=SEVERITY_PROBS)

    records = []

    for idx, severity in enumerate(severities):
        row = generate_record(severity)
        records.append(row)
        if (idx + 1) % 5_000 == 0:
            print(f"  {idx+1:,} records generated…")

    df = pd.DataFrame(records)

    # ── Reorder columns to match test.csv exactly ─────────────────────────────
    column_order = ['Name', 'Gender', 'Age Group', 'Povince', 'Cancer Type',
                    'Duration Since Diagnosis', 'Current Cancer Status',
                    'Current Treatment Type', 'Taking Medication', 'Medication Type',
                    'Medication Duration', 'Prescribed by Professional', 'Improved Well-being']
    column_order += [f'BDI_{i}' for i in range(1, 22)]
    column_order += [f'FCRI_{i}' for i in range(1, 43)]
    column_order.append('FCRI_Reassured')
    df = df[column_order]

    # ── Uniqueness check ───────────────────────────────────────────────────────
    dupes = len(df) - df['Name'].nunique()
    print(f"\nName uniqueness: {df['Name'].nunique():,}/{len(df):,} unique ({dupes} duplicate{'s' if dupes != 1 else ''})")

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
