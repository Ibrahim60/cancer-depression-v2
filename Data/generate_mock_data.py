"""
Mock Data Generator for Cancer Psychology Assessment
Creates balanced depression severity classes for ML training.
"""

import pandas as pd
import random
import os
from datetime import datetime, timedelta

# Get script directory for proper path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(SCRIPT_DIR, "Psychological Assessment Form for Cancer Patients (BDI & FCRI).csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "cancer_psychology.csv")  # Main data file for ML pipeline
NUM_RECORDS = 20000

print(f"Reading source data from: {SOURCE_FILE}")
df = pd.read_csv(SOURCE_FILE)

# Gender-specific cancer type mapping
FEMALE_ONLY_CANCERS = ["Breast Cancer", "Cervical Cancer", "Ovarian Cancer"]
MALE_ONLY_CANCERS = ["Prostate Cancer"]
GENDER_NEUTRAL_CANCERS = [
    "Blood Cancer (Leukemia / Lymphoma)",
    "Brain", "Brain tumour", "Brain Tumour",
    "Colorectal Cancer", "Liver", "Lung Cancer", "Stomach Cancer"
]

# Get column names
GENDER_COL = "Gender"
CANCER_TYPE_COL = "  Cancer Type  "

# BDI columns (1. through 21.)
BDI_COLUMNS = [f'{i}.' for i in range(1, 22)]

# Build cancer lists per gender from actual data
all_cancers = df[CANCER_TYPE_COL].dropna().unique().tolist()
male_cancers = [c for c in all_cancers if c not in FEMALE_ONLY_CANCERS]
female_cancers = [c for c in all_cancers if c not in MALE_ONLY_CANCERS]

# Get unique BDI responses per item (for controlled sampling)
bdi_responses = {}
for col in BDI_COLUMNS:
    if col in df.columns:
        responses = df[col].dropna().unique().tolist()
        # Sort responses by severity (0-3 based on typical BDI scoring patterns)
        bdi_responses[col] = responses

# Target distribution for depression severity (balanced classes)
# BDI Score ranges: Normal(0-10), Mild(11-16), Borderline(17-20), Moderate(21-30), Severe(31-40), Extreme(41+)
SEVERITY_DISTRIBUTION = {
    'normal': 0.20,           # ~20% no depression (score 0-10)
    'mild': 0.15,             # ~15% mild (score 11-16)  
    'borderline': 0.15,       # ~15% borderline (score 17-20)
    'moderate': 0.30,         # ~30% moderate (score 21-30)
    'severe': 0.15,           # ~15% severe (score 31-40)
    'extreme': 0.05           # ~5% extreme (score 41+)
}

def get_severity_category():
    """Randomly select a severity category based on distribution."""
    r = random.random()
    cumulative = 0
    for category, prob in SEVERITY_DISTRIBUTION.items():
        cumulative += prob
        if r <= cumulative:
            return category
    return 'moderate'

def get_bdi_score_range(severity):
    """Get target BDI score range for a severity level."""
    ranges = {
        'normal': (0, 10),
        'mild': (11, 16),
        'borderline': (17, 20),
        'moderate': (21, 30),
        'severe': (31, 40),
        'extreme': (41, 50)
    }
    return ranges.get(severity, (21, 30))

def categorize_bdi_response(response_text):
    """
    Categorize a BDI response text into severity score (0-3).
    Uses keyword matching similar to the ML pipeline.
    """
    if pd.isna(response_text):
        return None
    text = str(response_text).lower().strip()
    
    # Score 3 indicators (most severe)
    if any(kw in text for kw in ['so sad', 'unhappy that i can', 'hopeless', 'complete failure',
                                  'dissatisfied or bored', 'guilty all of the time', 'i am being punished',
                                  'hate myself', 'blame myself for everything', 'would kill myself if',
                                  "can't cry", 'irritated all the time', 'lost all of my interest',
                                  "can't make decisions", 'look ugly', "can't do any work", 
                                  'several hours earlier', 'too tired to do anything', 'no appetite at all',
                                  'lost more than fifteen', 'lost interest in sex completely']):
        return 3
    
    # Score 2 indicators
    if any(kw in text for kw in ['sad all the time', 'nothing to look forward', 'lot of failures',
                                  "don't get real satisfaction", 'quite guilty', 'expect to be punished',
                                  'disgusted with myself', 'blame myself all the time', 'would like to kill',
                                  'cry all the time', 'quite annoyed', 'lost most of my interest',
                                  'greater difficulty', 'permanent changes', 'push myself very hard',
                                  '1-2 hours earlier', 'tired from doing almost', 'much worse now',
                                  'lost more than ten', 'very worried', 'almost no interest in sex']):
        return 2
    
    # Score 1 indicators
    if any(kw in text for kw in ['i feel sad', 'feel discouraged', 'failed more than', "don't enjoy",
                                  'feel guilty a good part', 'may be punished', 'disappointed in myself',
                                  'critical of myself', 'thoughts of killing myself, but', 'cry more now',
                                  'slightly more irritated', 'less interested in other people',
                                  'put off making decisions', 'worried that i am looking', 'extra effort',
                                  "don't sleep as well", 'get tired more easily', 'not as good as it used to',
                                  'lost more than five', 'worried about physical', 'less interested in sex than']):
        return 1
    
    # Score 0 (least severe / normal)
    return 0

def build_bdi_response_map(source_df, bdi_cols):
    """
    Build a mapping of BDI responses categorized by score level.
    Returns: {col: {0: [responses], 1: [responses], 2: [responses], 3: [responses]}}
    """
    response_map = {}
    for col in bdi_cols:
        if col not in source_df.columns:
            continue
        response_map[col] = {0: [], 1: [], 2: [], 3: []}
        for resp in source_df[col].dropna().unique():
            score = categorize_bdi_response(resp)
            if score is not None:
                response_map[col][score].append(resp)
    return response_map

def sample_bdi_responses_for_target(target_score, bdi_cols, response_map):
    """
    Sample BDI responses to approximate a target total score.
    Uses direct score-level selection based on target.
    """
    responses = {}
    n_items = len([c for c in bdi_cols if c in response_map])
    if n_items == 0:
        return responses
    
    target_avg = target_score / n_items
    
    for col in bdi_cols:
        if col not in response_map:
            continue
        
        col_map = response_map[col]
        
        # Determine which score level to target
        if target_avg < 0.5:
            # Mostly 0s with some 1s
            probs = [0.7, 0.25, 0.04, 0.01]
        elif target_avg < 1.0:
            # Mix of 0s and 1s
            probs = [0.4, 0.45, 0.12, 0.03]
        elif target_avg < 1.5:
            # Mix of 1s and 2s
            probs = [0.15, 0.45, 0.30, 0.10]
        elif target_avg < 2.0:
            # Mix of 1s, 2s with some 3s
            probs = [0.05, 0.30, 0.45, 0.20]
        elif target_avg < 2.5:
            # Mix of 2s and 3s
            probs = [0.02, 0.15, 0.43, 0.40]
        else:
            # Mostly 3s with some 2s
            probs = [0.01, 0.05, 0.24, 0.70]
        
        # Select a score level
        score_level = random.choices([0, 1, 2, 3], weights=probs, k=1)[0]
        
        # Get response from that level, fallback to adjacent levels if empty
        available = col_map.get(score_level, [])
        if not available:
            # Try adjacent levels
            for alt in [score_level - 1, score_level + 1, 0, 1, 2, 3]:
                if 0 <= alt <= 3 and col_map.get(alt):
                    available = col_map[alt]
                    break
        
        if available:
            responses[col] = random.choice(available)
    
    return responses

# Pre-build BDI response map for efficient sampling
bdi_cols_in_df = [c for c in BDI_COLUMNS if c in df.columns]
bdi_response_map = build_bdi_response_map(df, bdi_cols_in_df)

# Debug: show available responses per score level for first BDI item
print("BDI Response Map for Item 1:")
for score, resps in bdi_response_map.get('1.', {}).items():
    print(f"  Score {score}: {len(resps)} responses")

rows = []

start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 6, 1)

print(f"\nGenerating {NUM_RECORDS:,} records with balanced depression severity...")

for i in range(NUM_RECORDS):
    row = {}

    # Generate realistic timestamp
    delta = end_date - start_date
    ts = start_date + timedelta(
        seconds=random.randint(0, int(delta.total_seconds()))
    )
    row["Timestamp"] = ts.strftime("%d/%m/%Y %H:%M:%S")

    # Generate name
    row["Name"] = f"Patient_{i+1:05d}"

    # Assign gender first
    gender_values = df[GENDER_COL].dropna().tolist()
    row[GENDER_COL] = random.choice(gender_values)

    # Assign cancer type based on gender
    if row[GENDER_COL] == "Male":
        row[CANCER_TYPE_COL] = random.choice(male_cancers)
    else:
        row[CANCER_TYPE_COL] = random.choice(female_cancers)

    # Determine target depression severity and score
    severity = get_severity_category()
    score_min, score_max = get_bdi_score_range(severity)
    target_score = random.randint(score_min, score_max)
    
    # Sample BDI responses to approximate target score
    bdi_row = sample_bdi_responses_for_target(target_score, bdi_cols_in_df, bdi_response_map)
    row.update(bdi_row)

    # Sample all remaining columns (non-BDI) from existing values
    for col in df.columns:
        if col in ["Timestamp", "Name", GENDER_COL, CANCER_TYPE_COL] or col in BDI_COLUMNS:
            continue

        non_null_values = df[col].dropna().tolist()

        if len(non_null_values) == 0:
            row[col] = ""
        else:
            row[col] = random.choice(non_null_values)

    rows.append(row)
    
    if (i + 1) % 5000 == 0:
        print(f"  Generated {i+1:,} records...")

dummy_df = pd.DataFrame(rows)

# Optional logical consistency
med_col = "Are you currently taking any medication for depression, anxiety, stress, or other emotional/psychological problems?  "
med_type_col = "If yes, what type of medication are you taking?  "
med_duration_col = "How long have you been taking this medication?  "
med_prescribed_col = "Was this medication prescribed by a healthcare professional? "
med_effect_col = "Do you feel the medication has improved your emotional well-being? "

for idx in dummy_df.index:
    if dummy_df.loc[idx, med_col] == "No":
        for c in [med_type_col, med_duration_col, med_prescribed_col, med_effect_col]:
            if c in dummy_df.columns:
                dummy_df.loc[idx, c] = ""

dummy_df.to_csv(OUTPUT_FILE, index=False)

print(f"Generated {len(dummy_df):,} records -> {OUTPUT_FILE}")
