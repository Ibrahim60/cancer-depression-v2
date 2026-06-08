
import pandas as pd
import random
from datetime import datetime, timedelta

SOURCE_FILE = "Psychological Assessment Form for Cancer Patients (BDI & FCRI).csv"
OUTPUT_FILE = "dummy_cancer_psychology_20000.csv"
NUM_RECORDS = 20000

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

# Build cancer lists per gender from actual data
all_cancers = df[CANCER_TYPE_COL].dropna().unique().tolist()
male_cancers = [c for c in all_cancers if c not in FEMALE_ONLY_CANCERS]
female_cancers = [c for c in all_cancers if c not in MALE_ONLY_CANCERS]

def weighted_choice(values):
    values = [v for v in values if pd.notna(v)]
    return random.choice(list(values))

rows = []

start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 6, 1)

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

    # Sample all remaining columns from existing values
    for col in df.columns:
        if col in ["Timestamp", "Name", GENDER_COL, CANCER_TYPE_COL]:
            continue

        non_null_values = df[col].dropna().tolist()

        if len(non_null_values) == 0:
            row[col] = ""
        else:
            row[col] = random.choice(non_null_values)

    rows.append(row)

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
