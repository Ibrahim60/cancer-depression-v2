# Methodology

## Hybrid Machine Learning Framework for Depression Screening in Cancer Patients

This document describes the data, preprocessing, feature engineering, model
architecture, training protocol, evaluation methodology, and the inference and
decision logic used by the depression screening system in this repository. It
is written to be used directly as (or adapted into) the Methodology chapter of
a thesis or research paper. All figures reported here were produced by
`depression_detection_pipeline.py` against `Data/cancer_psychology.csv` and are
reproducible by re-running that script (fixed random seed,
`RANDOM_STATE = 42`, throughout). The figures in §12 were re-verified against a
fresh training run of the current code.

---

## 1. Objective

To develop a screening instrument that estimates the presence and severity of
depressive symptomatology in cancer patients, using two validated
psychometric instruments — the **Beck Depression Inventory (BDI-II)** and the
**Fear of Cancer Recurrence Inventory (FCRI)** — combined with patient
demographic and clinical-context variables, as input to a supervised machine
learning framework. The system supports two use modes: a **Full Assessment**
(all instrument items) intended for first-time or clinically-validated
screening, and a **Quick Assessment** (a reduced, ML-selected item subset)
intended to lower respondent burden for repeat or triage screening.

The system's final output answers one question — *does this patient show
cancer-related depression?* — by combining the ML model's depression
prediction with the FCRI fear-of-recurrence score (§13.4).

It is important to state at the outset that this is a **screening aid, not a
diagnostic instrument** — a distinction preserved throughout the application
(see §16, Limitations, and the disclaimers surfaced in the tool itself).

---

## 2. End-to-End Workflow

The system has two phases that share one set of persisted artifacts in
`models/`: an **offline training phase**, run once per dataset version, and an
**online screening phase**, run once per patient.

```
 TRAINING (offline)  — python depression_detection_pipeline.py
 ─────────────────────────────────────────────────────────────────────────
 Data/cancer_psychology.csv  (1,532 rows × 78 columns)
   │
   ├─ 1. Preprocess (§4)    de-identify → text-to-score → drop all-zero FCRI
   │                        → reverse FCRI 13 → drop >50%-missing rows
   │                        → within-patient imputation → label-encode demographics
   ├─ 2. Targets (§5)       bdi_total = Σ 21 BDI items
   │                        → has_depression (≥11)  and  6-level severity
   ├─ 3. Features (§6)      21 BDI + 42 FCRI + 8 demographic = 71 columns
   ├─ 4. Split & scale      one stratified 80/20 split, one StandardScaler
   ├─ 5. Train (§7–§8)      LR, RF, XGBoost tuned by 5-fold CV (macro-F1)
   │                        → soft-voting ensemble, per task (binary, multiclass)
   ├─ 6. Threshold (§9)     out-of-fold search for the binary cut-off (0.60)
   ├─ 7. Importance (§10)   mean of normalised RF + XGBoost importances
   ├─ 8. Evaluate (§10)     held-out test set (307 patients)
   └─ 9. Export             models/*.joblib, feature_metadata.json,
                            feature_importance.csv, confusion matrices
                                   │
                                   ▼
 SCREENING (online)  — streamlit run streamlit_app.py   or   python predict_depression.py
 ─────────────────────────────────────────────────────────────────────────
   1. Load artifacts       ensembles, scaler, label encoders, feature metadata
   2. Choose mode (§14)    Quick (top-10 BDI + top-5 FCRI from feature_importance.csv)
                           or Full (all 21 BDI + all 42 FCRI)
   3. Collect              5 demographic fields → BDI items → FCRI items
   4. Feature vector (§13.1) answered items + training means for unasked items
                           + encoded demographics, in training column order
   5. Predict (§13.2)      scale → binary ensemble (P(depression) vs 0.60)
                           → multiclass ensemble (6 severity probabilities)
   6. Reference scores (§13.3) rule-based BDI total and FCRI subscale/severity scores
   7. Final verdict (§13.4) model severity × FCRI severity → YES / POSSIBLE / NO
   8. Report (§14.4)       Step 1 BDI → Step 2 FCRI → Step 3 model + verdict;
                           safety messages; CSV/PDF export; session history
```

---

## 3. Data Source and Instruments

### 3.1 Sample

The dataset (`Data/cancer_psychology.csv`) contains **1,532 self-reported
responses** collected from cancer patients, comprising 78 raw columns:
identifying/administrative fields (timestamp, name — removed prior to
analysis, see §4.1), demographic and clinical-context fields, and the two
psychometric instruments in full.

### 3.2 Beck Depression Inventory (BDI-II)

A 21-item self-report inventory, each item scored 0–3 (four graded statements
of increasing symptom severity), covering affective, cognitive, somatic, and
vegetative symptoms of depression (sadness, pessimism, past failure, loss of
pleasure, guilt, punishment feelings, self-dislike, self-criticism, suicidal
ideation, crying, agitation, loss of interest, indecisiveness, body image,
work inhibition, sleep disturbance, fatigability, appetite change, weight
loss, somatic preoccupation, loss of interest in sex). The instrument yields
a total score range of 0–63. Respondents rate the past week, including today.

### 3.3 Fear of Cancer Recurrence Inventory (FCRI)

A 42-item self-report inventory organized into **seven subscales**, each
covering a distinct construct related to fear of cancer recurrence (FCR).
Respondents rate the past month.

| Subscale                  | Items | Response scale (verified against raw data, §4.2) |
|----------------------------|-------|----------------------------------------------------|
| Triggers                   | 1–8   | Frequency: Never(0) → All the time(4)               |
| Severity                   | 9–17  | Mixed — see below                                   |
| Psychological Distress     | 18–21 | Intensity: Not at all(0) → A great deal(4)          |
| Functioning Impairments    | 22–27 | Intensity: Not at all(0) → A great deal(4)          |
| Insight                    | 28–30 | Intensity: Not at all(0) → A great deal(4)          |
| Reassurance                | 31–33 | Frequency: Never(0) → All the time(4)               |
| Coping Strategies          | 34–42 | Frequency: Never(0) → All the time(4)               |

The Severity subscale (items 9–17, range 0–36) mostly uses the Intensity
scale, except: item 13 ("I believe that I am cured...") is intensity-scaled
but **reverse-coded** at analysis time (agreement indicates *lower* fear; see
§4.4); item 14 uses a dedicated risk-perception scale (Not at all at risk(0)
→ A great deal at risk(4)); item 15 uses a thought-frequency scale (Never(0)
→ Several times a day(4)); item 16 uses a time-per-day scale ("I don't think
about it"(0) → Several hours(4)); item 17 uses a duration scale ("I don't
think about it"(0) → Several years(4)).

### 3.4 Demographic and Clinical-Context Variables

Eight fields are present in the current dataset and used as model features:
Gender, Age Group, Province, Cancer Type, Current Treatment Type, Current
Cancer Status, whether the patient is currently on medication for
depression/anxiety, and Duration Since Diagnosis. Six additional fields the
preprocessing pipeline is designed to ingest (Cancer Stage, Education Level,
Marital Status, Employment Status, Prior Psychiatric History, Collection
Site) are **not present in the current dataset** and are consequently
excluded from the trained feature set (see §16.3).

---

## 4. Data Preprocessing

Implemented in `ClinicalDataPreprocessor.fit_transform()`
(`depression_detection_pipeline.py`). Steps are applied in the following
fixed order; the order matters (see §4.3).

| Step | Operation | Effect on current data |
|---|---|---|
| 4.1 | Drop identifying fields | `Name`, `Timestamp` removed |
| 4.2 | Convert response text to numeric scores | BDI 100.0%, FCRI 99.8% converted |
| 4.3 | Drop mechanical (all-zero FCRI) responses | 0 rows dropped |
| 4.4 | Reverse-score FCRI item 13 | — |
| 4.5 | Drop rows with >50% missing items | 1 row dropped (1,532 → 1,531) |
| 4.6 | Within-patient imputation | 0 rows dropped by the sparse-subscale rule |
| 4.7 | Label-encode demographics | 8 encoded columns |

### 4.1 De-identification

`Name` and `Timestamp` (and, where present, `Patient Code / ID` /
`Patient_ID`) are dropped before any analysis is performed.

### 4.2 Response Normalization

Instrument responses were free-text/Likert-label strings as exported (e.g.
`"I feel sad"`, `"Somewhat"`), rather than pre-coded integers. Columns are
identified by name (BDI `1.`–`21.` or `BDI_1`–`BDI_21`; FCRI `N. <item text>`
or `FCRI_N`), and each response is converted to its numeric score via, in
order of precedence: (1) direct numeric parse, for already-coded values; (2)
a digit-prefix regular expression (`"2 - Somewhat"` → 2), for digit-prefixed
exports; (3) explicit keyword/phrase matching against the instrument's known
response text. BDI keyword matching is ordered from most-severe to
least-severe phrase and defaults to 0 if no phrase matches; FCRI matching
uses a fixed text-to-integer lookup table per response scale (§3.3) and
returns a missing value if no match is found (handled by §4.6).

A conversion-rate diagnostic is computed and logged for both instruments to
surface silent parsing failures (with a warning below 80%). On the current
dataset: **BDI conversion rate 100.0%, FCRI conversion rate 99.8%** (see §4.8
for the specific defect that was found and corrected to reach this rate).

### 4.3 Quality Filter: Mechanical Responses

Rows where **all** 42 FCRI responses equal 0 are dropped as likely
non-engaged/mechanical responses. This filter is deliberately applied to the
*raw* scores, **before** item 13 is reversed: reversing first would turn an
all-zero respondent's item 13 into 4, so they would never match "all FCRI = 0"
and would slip through. On the current dataset this filter removed no rows.

### 4.4 FCRI Item 13 Reversal

FCRI item 13 ("I believe that I am cured and that the cancer will not come
back") is worded in the opposite direction to the rest of the Severity
subscale — agreement indicates *lower*, not higher, fear of recurrence. It is
reverse-scored as `reversed = 4 - raw` before being used in any subscale
aggregate or as a model feature. The same reversal is applied to a patient's
live answer at inference time (§13.1, §13.3).

### 4.5 Quality Filter: Excessive Missingness

Rows with more than 50% missing values across the combined BDI+FCRI item set
are dropped as insufficiently complete. On the current dataset this removed
1 row (1,532 → 1,531).

### 4.6 Missing-Value Imputation

Remaining missing item-level values are imputed using **within-patient,
within-subscale means**, rounded to the nearest integer score, rather than a
global column mean or a fixed default: a missing BDI item is imputed from
that same respondent's mean across their answered BDI items; a missing FCRI
item is imputed from that respondent's mean across the answered items in the
*same FCRI subscale* (§3.3). This preserves each patient's individual
response tendency rather than regressing them toward the sample average.

Following the FCRI scoring procedure, imputation is not considered valid when
fewer than half of a subscale's items were answered; such respondents are
excluded rather than imputed. On the current dataset no rows met this
condition.

### 4.7 Categorical Encoding

Each demographic/clinical-context field (§3.4) is treated as a categorical
variable: missing values are filled with the literal category `"Unknown"`,
and each field is integer-encoded via `sklearn.preprocessing.LabelEncoder`
(categories sorted alphabetically). The fitted encoders are persisted
(`label_encoders.joblib`) so that new patient responses at inference time are
encoded identically; a category value not seen during training falls back to
code 0 rather than raising an error (see §16.6 for the consequence of this
in the current application).

### 4.8 Data-Quality Defect Found and Corrected

During verification of the FCRI response-to-score mapping against the raw
CSV, two items (16 and 17) were found to share the response option "I don't
think about it" (intended score 0), but the CSV export encodes this phrase
with a Unicode right single quotation mark (`’`, U+2019) rather than the
ASCII apostrophe (`'`) the original scoring table matched against. Because
string comparison in the scoring lookup is exact, this caused **161/1,531
(10.5%) of item 16 responses and 122/1,531 (8.0%) of item 17 responses** to
be silently treated as missing and subscale-mean-imputed, rather than
correctly scored as 0. The scoring table was corrected to match the
Unicode apostrophe (retaining the ASCII form as well, for robustness against
other export sources), which is what raised the FCRI conversion rate from
99.4% to 99.8% (§4.2) and shifted the per-feature training means for items
16 and 17 accordingly (from 1.372 to 1.951, and 1.523 to 2.029,
respectively).

Earlier corrections to the same table (recorded in the pipeline header) fixed
the item 15 frequency mapping (e.g. "A few times a month" → 1, not 2) and the
item 17 top category ("Several years" → 4).

### 4.9 BDI Items 6 and 7 Scoring Defect Found and Corrected

A cross-check of every BDI response option against the published BDI scoring
(found 2026-10-18, corrected 2026-10-26) showed that the keyword matcher
(§4.2) scored the score-0 statements of two items as symptomatic. The
keywords `'i am being punished'` (item 6, score 3) and
`'disappointed in myself'` (item 7, score 1) also occur inside the score-0
statements "I don't feel I am being punished" and "I don't feel disappointed
in myself", and the matcher accepts the first keyword found anywhere in the
answer. On the current dataset this mis-scored **374 item-6 answers (0 → 3)
and 343 item-7 answers (0 → 1)**, inflated the BDI total of 516 patients
(34%) by 1–4 points, and placed 185 patients (12%) one severity band too high
(none too low); 33 patients with BDI < 11 were labelled as depressed.

The keywords were narrowed to `'i feel i am being punished'` and
`'i am disappointed in myself'`. After the fix all 21 × 4 response options
map to their published scores (0 mismatches over 32,172 answers), and the
models were retrained. The figures in §5, §6, §8, §9 and §12 are from the
retrained models; the pre-fix run gave a BDI total mean of 30.7 (SD 12.1),
86 "Normal" patients, binary ensemble test accuracy 0.9935 and multiclass
ensemble test accuracy 0.8730.

---

## 5. Target Variable Definition

Implemented in `TargetVariableEngineer`. Two supervised targets are derived,
both purely as a function of the BDI-II total score (`bdi_total`, the
row-wise sum of the 21 BDI items — never the FCRI or demographic fields):

**Binary target** (`has_depression`): 1 if `bdi_total >= 11`, else 0. This
separates "no depression" (Normal) from "mood disturbance at any severity,"
so the positive class includes the sub-clinical Mild band (11–16).

**Multiclass target** (`depression_severity`, 6 ordinal classes), using the
standard BDI-II severity bands:

| Severity                          | BDI-II score range |
|------------------------------------|---------------------|
| Normal                              | 0–10                |
| Mild mood disturbance                | 11–16               |
| Borderline clinical depression       | 17–20               |
| Moderate depression                  | 21–30               |
| Severe depression                    | 31–40               |
| Extreme depression                   | 41–63               |

The severity labels are integer-encoded with `LabelEncoder`, which orders
them **alphabetically** (Borderline, Extreme, Mild, Moderate, Normal, Severe),
not by severity. Every consumer of the multiclass model maps class ids back
to labels through the encoder before display (§13.2); this mapping was the
source of an earlier label bug and is the reason per-class results in §12.2
are reported by name.

On the cleaned dataset (N = 1,531; BDI total mean 29.7, SD 13.0, range
0–63): binary class counts are 1,412 positive (92.2%) vs. 119 negative (7.8%)
— a substantial class imbalance addressed explicitly in training (§8).
Multiclass counts: Moderate 425, Extreme 399, Severe 317, Borderline 138,
Mild 133, Normal 119.

**Note on target construction (important methodological caveat):** because
both targets are deterministic thresholded functions of the same 21 BDI item
scores that are also supplied as model features, a model given complete BDI
responses (as in Full Assessment mode) has direct arithmetic access to the
label. The genuine machine-learning estimation problem this framework solves
is (a) predicting the label from FCRI and demographic features *in
combination with* incomplete BDI data (Quick Assessment mode, §14.1), and
(b) producing class probabilities rather than a rigid rule-based cutoff.
This is discussed further in §16.1.

---

## 6. Feature Representation

The final feature matrix combines all three sources into one vector per
patient:

| Feature group | Count | Source |
|---|---|---|
| BDI items | 21 | §3.2, integer 0–3 each |
| FCRI items | 42 | §3.3, integer 0–4 each (item 13 reverse-scored) |
| Demographic (label-encoded) | 8 | §3.4 |
| **Total** | **71** | |

Any residual missing value in the matrix is set to 0. The exact feature
ordering and each feature's training-set mean (used for imputing unasked
items at Quick-mode inference time, §13.1) are persisted to
`models/feature_metadata.json`, generated fresh on every training run so
that the deployed inference code and the trained model are always schema
matched.

On the current run: **feature matrix shape (1,531, 71)**; stratified 80/20
train/test split (§7.1) yields **1,224 training rows, 307 test rows**.

---

## 7. Model Architecture

Implemented in `DepressionClassificationPipeline`.

### 7.1 Train/Test Split and Feature Scaling

A single stratified 80/20 split (`random_state = 42`, stratified on the
binary label) is used, and the **same split and the same fitted
`StandardScaler`** (fit on the training partition only) are reused for both
the binary and multiclass tasks. This is a deliberate design choice: fitting
independent splits or independent scalers per task would make binary and
multiclass results non-comparable and risks a scaler being silently refit
and overwritten. The scaler is saved as both `binary_scaler.joblib` and
`multiclass_scaler.joblib` (identical objects); the applications load the
former.

### 7.2 Base Learners

Three heterogeneous classifiers are trained **independently per task**
(binary, multiclass) — i.e., six models total, plus two ensembles (§7.4):

1. **Logistic Regression** (`sklearn.linear_model.LogisticRegression`,
   `solver='lbfgs'`, `max_iter=1000`)
2. **Random Forest** (`sklearn.ensemble.RandomForestClassifier`)
3. **XGBoost** (`xgboost.XGBClassifier`; `binary:logistic`/`logloss` for the
   binary task, `multi:softprob`/`mlogloss` for the multiclass task —
   `softprob` rather than `softmax` is required so per-class probabilities
   are available for soft voting, §7.4)

### 7.3 Hyperparameter Tuning

Each base learner is tuned via `GridSearchCV` with **5-fold stratified
cross-validation** (`StratifiedKFold`, `shuffle=True`, `random_state=42`),
scored on **macro-averaged F1** (`f1_macro`). Macro-F1 (the unweighted mean
of per-class F1) was chosen over the sample-weighted alternative
(`f1_weighted`) specifically because the latter re-biases *model selection*
back toward the majority class in proportion to its support, which would
partially undo the class-imbalance handling described in §8 — macro-F1
gives the minority class equal weight in the selection criterion, consistent
with the framework's goal of not missing genuinely non-depressed patients.

| Model | Grid searched |
|---|---|
| Logistic Regression | `C ∈ {0.01, 0.1, 1, 10}`, `penalty = l2` |
| Random Forest | `n_estimators ∈ {100, 200}`, `max_depth ∈ {5, 10, None}`, `min_samples_split ∈ {2, 5}` |
| XGBoost | `n_estimators ∈ {100, 200}`, `max_depth ∈ {3, 5, 7}`, `learning_rate ∈ {0.01, 0.1}` |

### 7.4 Ensemble

A **soft-voting ensemble** (`sklearn.ensemble.VotingClassifier`,
`voting='soft'`) combines the three tuned base learners by averaging their
predicted class probabilities. The ensembles are the models the applications
use for every prediction. Binary and multiclass ensembles are weighted
differently, based on empirical comparison of the three base learners:

- **Binary task**: equal weights (1, 1, 1). All three base learners perform
  comparably well on this task (§12.1), so equal weighting is appropriate.
- **Multiclass task**: weights (2, 1, 1) favoring Logistic Regression. In an
  earlier verification run, an equal-weight ensemble under-performed
  standalone Logistic Regression (86.0% vs. 87.6% accuracy), because Random
  Forest and XGBoost are substantially weaker on this harder 6-class task
  (§12.2) and were dragging the ensemble average down. Re-weighting the vote
  toward the strongest individual learner closed this gap: on the current
  run the weighted ensemble reaches 88.6% accuracy vs. 87.0% for standalone
  Logistic Regression.

This is reported as a methodological decision rather than a default,
because it materially changed which model would otherwise have been the
best-performing option — a finding worth stating explicitly rather than
presenting the three-model ensemble as uniformly superior by construction.

---

## 8. Class Imbalance Handling

The binary target is markedly imbalanced (94.4% positive / 5.6% negative,
§5), and the multiclass "Normal" and "Mild" categories are similarly
under-represented (5.6% and 7.4% of the sample, respectively). Three
complementary techniques are used, deliberately **not** including synthetic
oversampling (e.g. SMOTE):

1. **Class-weighted loss** (`class_weight='balanced'`) for Logistic
   Regression and Random Forest, which re-weights the loss function
   inversely proportional to class frequency.
2. **Sample-weighted loss** for XGBoost, which has no native `class_weight`
   parameter: inverse-frequency weights are computed via
   `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` and
   passed as `sample_weight` at `.fit()` time during tuning, so all three
   base learners receive consistent imbalance treatment (in an earlier
   iteration of this pipeline, only LR and RF were balanced and XGBoost
   received none).
3. **Decision threshold calibration** (§9), rather than resampling, to
   further tune the binary classifier's operating point.

Synthetic oversampling was deliberately avoided given (a) the small absolute
minority-class count (119 patients), which limits the diversity of
interpolated synthetic samples and risks tight clustering around a few real
points, and (b) all features being ordinal (BDI 0–3, FCRI 0–4) or categorical
— SMOTE-style interpolation can generate non-integer "phantom" item
responses that do not correspond to any real questionnaire answer, which
would compromise the clinical interpretability of what the model has learned
from.

---

## 9. Decision Threshold Calibration (Binary Task)

The binary classifier's default operating point (predict positive if
P(depression) ≥ 0.5) is not necessarily optimal under class imbalance. A
threshold search is performed as follows: **out-of-fold predicted
probabilities are obtained on the training set only**, via
`sklearn.model_selection.cross_val_predict` with the same 5-fold stratified
CV split used elsewhere; macro-F1 is computed at 91 candidate thresholds
evenly spaced across [0.05, 0.95]; the threshold maximizing macro-F1 is
selected. Critically, **the test set is never used to select the
threshold** — it is used only to *report* the effect of the already-chosen
threshold, preserving a clean train/test separation. The selected threshold
is persisted (`feature_metadata.json → binary_threshold`) and applied at
inference time in place of the sklearn default (falling back to 0.5 if an
older metadata file lacks it).

On the current dataset, the tuned threshold was **0.60** (CV macro-F1 0.949
vs. 0.948 at the default 0.50). On the held-out test set it trades precision
for recall in the "No Depression" class: recall rises from 0.88 to 0.96
(21 → 23 of 24 patients) and precision falls from 1.00 to 0.85 (§12.1). The
CV gain over 0.50 is small (0.001), so the choice between the two thresholds
is close to a tie on this data.

---

## 10. Evaluation Protocol

For each of the 4 models per task (3 base learners + ensemble), the
following are computed on the held-out test set (307 patients, never used in
training or hyperparameter/threshold selection):

- Accuracy, weighted-average precision/recall/F1
- Per-class precision/recall/F1 (`classification_report`)
- ROC-AUC — standard formulation for the binary task; macro-averaged
  one-vs-rest for the multiclass task
- Confusion matrices (one panel per model, saved as
  `models/binary_confusion.png` and `models/multiclass_confusion.png`)

Feature importance is computed by normalizing the Random Forest's and
XGBoost's `feature_importances_` (from the *binary* task models) to sum to
1 each, then averaging the two normalized vectors per feature. This combined
importance ranking drives the Quick Assessment item-selection logic (§14.1)
and is persisted to `models/feature_importance.csv`.

---

## 11. Reproducibility

- Fixed random seed (`RANDOM_STATE = 42`) applied to the train/test split,
  all three base learners, and all cross-validation folds. Re-running the
  pipeline reproduces the figures in §12 exactly.
- Package versions are pinned to minimum bounds in `requirements.txt`
  (numpy, pandas, scikit-learn, xgboost, matplotlib, joblib, streamlit,
  reportlab).
- Every training run regenerates `feature_metadata.json` (feature order,
  per-feature training means, tuned binary threshold) alongside the model
  artifacts, so inference code (`predict_depression.py`, `streamlit_app.py`)
  is guaranteed schema-consistent with whichever model artifacts are
  currently deployed.
- The full pipeline is a single command: `python depression_detection_pipeline.py`,
  reading `Data/cancer_psychology.csv` and writing all artifacts to `models/`.

---

## 12. Results

Figures below are from the training run of the current code, incorporating
all corrections described in §4.8, §4.9, §7.4, §8, and §9 (the current deployed
model artifacts).

### 12.1 Binary Task (Has Depression: Yes/No)

Cross-validated hyperparameter selection (macro-F1):

| Model | Best hyperparameters | CV macro-F1 |
|---|---|---|
| Logistic Regression | `C=1` | 0.9437 |
| Random Forest | `max_depth=None, min_samples_split=5, n_estimators=200` | 0.9243 |
| XGBoost | `learning_rate=0.1, max_depth=3, n_estimators=100` | 0.9625 |

Held-out test-set performance (default 0.5 threshold):

| Model | Accuracy | F1 (weighted) | ROC-AUC | "No Depression" P / R / F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9837 | 0.9839 | 0.9971 | 0.88 / 0.92 / 0.90 |
| Random Forest | 0.9870 | 0.9870 | 0.9981 | 0.92 / 0.92 / 0.92 |
| XGBoost | 0.9674 | 0.9695 | 0.9956 | 0.72 / 0.96 / 0.82 |
| **Voting Ensemble** | **0.9902** | **0.9899** | **0.9987** | **1.00 / 0.88 / 0.93** |

At the CV-tuned threshold (0.60) applied to the test set, the ensemble
reaches **98% accuracy** (rounded), with "No Depression" precision/recall/F1
of **0.85 / 0.96 / 0.90**.

*Interpretation caveat:* the "No Depression" class has only 24 test-set
members; a recall of 0.88 is 21 of 24 and 0.96 is 23 of 24, so
differences of about four percentage points in its metrics correspond to one
patient. They should be
read as indicative rather than statistically precise given this sample size.

### 12.2 Multiclass Task (6-Level Severity)

Cross-validated hyperparameter selection (macro-F1):

| Model | Best hyperparameters | CV macro-F1 |
|---|---|---|
| Logistic Regression | `C=10` | 0.7822 |
| Random Forest | `max_depth=None, min_samples_split=2, n_estimators=200` | 0.6856 |
| XGBoost | `learning_rate=0.1, max_depth=3, n_estimators=200` | 0.6551 |

Held-out test-set performance:

| Model | Accuracy | F1 (weighted) | ROC-AUC (macro OvR) |
|---|---|---|---|
| Logistic Regression | 0.8697 | 0.8706 | 0.9873 |
| Random Forest | 0.7199 | 0.7123 | 0.9386 |
| XGBoost | 0.7036 | 0.7060 | 0.9264 |
| **Voting Ensemble (LR-weighted)** | **0.8860** | **0.8860** | **0.9835** |

Per-class F1 on the test set, all models (classes in severity order):

| Severity class | n (test) | Logistic Regression | Random Forest | XGBoost | Voting Ensemble (P / R / F1) |
|---|---|---|---|---|---|
| Normal | 24 | 0.83 | 0.84 | 0.83 | 0.91 / 0.88 / **0.89** |
| Mild mood disturbance | 27 | 0.75 | 0.56 | 0.52 | 0.84 / 0.78 / **0.81** |
| Borderline clinical depression | 30 | 0.73 | 0.28 | 0.34 | 0.73 / 0.80 / **0.76** |
| Moderate depression | 87 | 0.89 | 0.78 | 0.73 | 0.90 / 0.91 / **0.90** |
| Severe depression | 59 | 0.88 | 0.66 | 0.67 | 0.89 / 0.85 / **0.87** |
| Extreme depression | 80 | 0.95 | 0.86 | 0.87 | 0.94 / 0.96 / **0.95** |

The **"Borderline clinical depression" class (BDI 17–20) is the hardest to
classify** for every model (F1 0.28–0.76). It is the narrowest severity band
(four score points) and sits between two neighbouring classes, so a one- or
two-point disagreement in a few items moves a patient across a boundary. The
"Normal" class, despite being the smallest, is classified well (ensemble F1
0.89); like Extreme, it is an end band with only one neighbouring class to
be confused with. This is discussed further in §16.2.

*Correction note:* an earlier version of this document listed per-class
figures in severity order against the encoder's alphabetical class order
(§5), which mislabeled every row and led to the incorrect conclusion that
"Normal" was the weakest class. The table above is taken from the
`classification_report` output, where labels are resolved by name.

### 12.3 Feature Importance

Top BDI items by combined Random Forest + XGBoost importance (binary task),
in rank order — the first 10 form the Quick Assessment BDI item set (§14.1):
item 6 (Punishment Feelings), item 5 (Guilty Feelings), item 2
(Pessimism), item 16 (Sleep Disturbance), item 15 (Work Inhibition), item 3
(Past Failure), item 18 (Appetite Changes), item 4 (Loss of Pleasure), item
14 (Body Image), item 11 (Agitation).

Item 6 ranks first with a combined importance of 0.197, against 0.075 for
item 5. Most of this comes from XGBoost (0.285 vs. 0.110 in Random Forest);
gradient boosting tends to concentrate importance on one of several
correlated predictors. Item 6's correlation with the binary label (0.36) is
in line with the other top items (items 5 and 18: 0.37 and 0.36). Before the
§4.9 fix, 374 item-6 "no symptom" answers were scored 3, which made the item
noisy and kept it out of the top 10.

Top 5 FCRI items — the Quick Assessment FCRI item set: item 14 (Perceived
risk of recurrence), item 25 (Ability to make future plans), item 15
(Frequency of recurrence thoughts), item 9 (Worry about recurrence), item 16
(Time spent thinking about recurrence). Four belong to the Severity subscale;
item 25 belongs to the Functioning impairments subscale.

These rankings are regenerated on every training run, so the specific items
may shift as the underlying dataset grows or is corrected.

---

## 13. Inference and Decision Logic

This section describes what happens to one patient's answers after they are
submitted. The logic is identical in the web application and the CLI.

### 13.1 Feature-Vector Construction

`build_feature_vector()` reproduces the 71-column training schema in the
exact order stored in `feature_metadata.json`:

1. Every feature is first initialised to its **training-set mean**.
2. Each answered BDI item overwrites its slot with the 0–3 score.
3. Each answered FCRI item overwrites its slot with the 0–4 score; item 13 is
   stored reverse-scored (`4 − raw`), matching training (§4.4).
4. Demographic answers that correspond to a trained feature (Gender, Age
   Group, Current Cancer Status) are encoded with the persisted label
   encoders; an unrecognised value is encoded as 0. Trained demographic
   features the application does not ask about (Province, Cancer Type,
   Treatment Type, On Medication, Duration Since Diagnosis) keep their
   training mean from step 1.

In Full mode every BDI and FCRI slot is filled from the patient's own
answers; in Quick mode the 48 unasked items (11 BDI + 37 FCRI) keep their
training means.

### 13.2 Model Prediction

The vector is standardised with the saved scaler, then:

- **Binary ensemble** → P(depression). The patient is classed as "depression
  detected" if P ≥ the tuned threshold (0.60, §9).
- **Multiclass ensemble** → six class probabilities. These are mapped from
  the encoder's alphabetical class ids back to severity order (§5); the
  predicted severity is the most probable class, and its probability is
  reported as the model's confidence.

### 13.3 Rule-Based Reference Scores

Alongside the model, the instruments are scored directly from the answers
the patient actually gave, so a clinician can see the conventional scores:

- **BDI total.** Raw sum of answered items. In Quick mode it is extrapolated
  to the 21-item scale as `round(raw × 21 / n_answered)` and labelled with the
  §5 severity bands. A BDI total ≥ 17 (Borderline or above) is treated as
  clinically relevant.
- **FCRI scores.** Per-subscale raw totals (item 13 reversed), an overall
  total, and the **Severity subscale** score (items 9–17, range 0–36),
  extrapolated to 9 items when only some were asked. The Severity score is
  interpreted with the cut-offs of Simard & Savard (2009, 2015): **≥ 13**
  clinically significant fear of recurrence; **≥ 16** clinical-level fear of
  recurrence warranting specialist referral; below 13 non-clinical.

### 13.4 Final Verdict: Cancer-Related Depression

`final_verdict()` (defined once in `predict_depression.py` and imported by
the web application, so both interfaces always agree) combines the model's
severity prediction with the FCRI Severity score:

| Model predicted severity | FCRI Severity score | Final result |
|---|---|---|
| Borderline clinical or worse | ≥ 13 | **YES** — cancer-related depression |
| Borderline clinical or worse | < 13 | **YES** — depression, not driven by fear of recurrence |
| Mild mood disturbance, or binary model positive | ≥ 13 | **POSSIBLE** — at risk of cancer-related depression |
| anything else | any | **NO** — no cancer-related depression detected |

The model, not the rule-based BDI total, decides whether depression is
present; the FCRI decides whether that depression is linked to fear of
cancer recurrence. When FCRI is elevated but the result is NO, the summary
notes that fear of recurrence alone may still benefit from psycho-oncology
support. The BDI total is used as a **sanity check**: if BDI ≥ 17 and the
model's "Borderline or worse" judgement disagree, the report flags the case
for clinician review.

---

## 14. Application Workflow

### 14.1 Quick Assessment

Asks the top 10 BDI items and top 5 FCRI items by combined feature importance
(§12.3), read from `feature_importance.csv` at the start of each assessment —
15 questions, about 7 minutes. Any item not asked is imputed using that
item's training-set mean (§13.1) rather than a neutral or zero value, which
would otherwise bias the imputed value toward "no symptom present." If
`feature_importance.csv` does not exist (before any model has been trained),
a fixed clinical-default item set is used instead: BDI items 1, 2, 4, 5, 9,
15, 16, 17, 18, 20 (covering the core DSM-5 depression domains) and the first
five of a one-item-per-subscale FCRI set (items 9, 14, 15, 18, 22).

Because four of the current top-5 FCRI items come from the Severity
subscale, Quick mode reports a Severity score extrapolated from 4 of 9 items;
the fifth item (25) is one of six Functioning impairments items, and the
other five FCRI subscales are not assessed.

### 14.2 Full Assessment

Presents **all 21 BDI items and all 42 FCRI items** — no item is imputed.
Estimated completion time is approximately 25–30 minutes (63 questions), and
this mode is recommended for first-time patients and for clinical validation
of the screening result.

### 14.3 Data Collection Sequence

Both interfaces follow the same sequence:

1. **Mode selection** — Quick or Full.
2. **Demographics** — five fields: Gender, Age Group, Education Level, Cancer
   Stage, Current Cancer Status. Only Gender, Age Group and Cancer Status
   correspond to trained features; Education Level and Cancer Stage have no
   column in the training data (§3.4) and are collected but do not affect
   the prediction (§16.3, §16.6).
3. **BDI items** — one forced-choice item per question (past week).
4. **FCRI items** — one forced-choice item per question (past month).
   Every item must be answered before continuing.
5. **Prediction and report** — §13.

In the web application each step is a state of the Assessment page
(`mode_selection → demographics → bdi_assessment → fcri_assessment →
results`); the patient can return from FCRI to BDI before submitting.

### 14.4 Results Report

The report is presented in the order the conclusion is reached:

1. **Step 1 — BDI results:** raw and (if Quick) extrapolated total, BDI
   severity, and a table of every answer with its score.
2. **Step 2 — FCRI results:** Severity subscale score and interpretation,
   total score, per-subscale scores, and a table of every answer (item 13
   shown with its reversed score).
3. **Step 3 — Model result and final verdict:** depression probability and
   detection, predicted severity with confidence, the YES / POSSIBLE / NO
   verdict (§13.4), a BDI-disagreement warning where applicable, the
   decision rule, and the full six-class probability breakdown.

This is followed by a clinical-guidance paragraph, a follow-up
recommendation, crisis resources where indicated (§14.5), CSV and PDF export
of the full report, and a disclaimer. Each completed assessment is added to
the session **History** page, which lists past assessments and exports them
as one CSV.

### 14.5 Safety Behavior

- **Suicidal-ideation warning.** If BDI item 9 (Suicidal Thoughts) is scored
  ≥ 2, the interface displays a warning with crisis helpline numbers,
  independent of the ML prediction. Because item 9 is not among the current
  Quick-mode BDI items (§12.3), this check is only reachable in Full mode
  (see §16.6).
- **Follow-up recommendation.** Shown whenever the verdict is YES or
  POSSIBLE, or the result is otherwise clinically relevant.
- **Crisis resources.** Five Pakistan crisis and psycho-oncology helplines
  are listed whenever the result is Moderate depression or above.
- **Disclaimer.** Every result screen and exported report states that the
  tool is for screening only and is not a diagnostic instrument.

---

## 15. Software Implementation

| Component | File | Role |
|---|---|---|
| Training pipeline | `depression_detection_pipeline.py` | Implements §4–§12; the sole source of truth for trained artifacts |
| CLI screening tool | `predict_depression.py` | Terminal-based Full/Quick assessment; also defines `final_verdict()`, shared with the web app |
| Web application — home | `streamlit_app.py` | Landing page; shared question banks, artifact loading, feature-vector construction, prediction, reference scoring, CSV/PDF export |
| Web application — pages | `pages/1_Assessment.py`, `pages/2_Results.py`, `pages/3_History.py` | Questionnaire flow; latest result; session history and history export |
| Shared results view | `results_view.py` | Renders the Step 1 → Step 2 → Step 3 report used by the Assessment and Results pages |
| Trained artifacts | `models/*.joblib`, `models/feature_metadata.json`, `models/feature_importance.csv` | Regenerated by the training pipeline; the applications load the two voting ensembles, `binary_scaler.joblib`, `label_encoders.joblib` and the two metadata files |

---

## 16. Limitations

### 16.1 Target–Feature Circularity in Full Assessment Mode

As noted in §5, both prediction targets are deterministic functions of the
same 21 BDI item scores supplied as model inputs. In Full Assessment mode,
the model is not performing genuine out-of-sample estimation of the label —
it has sufficient information to compute it directly. The framework's
actual predictive value is concentrated in (a) Quick Assessment mode, where
BDI information is incomplete and FCRI/demographic features must compensate,
and (b) producing probability estimates and severity-class confidence scores
rather than a rigid cutoff rule. This should be stated explicitly when
reporting Full-mode accuracy figures, since a naive reading could overstate
the novelty of the modeling contribution. The reported test-set figures
(§12) are Full-information figures; Quick-mode accuracy (with 48 items
mean-imputed) has not been separately evaluated and is expected to be lower.

### 16.2 Boundary-Class Performance

Multiclass performance is weakest on the **Borderline clinical depression**
band (ensemble F1 0.76; §12.2) and, to a lesser degree, the Mild band (F1
0.81). Both are narrow bands sitting between neighbouring classes, so small
item-level differences move patients across a boundary. This matters
clinically because Borderline is exactly the band at which the final verdict
(§13.4) switches from NO/POSSIBLE to YES. The "Normal" class is small in
absolute terms (119 of 1,531 patients, 24 in the test set), so its strong
figures (binary recall 21/24 at 0.50 and 23/24 at the tuned threshold;
multiclass F1 0.89) rest on very few patients;
recruiting more genuinely non-depressed and borderline patients is the most
useful direction for future data collection.

### 16.3 Incomplete Demographic Feature Coverage

The preprocessing pipeline is designed to ingest 14 demographic/clinical
fields (§3.4), but only 8 are present in the current dataset. The absent six
— Cancer Stage, Education Level, Marital Status, Employment Status, Prior
Psychiatric History, Collection Site — are plausible depression risk factors
per the broader psycho-oncology literature and represent a concrete
direction for future data collection, rather than a modeling change that can
be made without new data.

### 16.4 Sample and Design Constraints

The dataset is a single-timepoint, self-report, convenience sample drawn
from patients in Pakistan; it has not been externally validated against an
independent cohort, a clinician-administered diagnostic interview, or a
different population/healthcare setting. Generalizability beyond the
sampled population should not be assumed without such validation.

### 16.5 Instrument Response-Scale Verification

Several FCRI items use non-obvious or instrument-specific response scales
(§3.3) that were verified empirically against the raw response distributions
in this study rather than assumed from the instrument's general
documentation, and one genuine text-encoding defect affecting two items was
identified and corrected during this process (§4.8). This underscores the
importance of validating automated text-to-score parsing against the actual
data distribution — rather than the intended scale alone — for any
free-text or Likert-label survey export.

### 16.6 Known Implementation Gaps in the Current Application

These were identified while verifying this document against the code and
should be resolved before the application is used with patients:

- **Demographic category mismatch.** The application's answer options do not
  all match the categories in the training data. Gender "Other"/"Prefer not
  to say", Age Group "Under 18"/"Over 60" (training data: "Below 18"/"Above
  60"), and Cancer Status values other than "Newly Diagnosed" (training data:
  "Recovered / Survivor", "Recurrence", "Under Treatment") are unseen by the
  encoders and therefore fall back to code 0 (Female, 18–30, Newly
  Diagnosed respectively).
- **Uncollected trained demographics.** Five trained demographic features
  (§13.1) are never asked and are always set to the training mean of their
  integer codes, which is not itself a valid category.
- **Suicidal-ideation check in Quick mode.** BDI item 9 is not in the current
  top-10 Quick item set, so the item 9 warning cannot fire in Quick mode.
- **Suicidal-ideation warning in the web application.** The warning is
  rendered immediately before the page advances to the FCRI step, which
  re-runs the page, so the message is not displayed persistently; the
  `show_suicide_warning` flag it sets is not read anywhere.
- **Inconsistent result triggers between screen and PDF/CLI.** The on-screen
  report shows crisis resources when BDI total ≥ 21 *or* model severity ≥
  Moderate, while the PDF and CLI use model severity only; the on-screen
  follow-up recommendation uses model severity ≥ Borderline, while the PDF and
  CLI use BDI total ≥ 17; and the on-screen clinical-guidance paragraph is
  keyed to the BDI severity while the PDF uses the model's severity.

---

## 17. Ethical Considerations

- Identifying fields (name, timestamp) are removed prior to any analysis
  (§4.1).
- The tool displays crisis-support resources whenever the result is Moderate
  depression or above, and a suicidal-ideation warning when BDI item 9 is
  answered at ≥ 2 (subject to the gaps in §16.6).
- Every result screen carries an explicit disclaimer that the tool is a
  screening aid, not a diagnostic instrument, and recommends professional
  clinical follow-up.
- Session history in the web application is held only in-memory for the
  duration of the browser session (no server-side persistence), limiting
  incidental data retention.
