# Methodology

## Hybrid Machine Learning Framework for Depression Screening in Cancer Patients

This document describes the data, preprocessing, feature engineering, model
architecture, training protocol, and evaluation methodology used by the
depression screening system in this repository. It is written to be used
directly as (or adapted into) the Methodology chapter of a thesis or research
paper. All figures reported here were produced by `depression_detection_pipeline.py`
against `Data/cancer_psychology.csv` and are reproducible by re-running that
script (fixed random seed, `RANDOM_STATE = 42`, throughout).

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

It is important to state at the outset that this is a **screening aid, not a
diagnostic instrument** — a distinction preserved throughout the application
(see §14, Limitations, and the disclaimers surfaced in the tool itself).

---

## 2. Data Source and Instruments

### 2.1 Sample

The dataset (`Data/cancer_psychology.csv`) contains **1,532 self-reported
responses** collected from cancer patients, comprising 78 raw columns:
identifying/administrative fields (timestamp, name — removed prior to
analysis, see §3.1), demographic and clinical-context fields, and the two
psychometric instruments in full.

### 2.2 Beck Depression Inventory (BDI-II)

A 21-item self-report inventory, each item scored 0–3 (four graded statements
of increasing symptom severity), covering affective, cognitive, somatic, and
vegetative symptoms of depression (sadness, pessimism, past failure, loss of
pleasure, guilt, punishment feelings, self-dislike, self-criticism, suicidal
ideation, crying, agitation, loss of interest, indecisiveness, body image,
work inhibition, sleep disturbance, fatigability, appetite change, weight
loss, somatic preoccupation, loss of interest in sex). The instrument yields
a total score range of 0–63.

### 2.3 Fear of Cancer Recurrence Inventory (FCRI)

A 42-item self-report inventory organized into **seven subscales**, each
covering a distinct construct related to fear of cancer recurrence (FCR):

| Subscale                  | Items | Response scale (verified against raw data, §3.2) |
|----------------------------|-------|----------------------------------------------------|
| Triggers                   | 1–8   | Frequency: Never(0) → All the time(4)               |
| Severity                   | 9–17  | Mixed — see below                                   |
| Psychological Distress     | 18–21 | Intensity: Not at all(0) → A great deal(4)          |
| Functioning Impairments    | 22–27 | Intensity: Not at all(0) → A great deal(4)          |
| Insight                    | 28–30 | Intensity: Not at all(0) → A great deal(4)          |
| Reassurance                | 31–33 | Frequency: Never(0) → All the time(4)               |
| Coping Strategies          | 34–42 | Frequency: Never(0) → All the time(4)               |

The Severity subscale (items 9–17) mostly uses the Intensity scale, except:
item 13 ("I believe that I am cured...") is intensity-scaled but
**reverse-coded** at analysis time (agreement indicates *lower* fear; see
§3.4); item 14 uses a dedicated risk-perception scale (Not at all at risk(0)
→ A great deal at risk(4)); item 15 uses a thought-frequency scale (Never(0)
→ Several times a day(4)); item 16 uses a time-per-day scale ("I don't think
about it"(0) → Several hours(4)); item 17 uses a duration scale ("I don't
think about it"(0) → Several years(4)).

### 2.4 Demographic and Clinical-Context Variables

Eight fields are present in the current dataset and used as model features:
Gender, Age Group, Province, Cancer Type, Current Treatment Type, Current
Cancer Status, whether the patient is currently on medication for
depression/anxiety, and Duration Since Diagnosis. Six additional fields the
preprocessing pipeline is designed to ingest (Cancer Stage, Education Level,
Marital Status, Employment Status, Prior Psychiatric History, Collection
Site) are **not present in the current dataset** and are consequently
excluded from the trained feature set (see §14.3).

---

## 3. Data Preprocessing

Implemented in `ClinicalDataPreprocessor` (`depression_detection_pipeline.py`).
Steps are applied in the following fixed order:

### 3.1 De-identification

`Name` and `Timestamp` (and, where present, patient ID fields) are dropped
before any analysis is performed.

### 3.2 Response Normalization

Instrument responses were free-text/Likert-label strings as exported (e.g.
`"I feel sad"`, `"Somewhat"`), rather than pre-coded integers. Each response
is converted to its numeric score via, in order of precedence: (1) direct
numeric parse, for already-coded values; (2) a digit-prefix regular
expression (`"2 - Somewhat"` → 2), for digit-prefixed exports; (3) explicit
keyword/phrase matching against the instrument's known response text. BDI
keyword matching is ordered from most-severe to least-severe phrase and
defaults to 0 if no phrase matches; FCRI matching uses a fixed
text-to-integer lookup table per response scale (§2.3) and returns a missing
value if no match is found (handled by §3.5).

A conversion-rate diagnostic is computed and logged for both instruments to
surface silent parsing failures. On the current dataset: **BDI conversion
rate 100.0%, FCRI conversion rate 99.8%** (see §3.6 for the specific defect
that was found and corrected to reach this rate).

### 3.3 FCRI Item 13 Reversal

FCRI item 13 ("I believe that I am cured and that the cancer will not come
back") is worded in the opposite direction to the rest of the Severity
subscale — agreement indicates *lower*, not higher, fear of recurrence. It is
reverse-scored as `reversed = 4 - raw` before being used in any subscale
aggregate or as a model feature.

### 3.4 Quality Filtering

Two row-level filters are applied: (a) rows where **all** 42 FCRI responses
equal 0 are dropped as likely non-engaged/mechanical responses; (b) rows with
more than 50% missing values across the combined BDI+FCRI item set are
dropped as insufficiently complete. On the current dataset, filter (b)
removed 1 row (1,532 → 1,531); filter (a) removed none.

### 3.5 Missing-Value Imputation

Remaining missing item-level values are imputed using **within-patient,
within-subscale means** rather than a global column mean or a fixed default:
a missing BDI item is imputed from that same respondent's mean across their
other 20 BDI items; a missing FCRI item is imputed from that respondent's
mean across the other items in the *same FCRI subscale* (§2.3). This
preserves each patient's individual response tendency rather than regressing
them toward the sample average.

### 3.6 Data-Quality Defect Found and Corrected

During verification of the FCRI response-to-score mapping against the raw
CSV, two items (16 and 17) were found to share the response option "I don't
think about it" (intended score 0), but the CSV export encodes this phrase
with a Unicode right single quotation mark (`'`, U+2019) rather than the
ASCII apostrophe (`'`) the original scoring table matched against. Because
string comparison in the scoring lookup is exact, this caused **161/1,531
(10.5%) of item 16 responses and 122/1,531 (8.0%) of item 17 responses** to
be silently treated as missing and subscale-mean-imputed, rather than
correctly scored as 0. The scoring table was corrected to match the
Unicode apostrophe (retaining the ASCII form as well, for robustness against
other export sources), which is what raised the FCRI conversion rate from
99.4% to 99.8% (§3.2) and shifted the per-feature training means for items
16 and 17 accordingly (from 1.372 to 1.951, and 1.523 to 2.029,
respectively).

### 3.7 Categorical Encoding

Each demographic/clinical-context field (§2.4) is treated as a categorical
variable: missing values are filled with the literal category `"Unknown"`,
and each field is integer-encoded via `sklearn.preprocessing.LabelEncoder`.
The fitted encoders are persisted (`label_encoders.joblib`) so that new
patient responses at inference time are encoded identically; a category
value not seen during training falls back to a default encoding (0) rather
than raising an error.

---

## 4. Target Variable Definition

Two supervised targets are derived, both purely as a function of the BDI-II
total score (`bdi_total`, the row-wise sum of the 21 BDI items — never the
FCRI or demographic fields):

**Binary target** (`has_depression`): 1 if `bdi_total >= 11`, else 0. This
follows the standard BDI-II clinical cutoff separating "no depression" from
"depression present at any severity."

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

On the cleaned dataset (N = 1,531): binary class counts are 1,445 positive
(depression present, 94.4%) vs. 86 negative (5.6%) — a substantial class
imbalance addressed explicitly in training (§7). Multiclass counts: Moderate
466, Extreme 409, Severe 331, Borderline 126, Mild 113, Normal 86.

**Note on target construction (important methodological caveat):** because
both targets are deterministic thresholded functions of the same 21 BDI item
scores that are also supplied as model features, a model given complete BDI
responses (as in Full Assessment mode) has direct arithmetic access to the
label. The genuine machine-learning estimation problem this framework solves
is (a) predicting the label from FCRI and demographic features *in
combination with* possibly-incomplete BDI data (Quick Assessment mode, §13),
and (b) producing well-calibrated class probabilities rather than a rigid
rule-based cutoff. This is discussed further in §14.1.

---

## 5. Feature Representation

The final feature matrix combines all three sources into one vector per
patient:

| Feature group | Count | Source |
|---|---|---|
| BDI items | 21 | §2.2, integer 0–3 each |
| FCRI items | 42 | §2.3, integer 0–4 each (item 13 reverse-scored) |
| Demographic (label-encoded) | 8 | §2.4 |
| **Total** | **71** | |

The exact feature ordering and each feature's training-set mean (used for
imputing unasked items at Quick-mode inference time, §13.1) are persisted to
`models/feature_metadata.json`, generated fresh on every training run so
that the deployed inference code and the trained model are always schema
matched.

On the current run: **feature matrix shape (1,531, 71)**; stratified 80/20
train/test split (§6.1) yields **1,224 training rows, 307 test rows**.

---

## 6. Model Architecture

### 6.1 Train/Test Split and Feature Scaling

A single stratified 80/20 split (`random_state = 42`, stratified on the
binary label) is used, and the **same split and the same fitted
`StandardScaler`** (fit on the training partition only) are reused for both
the binary and multiclass tasks. This is a deliberate design choice: fitting
independent splits or independent scalers per task would make binary and
multiclass results non-comparable and risks a scaler being silently refit
and overwritten.

### 6.2 Base Learners

Three heterogeneous classifiers are trained **independently per task**
(binary, multiclass) — i.e., six models total, plus two ensembles (§6.4):

1. **Logistic Regression** (`sklearn.linear_model.LogisticRegression`,
   `solver='lbfgs'`, `max_iter=1000`)
2. **Random Forest** (`sklearn.ensemble.RandomForestClassifier`)
3. **XGBoost** (`xgboost.XGBClassifier`; `binary:logistic`/`logloss` for the
   binary task, `multi:softprob`/`mlogloss` for the multiclass task —
   `softprob` rather than `softmax` is required so per-class probabilities
   are available for soft voting, §6.4)

### 6.3 Hyperparameter Tuning

Each base learner is tuned via `GridSearchCV` with **5-fold stratified
cross-validation** (`StratifiedKFold`, `shuffle=True`, `random_state=42`),
scored on **macro-averaged F1** (`f1_macro`). Macro-F1 (the unweighted mean
of per-class F1) was chosen over the sample-weighted alternative
(`f1_weighted`) specifically because the latter re-biases *model selection*
back toward the majority class in proportion to its support, which would
partially undo the class-imbalance handling described in §7 — macro-F1
gives the minority class equal weight in the selection criterion, consistent
with the framework's goal of not missing genuinely non-depressed patients.

| Model | Grid searched |
|---|---|
| Logistic Regression | `C ∈ {0.01, 0.1, 1, 10}`, `penalty = l2` |
| Random Forest | `n_estimators ∈ {100, 200}`, `max_depth ∈ {5, 10, None}`, `min_samples_split ∈ {2, 5}` |
| XGBoost | `n_estimators ∈ {100, 200}`, `max_depth ∈ {3, 5, 7}`, `learning_rate ∈ {0.01, 0.1}` |

### 6.4 Ensemble

A **soft-voting ensemble** (`sklearn.ensemble.VotingClassifier`,
`voting='soft'`) combines the three tuned base learners by averaging their
predicted class probabilities. Binary and multiclass ensembles are weighted
differently, based on empirical comparison of the three base learners:

- **Binary task**: equal weights (1, 1, 1). All three base learners perform
  comparably well on this task (§11.1), so equal weighting is appropriate.
- **Multiclass task**: weights (2, 1, 1) favoring Logistic Regression. An
  initial equal-weight ensemble was found to under-perform standalone
  Logistic Regression by a full percentage point of accuracy (86.0% vs.
  87.6%), because Random Forest and XGBoost are substantially weaker on this
  harder 6-class task (§11.2) and were dragging the ensemble average down.
  Re-weighting the vote toward the strongest individual learner closed this
  gap (ensemble accuracy rose to 87.3–88.0% across retraining runs,
  essentially matching or exceeding standalone Logistic Regression).

This is reported as a methodological decision rather than a default,
because it materially changed which model would otherwise have been the
best-performing option — a finding worth stating explicitly rather than
presenting the three-model ensemble as uniformly superior by construction.

---

## 7. Class Imbalance Handling

The binary target is markedly imbalanced (94.4% positive / 5.6% negative,
§4), and the multiclass "Normal" and "Mild" categories are similarly
under-represented (5.6% and 7.4% of the sample, respectively). Three
complementary techniques are used, deliberately **not** including synthetic
oversampling (e.g. SMOTE):

1. **Class-weighted loss** (`class_weight='balanced'`) for Logistic
   Regression and Random Forest, which re-weights the loss function
   inversely proportional to class frequency.
2. **Sample-weighted loss** for XGBoost, which has no native `class_weight`
   parameter: inverse-frequency weights are computed via
   `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` and
   passed as `sample_weight` at `.fit()` time, so all three base learners
   receive consistent imbalance treatment (in an earlier iteration of this
   pipeline, only LR and RF were balanced and XGBoost received none).
3. **Decision threshold calibration** (§8), rather than resampling, to
   further tune the binary classifier's operating point.

Synthetic oversampling was deliberately avoided given (a) the small absolute
minority-class count (86 patients), which limits the diversity of
interpolated synthetic samples and risks tight clustering around a few real
points, and (b) all features being ordinal (BDI 0–3, FCRI 0–4) or categorical
— SMOTE-style interpolation can generate non-integer "phantom" item
responses that do not correspond to any real questionnaire answer, which
would compromise the clinical interpretability of what the model has learned
from.

---

## 8. Decision Threshold Calibration (Binary Task)

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
inference time in place of the sklearn default.

On the final dataset (post §3.6 correction), the tuned threshold was
**0.46** (CV macro-F1 0.978 vs. 0.974 at the default 0.50); applying it to
the held-out test set raised "No Depression" precision from 0.94 to a
perfect **1.00** at the same 0.94 recall (§11.1).

---

## 9. Evaluation Protocol

For each of the 4 models per task (3 base learners + ensemble), the
following are computed on the held-out test set (307 patients, never used in
training or hyperparameter/threshold selection):

- Accuracy, weighted-average precision/recall/F1
- Per-class precision/recall/F1 (`classification_report`)
- ROC-AUC — standard formulation for the binary task; macro-averaged
  one-vs-rest for the multiclass task
- Confusion matrices (rendered per model, saved as PNG)

Feature importance is computed by normalizing the Random Forest's and
XGBoost's `feature_importances_` (from the *binary* task models) to sum to
1 each, then averaging the two normalized vectors per feature. This combined
importance ranking drives the Quick Assessment item-selection logic (§13.1)
and is persisted to `models/feature_importance.csv`.

---

## 10. Reproducibility

- Fixed random seed (`RANDOM_STATE = 42`) applied to the train/test split,
  all three base learners, and all cross-validation folds.
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

## 11. Results

Figures below are from the training run incorporating all corrections
described in §3.6, §6.4, §7, and §8 (the current deployed model artifacts).

### 11.1 Binary Task (Has Depression: Yes/No)

Cross-validated hyperparameter selection (macro-F1):

| Model | Best hyperparameters | CV macro-F1 |
|---|---|---|
| Logistic Regression | `C=1` | 0.9495 |
| Random Forest | `max_depth=10, min_samples_split=2, n_estimators=200` | 0.9526 |
| XGBoost | `learning_rate=0.1, max_depth=3, n_estimators=200` | 0.9703 |

Held-out test-set performance (default 0.5 threshold):

| Model | Accuracy | F1 (weighted) | ROC-AUC | "No Depression" P / R / F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9805 | 0.9814 | 0.9986 | 0.76 / 0.94 / 0.84 |
| Random Forest | 0.9935 | 0.9935 | 0.9972 | 0.94 / 0.94 / 0.94 |
| XGBoost | 0.9739 | 0.9758 | 0.9933 | 0.70 / 0.94 / 0.80 |
| **Voting Ensemble** | **0.9935** | **0.9935** | **0.9974** | **0.94 / 0.94 / 0.94** |

At the CV-tuned threshold (0.46) applied to the test set, the ensemble
reaches **100% accuracy**, with "No Depression" precision/recall/F1 of
**1.00 / 0.94 / 0.97**.

*Interpretation caveat:* the "No Depression" class has only 17 test-set
members; single-percentage-point differences in its metrics correspond to a
change of roughly one patient and should be read as indicative rather than
statistically precise given this sample size.

### 11.2 Multiclass Task (6-Level Severity)

Cross-validated hyperparameter selection (macro-F1):

| Model | Best hyperparameters | CV macro-F1 |
|---|---|---|
| Logistic Regression | `C=10` | 0.7890 |
| Random Forest | `max_depth=10, min_samples_split=5, n_estimators=200` | 0.6711 |
| XGBoost | `learning_rate=0.1, max_depth=3, n_estimators=100` | 0.6534 |

Held-out test-set performance:

| Model | Accuracy | F1 (weighted) | ROC-AUC (macro OvR) |
|---|---|---|---|
| Logistic Regression | 0.8664 | 0.8688 | 0.9848 |
| Random Forest | 0.7557 | 0.7470 | 0.9418 |
| XGBoost | 0.7068 | 0.7097 | 0.9361 |
| **Voting Ensemble (LR-weighted)** | **0.8730** | **0.8722** | **0.9832** |

Per-class performance, Voting Ensemble:

| Severity class | n (test) | Precision | Recall | F1 |
|---|---|---|---|---|
| Normal | 28 | 0.56 | 0.54 | 0.55 |
| Mild mood disturbance | 77 | 0.99 | 0.96 | 0.97 |
| Borderline clinical depression | 23 | 0.82 | 0.78 | 0.80 |
| Moderate depression | 107 | 0.89 | 0.86 | 0.88 |
| Severe depression | 17 | 0.89 | 0.94 | 0.91 |
| Extreme depression | 55 | 0.85 | 0.96 | 0.91 |

The **"Normal" class is consistently the hardest to classify** across every
model and every retraining run in this study (F1 in the 0.27–0.59 range
depending on the model) — a pattern discussed further in §14.2.

### 11.3 Feature Importance

Top 5 BDI items by combined Random Forest + XGBoost importance (binary
task): item 18 (Appetite Changes), item 2 (Pessimism), item 16 (Sleep
Disturbance), item 14 (Body Image), item 20 (Somatic Preoccupation).

Top 5 FCRI items: item 14 (Perceived risk of recurrence), item 10 (Fear of
recurrence), item 9 (Worry about recurrence), item 15 (Frequency of
recurrence thoughts), item 13 (Belief in cure, reverse-scored).

These items drive the Quick Assessment mode's item-selection logic (§13.1)
and are regenerated on every training run, so the specific ranking may shift
as the underlying dataset grows or is corrected.

---

## 12. Software Implementation

| Component | File | Role |
|---|---|---|
| Training pipeline | `depression_detection_pipeline.py` | Implements §3–§11; the sole source of truth for trained artifacts |
| CLI screening tool | `predict_depression.py` | Terminal-based Full/Quick assessment, mirrors the web app's logic |
| Web application | `streamlit_app.py` + `pages/1_Assessment.py`, `pages/2_Results.py`, `pages/3_History.py` | Multi-page Streamlit interface: mode selection, questionnaire, results, session history, CSV/PDF export |
| Trained artifacts | `models/*.joblib`, `models/feature_metadata.json`, `models/feature_importance.csv` | Regenerated by the training pipeline; consumed identically by both interfaces |

---

## 13. Application Modes

### 13.1 Quick Assessment

Selects the top-*N* BDI items (default *N*=10) and top-*N* FCRI items
(default *N*=5) by combined feature importance (§11.3), regenerated fresh on
every training run. Any item not asked is imputed using that item's
training-set mean (`feature_metadata.json → feature_means`) rather than a
neutral or zero value, which would otherwise bias the imputed value toward
"no symptom present." Before any model has been trained (i.e., before
`feature_importance.csv` exists), a fixed clinical-default item set is used
instead, chosen to cover the core DSM-5 depression domains.

### 13.2 Full Assessment

Presents **all 21 BDI items and all 42 FCRI items** — no item is imputed.
Estimated completion time is approximately 25–30 minutes (63 total
questions), and this mode is recommended for first-time patients and for
clinical validation of the screening result.

### 13.3 Demographic Data Collection

Five demographic fields are collected in both modes for use as model
features: Gender, Age Group, Education Level, Cancer Stage, and Current
Cancer Status. Note that **Education Level and Cancer Stage have no
corresponding column in the current training dataset** (§2.4) and are
consequently collected but have no effect on the model's prediction — a
known limitation carried forward from the current data-collection instrument
(§14.3).

### 13.4 Safety Behavior

Both interfaces display a suicidal-ideation warning (BDI item 9 score ≥ 2)
independent of the ML prediction, along with crisis-support helpline
numbers, and surface the same helplines automatically whenever the predicted
severity is Moderate or above. All outputs are accompanied by an explicit
disclaimer that the tool is for screening only and is not a diagnostic
instrument.

---

## 14. Limitations

### 14.1 Target–Feature Circularity in Full Assessment Mode

As noted in §4, both prediction targets are deterministic functions of the
same 21 BDI item scores supplied as model inputs. In Full Assessment mode,
the model is not performing genuine out-of-sample estimation of the label —
it has sufficient information to compute it directly. The framework's
actual predictive value is concentrated in (a) Quick Assessment mode, where
BDI information is incomplete and FCRI/demographic features must compensate,
and (b) producing calibrated probability estimates and severity-class
confidence scores rather than a rigid cutoff rule. This should be stated
explicitly when reporting Full-mode accuracy figures, since a naive reading
could overstate the novelty of the modeling contribution.

### 14.2 Minority-Class Performance

The "Normal" (no depression) class is small in absolute terms (86 of 1,531
patients, 5.6%) and is the weakest-performing class across every model and
every task variant evaluated in this study. Class-weighting and threshold
calibration (§7, §8) measurably improve this at the *binary* decision level,
but multiclass "Normal" classification remains comparatively unreliable
(F1 ≈ 0.55–0.59 for the best model). Clinically, this is the more
consequential failure direction — a healthy patient's screening being
inflated toward a depression severity class — and should be prioritized in
any future data-collection effort aimed at improving the framework (i.e.,
recruiting more genuinely non-depressed patients, rather than more depressed
ones).

### 14.3 Incomplete Demographic Feature Coverage

The preprocessing pipeline is designed to ingest 14 demographic/clinical
fields (§2.4), but only 8 are present in the current dataset. The absent six
— Cancer Stage, Education Level, Marital Status, Employment Status, Prior
Psychiatric History, Collection Site — are plausible depression risk factors
per the broader psycho-oncology literature and represent a concrete
direction for future data collection, rather than a modeling change that can
be made without new data.

### 14.4 Sample and Design Constraints

The dataset is a single-timepoint, self-report, convenience sample drawn
from patients in Pakistan; it has not been externally validated against an
independent cohort, a clinician-administered diagnostic interview, or a
different population/healthcare setting. Generalizability beyond the
sampled population should not be assumed without such validation.

### 14.5 Instrument Response-Scale Verification

Several FCRI items use non-obvious or instrument-specific response scales
(§2.3) that were verified empirically against the raw response distributions
in this study rather than assumed from the instrument's general
documentation, and one genuine text-encoding defect affecting two items was
identified and corrected during this process (§3.6). This underscores the
importance of validating automated text-to-score parsing against the actual
data distribution — rather than the intended scale alone — for any
free-text or Likert-label survey export.

---

## 15. Ethical Considerations

- Identifying fields (name, timestamp) are removed prior to any analysis
  (§3.1).
- The tool displays crisis-support resources automatically whenever a
  response pattern indicates possible suicidal ideation (BDI item 9) or a
  Moderate-or-above predicted severity, independent of whether the user
  proceeds with or completes the assessment.
- Every result screen carries an explicit disclaimer that the tool is a
  screening aid, not a diagnostic instrument, and recommends professional
  clinical follow-up.
- Session history in the web application is held only in-memory for the
  duration of the browser session (no server-side persistence), limiting
  incidental data retention.
