# Training Pipeline Explained: depression\_detection\_pipeline.py

Oct 14, 2026 · @Amal

## Overview

`depression_detection_pipeline.py` turns the raw survey file into two trained models that the screening tool later loads. You run it once, offline; the CLI (`predict_depression.py`) and the Streamlit app only read what it saves.

- **Input:** `Data/cancer_psychology.csv` — one row per cancer patient (about 1,531 rows), with 21 BDI answers, 42 FCRI answers and demographic fields.
- **Two questions it learns to answer:**
  1. *Binary* — does this patient have depression (BDI total of 11 or more)?
  2. *Multiclass* — which of 6 severity bands are they in (Normal → Extreme)?
- **Models:** for each question it tunes Logistic Regression, Random Forest and XGBoost, then combines them into one soft-voting ensemble.
- **Output:** everything goes into `models/` — the trained models, the scaler, the demographic encoders, `feature_metadata.json` and `feature_importance.csv`.

The script is organised in five classes, run in order by `main()`:

| Section | Class | Job |
| --- | --- | --- |
| 1 | `ClinicalDataPreprocessor` | Clean the raw answers and turn them into numbers |
| 2 | `TargetVariableEngineer` | Create the answers the model must learn (labels) |
| 3 | `DepressionClassificationPipeline` | Tune and train the models, build the ensemble, tune the threshold |
| 4 | `FeatureImportanceAnalyzer` | Rank which questions matter most |
| 5 | `ModelEvaluator` | Score models on unseen data and save everything |

## Step-by-step walkthrough

Each step below matches a numbered comment in `main()`.

### Step 1 — Clean the data (`ClinicalDataPreprocessor.fit_transform`)

Survey answers arrive as text ("I feel sad", "A few times a week"). The model needs numbers, so this step converts and cleans them in a fixed order:

1. **Find the columns** (`_identify_columns`). BDI items are the columns named `1.` to `21.` (or `BDI_1`…). FCRI items start with a number and text, such as `9. I am worried…` (or `FCRI_9`). Demographics are found by keywords like "gender" or "cancer type".
2. **Remove personal data.** Name, Timestamp and Patient ID are dropped so the model never sees them.
3. **Turn text into scores** (`_parse_bdi`, `_parse_fcri`). BDI answers become 0–3 by matching key phrases ("so sad and unhappy" = 3). FCRI answers become 0–4 using `FCRI_SCORE_MAP` ("never" = 0 … "all the time" = 4).
4. **Drop mechanical responders** (`_drop_mechanical`). A patient who answered 0 to every FCRI item is assumed not to have read the questions. This check runs on raw answers, before step 5.
5. **Reverse-score FCRI item 13** (`_reverse_fcri_item_13`). "I believe that I am cured" is worded the opposite way to the other items, so its score becomes 4 − answer. A high score then always means more fear.
6. **Drop incomplete rows** (`_drop_excessive_missing`). Patients missing more than 50% of questionnaire items are removed.
7. **Fill small gaps** (`_impute_subscale_means`). A missing BDI answer is filled with that patient's own average BDI answer. A missing FCRI answer is filled with that patient's average on the same FCRI subscale. If a patient answered fewer than half of a subscale, the row is dropped instead.
8. **Encode demographics** (`_encode_categoricals`). Each text category is turned into a number with a `LabelEncoder` (for example Female = 0, Male = 1), stored in a new `<name>_encoded` column. The encoders are saved so the app can apply the same mapping later.

### Step 2 — Create the labels (`TargetVariableEngineer.create_targets`)

A supervised model needs a correct answer for every training row. Here the answer comes from the BDI:

- `bdi_total` = sum of the 21 BDI items (0–63).
- `has_depression` = 1 if `bdi_total` ≥ 11, else 0 — the **binary** target.
- `depression_severity` = one of 6 bands from the standard BDI cut-offs — the **multiclass** target:

| BDI total | Severity band |
| --- | --- |
| 0–10 | Normal |
| 11–16 | Mild mood disturbance |
| 17–20 | Borderline clinical depression |
| 21–30 | Moderate depression |
| 31–40 | Severe depression |
| 41–63 | Extreme depression |

The band names are then turned into numbers 0–5 by a `LabelEncoder`. It sorts alphabetically, so the numbers are not in severity order; `predict()` in the app re-orders them.

### Step 3 — Build the feature matrix

`X` holds every input the model sees: the 21 BDI items, the 42 FCRI items and the encoded demographics (71 columns in the current `feature_metadata.json`). Any value still missing is set to 0.

### Step 4 — Split into training and test sets

`train_test_split(test_size=0.2, stratify=y_bin)` keeps 80% of patients for training and hides 20% for the final test. **Stratify** means both parts keep the same share of depressed and non-depressed patients. One split is shared by both tasks, so the test patients are the same for binary and multiclass.

### Step 5 — Scale the features

`StandardScaler` rescales every column to mean 0 and standard deviation 1. It is **fit on the training set only**, then applied to the test set, so no information about the test patients leaks into training. The same scaler is saved and reused by the app.

### Step 6 — Train the models (`DepressionClassificationPipeline.train`)

This runs twice: once for the binary target, once for the multiclass target.

1. **Handle class imbalance.** Very few patients are "No Depression" (about 5.6%). `class_weight='balanced'` makes Logistic Regression and Random Forest pay more attention to rare classes. XGBoost has no such option, so it gets matching per-row `sample_weight` values.
2. **Tune each model** (`_tune`). `GridSearchCV` tries every combination in a small settings grid and keeps the best, judged by **5-fold stratified cross-validation** scored with **macro F1**:
   - Logistic Regression: `C` in 0.01, 0.1, 1, 10
   - Random Forest: 100 or 200 trees, depth 5, 10 or unlimited, `min_samples_split` 2 or 5
   - XGBoost: 100 or 200 trees, depth 3, 5 or 7, learning rate 0.01 or 0.1
3. **Build the ensemble.** A `VotingClassifier` with `voting='soft'` averages the three models' probabilities. For multiclass, Logistic Regression's vote counts double (weights 2:1:1) because it scored best on its own; binary uses equal weights.

### Step 7 — Rank feature importance (`FeatureImportanceAnalyzer`)

Random Forest and XGBoost each report how much every feature helped. Both lists are scaled to sum to 1 and averaged into a `combined` score, then saved to `feature_importance.csv`. Quick mode in the app reads this file to pick its top 10 BDI and top 5 FCRI questions.

### Step 8 — Evaluate on the test set (`ModelEvaluator.evaluate`)

Every model (the three base models and the ensemble) predicts the hidden 20%. The script prints accuracy, precision, recall, F1 and ROC-AUC, a per-class report, and saves confusion-matrix images (`binary_confusion.png`, `multiclass_confusion.png`).

**Threshold tuning (binary only)** — `tune_binary_threshold`. By default a model says "depressed" when its probability is 0.50 or more. The script tries every cut-off from 0.05 to 0.95 on out-of-fold predictions from the training set and keeps the one with the best macro F1 (currently **0.46**). The test set is only used to report the effect, never to choose it.

### Step 9 — Export (`ModelEvaluator.export`)

Saves models, scaler, encoders, and `feature_metadata.json`. The metadata holds the exact column order, each feature's training average (used to fill unanswered questions in Quick mode), and the tuned threshold.

### Step 10 — Summary

Prints the best binary and multiclass model by F1 and the top predictive items.

## Glossary of technical terms

Terms are grouped by topic and explained in plain language, with where each appears in the script.

### Clinical scales

| Term | Meaning | In the script |
| --- | --- | --- |
| BDI (Beck Depression Inventory) | A 21-question depression questionnaire. Each item is scored 0–3, so the total runs 0–63. Higher = more depressive symptoms. | `bdi_columns`, `_parse_bdi`, `bdi_total` |
| FCRI (Fear of Cancer Recurrence Inventory) | A 42-question survey about fear that cancer will come back. Each item is scored 0–4. | `fcri_columns`, `_parse_fcri` |
| Subscale | A group of FCRI items measuring one theme: Triggers (1–8), Severity (9–17), Psychological Distress (18–21), Functioning Impairments (22–27), Insight (28–30), Reassurance (31–33), Coping Strategies (34–42). | `FCRI_SUBSCALES` |
| Reverse-scored item | An item worded the opposite way, so its score is flipped (4 − answer) to point the same direction as the rest. Here: FCRI item 13. | `_reverse_fcri_item_13` |
| Severity band | A named range of BDI totals (Normal, Mild … Extreme) from standard BDI cut-offs. | `SEVERITY_BOUNDS` |
| Mechanical response | Answering the same value to everything without reading. Treated as unreliable data. | `_drop_mechanical` |

### Data preparation

| Term | Meaning | In the script |
| --- | --- | --- |
| Preprocessing | Cleaning raw data into a form a model can use: numbers, no gaps, no junk rows. | `ClinicalDataPreprocessor` |
| PII (personally identifiable information) | Data that identifies a person, such as name or patient ID. Removed before training. | `fit_transform` |
| Missing value / NaN | An empty answer. NaN means "not a number". | `isna()`, `fillna` |
| Imputation | Filling a missing value with a sensible estimate instead of deleting the row. | `_impute_subscale_means` |
| Subscale-mean imputation | Filling a gap with the same patient's average on the related items. | `_impute_subscale_means` |
| Categorical variable | A field with named categories, not numbers (gender, province, cancer type). | `demographic_columns` |
| Label encoding | Giving each category a number (e.g. Punjab = 0, Sindh = 1). | `LabelEncoder`, `*_encoded` |
| Feature | One input column the model learns from (one question, or one encoded demographic). | `feature_cols` |
| Feature matrix (X) | The table of all features: one row per patient, one column per feature. | `X` |
| Target / label (y) | The answer the model learns to predict. | `has_depression`, `depression_severity_encoded` |
| Standardisation / scaling | Rescaling each feature to mean 0 and standard deviation 1 so no feature dominates just because of its range. | `StandardScaler` |

### Training

| Term | Meaning | In the script |
| --- | --- | --- |
| Supervised learning | Learning from examples where the correct answer is already known. | the whole pipeline |
| Binary classification | Predicting one of two classes (depressed or not). | `task='binary'` |
| Multiclass classification | Predicting one of more than two classes (the 6 severity bands). | `task='multiclass'` |
| Train/test split | Keeping some patients (here 20%) hidden from training to test the model fairly afterwards. | `train_test_split` |
| Stratified | Splitting so each part keeps the same class proportions as the whole data. | `stratify=y_bin`, `StratifiedKFold` |
| Random state / seed | A fixed number (42) that makes "random" steps repeat identically on every run. | `RANDOM_STATE` |
| Data leakage | When information from the test set (or from the answer) sneaks into training, making results look better than they really are. | scaler fit on train only |
| Class imbalance | One class has far more rows than another (here few patients are "No Depression"). | `class_weight` |
| Class weight / sample weight | Making errors on rare classes cost more, so the model doesn't ignore them. `balanced` sets weights inversely to class size. | `class_weight='balanced'`, `compute_class_weight` |
| Logistic Regression | A simple linear model that turns a weighted sum of features into a probability. | `LogisticRegression` |
| `C` (regularisation) | Logistic Regression setting: smaller C = simpler model that is less likely to memorise noise. | `{'C': [...]}` |
| Decision tree | A model that asks a series of yes/no questions about features to reach a prediction. | inside RF and XGBoost |
| Random Forest | Many decision trees trained on random samples of the data; they vote. | `RandomForestClassifier` |
| XGBoost | Gradient-boosted trees: trees built one after another, each fixing the previous ones' errors. | `XGBClassifier` |
| `n_estimators` | Number of trees. | RF, XGBoost grids |
| `max_depth` | How many questions deep a tree may go. Deeper = more complex. | RF, XGBoost grids |
| `min_samples_split` | Fewest patients a tree node needs before it may split further. | RF grid |
| `learning_rate` | How big a correction each new XGBoost tree makes. Smaller = slower but steadier. | XGBoost grid |
| `objective` / `eval_metric` | What XGBoost optimises: `binary:logistic` / `logloss` for two classes, `multi:softprob` / `mlogloss` for several. `softprob` returns probabilities, which soft voting needs. | `xgb_kwargs` |
| Hyperparameter | A setting chosen before training (like `max_depth`), not learned from data. | the grids in `train` |
| Grid search | Trying every combination of hyperparameter values and keeping the best. | `GridSearchCV` |
| Cross-validation (k-fold) | Splitting the training data into k parts (here 5), training on 4 and checking on the 5th, rotating 5 times. Gives a more reliable score than one check. | `StratifiedKFold(n_splits=5)` |
| Out-of-fold predictions | Predictions for each training row made by a model that did not see that row. | `cross_val_predict` |
| Ensemble | Combining several models into one, usually more accurate and stable. | `VotingClassifier` |
| Soft voting | The ensemble averages the models' probabilities (rather than counting their yes/no votes). | `voting='soft'` |
| Voting weights | How much each model's vote counts. `[2, 1, 1]` = Logistic Regression counts double. | `weights` |
| Probability / `predict_proba` | The model's confidence for each class, from 0 to 1. | `predict_proba` |
| Decision threshold | The probability above which the binary model says "depressed". Default 0.50; tuned here to 0.46. | `tune_binary_threshold` |
| Feature importance | How much each feature helped a tree model's decisions. | `feature_importances_` |

### Evaluation

| Term | Meaning | In the script |
| --- | --- | --- |
| Accuracy | Share of all predictions that were correct. Misleading when classes are imbalanced. | `accuracy_score` |
| Precision | Of the patients the model called depressed, how many really were. | `precision_score` |
| Recall (sensitivity) | Of the patients who really were depressed, how many the model found. | `recall_score` |
| F1 score | A single score balancing precision and recall (their harmonic mean). | `f1_score` |
| Macro average | Average of per-class scores, every class counted equally. Good for imbalanced data. | `scoring='f1_macro'` |
| Weighted average | Average of per-class scores weighted by class size. Favours big classes. | `average='weighted'` in `evaluate` |
| ROC-AUC | How well the model ranks positive cases above negative ones, from 0.5 (random) to 1.0 (perfect). Multiclass uses one-vs-rest (OvR). | `roc_auc_score` |
| Confusion matrix | A table of actual vs predicted classes, showing exactly which classes get confused. | `plot_confusion` |
| Classification report | Precision, recall and F1 for every class, printed together. | `classification_report` |

### Saving

| Term | Meaning | In the script |
| --- | --- | --- |
| Artefact | A saved output file the app needs (model, scaler, encoder, metadata). | `export` |
| joblib | Library for saving Python objects (trained models) to disk and loading them back. | `joblib.dump` |
| Feature metadata | JSON listing the feature order, each feature's training mean and the tuned threshold. | `feature_metadata.json` |

## Review findings

The flow runs correctly end to end, and several earlier bugs are already fixed (one shared split, scaler fit on train only, threshold tuned without touching the test set). The biggest open issue is methodological: the labels are built from the same BDI answers the model takes as input. Findings are ranked by impact.

### 1. The labels are computed from the input features (high impact)

`has_depression` and `depression_severity` are both derived from `bdi_total`, which is the sum of the 21 BDI columns. Those same 21 columns are also features in `X`. The model can therefore learn "add up columns 1–21 and compare with 11", which is a fixed rule, not a pattern.

- This is why binary accuracy reaches about 99.4% (noted in the code comments). The score shows the model can re-add the BDI, not that it detects depression beyond the questionnaire.
- In Full mode the ML prediction is close to the plain BDI score. Its real value is in Quick mode, where it estimates severity from fewer questions.
- **Suggestion:** state this clearly in the methodology. To test what the FCRI and demographics add, also train a model without the BDI items, or report Quick-mode accuracy (top items only) against the full BDI band.

### 2. Unrecognised BDI text silently becomes 0 (medium)

`_parse_bdi` returns 0 ("no symptom") when no keyword matches, while `_parse_fcri` returns NaN. A changed survey wording would quietly push patients towards "Normal", and the conversion-rate check would not notice because 0 is not missing.

- **Suggestion:** return `np.nan` for no match and match the 0-level phrases explicitly, so the existing 80% conversion warning can catch format changes.

### 3. Demographics are label-encoded as if they were ordered (medium)

Province, cancer type and treatment type have no natural order, but `LabelEncoder` gives them 0, 1, 2… Logistic Regression then treats "cancer type 6" as more than "cancer type 2". Tree models are less affected.

- **Suggestion:** use one-hot encoding (one 0/1 column per category) for nominal fields; keep label encoding only for ordered ones such as age group.

### 4. Binary "depression" starts at Mild, but the app's clinical cut-off is Borderline (medium)

`has_depression` = BDI ≥ 11, which includes "Mild mood disturbance". The app's verdict and follow-up rules treat clinical depression as BDI ≥ 17. So the binary model flags many patients the app later calls sub-clinical, and "No Depression" becomes a rare class (about 5.6%).

- **Suggestion:** decide which definition the thesis uses and either set the binary cut-off to 17 or document that the binary model screens for any mood disturbance.

### 5. A patient with no BDI answers is labelled "Normal" (low)

The 50% missing check counts BDI and FCRI together (63 items). A patient who skipped all 21 BDI items but answered the FCRI is kept. Their BDI gaps cannot be filled from their own mean, so they become 0, `bdi_total` = 0, and the row is labelled Normal.

- **Suggestion:** also drop rows missing more than half of the BDI items specifically.

### 6. Scoring is inconsistent between tuning and evaluation (low)

Tuning and threshold selection use **macro** F1, but `evaluate()` and the final "best model" summary use **weighted** F1, which favours the large class. The ensemble's test report in `evaluate()` also uses the default 0.50 cut-off, not the tuned 0.46 (it is reported separately afterwards).

- **Suggestion:** report macro F1 alongside weighted F1 and pick the best model by macro F1.

### 7. Other notes (low)

- `warnings.filterwarnings('ignore')` hides all warnings, including convergence warnings from Logistic Regression. Consider silencing only specific ones.
- Feature importance comes from the binary models only. Quick-mode items are therefore chosen for detecting depression, not for grading severity.
- Rows with all FCRI = 0 are dropped as mechanical. This also removes any genuine patients with no fear of recurrence; mention it as an assumption.

## Output files in models/

The app needs only five of these files to make a prediction; the rest are for reporting and Quick mode.

| File | Contents | Read by |
| --- | --- | --- |
| `binary_voting_ensemble_model.joblib` | Binary ensemble (depressed or not) | `load_artefacts()` — used for every prediction |
| `multiclass_voting_ensemble_model.joblib` | Severity ensemble (6 bands) | `load_artefacts()` — used for every prediction |
| `binary_scaler.joblib` | The fitted `StandardScaler` (shared by both tasks) | `load_artefacts()` |
| `label_encoders.joblib` | One `LabelEncoder` per demographic field | `build_feature_vector()` |
| `feature_metadata.json` | Feature order, training means, tuned threshold (0.46) | `load_artefacts()`, `build_feature_vector()`, `predict()` |
| `feature_importance.csv` | RF, XGBoost and combined importance per feature | `load_top_items()` — picks Quick-mode questions |
| `binary_*_model.joblib`, `multiclass_*_model.joblib` | The individual LR, RF and XGBoost models | Not used by the app; available for SHAP and comparison |
| `multiclass_scaler.joblib` | Same scaler saved a second time | Not used |
| `binary_confusion.png`, `multiclass_confusion.png` | Confusion-matrix images | Reports / thesis |

If you retrain, all of these are overwritten together, so the app always loads a matching set.
