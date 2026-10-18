# Model Flow

How the depression-detection framework is trained, how it is used, and how a
single prediction is made. Diagrams are Mermaid and render on GitHub and in
VS Code's Markdown preview.

1. [System overview](#1-system-overview)
2. [Training pipeline](#2-training-pipeline-depression_detection_pipelinepy)
3. [Assessment workflow](#3-assessment-workflow-predict_depressionpy--streamlit-app)
4. [Model prediction](#4-model-prediction)
5. [Final verdict and safety rules](#5-final-verdict-and-safety-rules)

---

## 1. System overview

```mermaid
flowchart LR
    DATA[("Data/cancer_psychology.csv<br/>BDI + FCRI + demographics")]

    subgraph TRAIN["Offline training — depression_detection_pipeline.py"]
        PIPE["Preprocess → targets → train<br/>LR + RF + XGBoost → soft-vote ensemble"]
    end

    subgraph ART["models/ artefacts"]
        M1["binary_voting_ensemble_model.joblib"]
        M2["multiclass_voting_ensemble_model.joblib"]
        SC["binary_scaler.joblib"]
        LE["label_encoders.joblib"]
        META["feature_metadata.json<br/>feature_cols, feature_means,<br/>binary_threshold"]
        FI["feature_importance.csv"]
    end

    subgraph USE["Online screening"]
        CLI["predict_depression.py (CLI)"]
        WEB["streamlit_app.py<br/>Assessment · Results · History"]
    end

    XAI["shap_analyzer.py<br/>patient-level explanation"]
    OUT["Results<br/>Step 1 BDI · Step 2 FCRI · Step 3 model verdict<br/>CSV / PDF export"]

    DATA --> PIPE --> ART
    ART --> CLI
    ART --> WEB
    FI -. "Quick-mode item ranking" .-> CLI
    FI -.-> WEB
    CLI --> OUT
    WEB --> OUT
    CLI -. "optional" .-> XAI
```

---

## 2. Training pipeline (`depression_detection_pipeline.py`)

```mermaid
flowchart TD
    A[("Load Data/cancer_psychology.csv")] --> B

    subgraph PRE["1 · ClinicalDataPreprocessor.fit_transform"]
        B["Identify BDI / FCRI / demographic columns"] --> B1["Drop PII<br/>Name, Timestamp, Patient ID"]
        B1 --> B2["Parse text answers → numeric<br/>BDI 0-3, FCRI 0-4"]
        B2 --> B3["Drop mechanical responders<br/>checked on raw answers"]
        B3 --> B4["Reverse-score FCRI item 13<br/>4 − raw"]
        B4 --> B5["Drop rows > 50% missing"]
        B5 --> B6["Impute with subscale means"]
        B6 --> B7["Label-encode demographics<br/>*_encoded columns"]
    end

    B7 --> C

    subgraph TGT["2 · TargetVariableEngineer"]
        C["bdi_total = sum of 21 BDI items"] --> C1["Binary target<br/>has_depression = bdi_total ≥ 11"]
        C --> C2["Multiclass target<br/>6 severity bands"]
    end

    C1 --> D
    C2 --> D
    D["3 · Feature matrix X<br/>21 BDI + 42 FCRI + encoded demographics"] --> E
    E["4 · Stratified 80/20 split<br/>(one split shared by both tasks)"] --> F
    F["5 · StandardScaler<br/>fit on train only"] --> G

    subgraph MOD["6 · DepressionClassificationPipeline.train — run for binary AND multiclass"]
        G["StratifiedKFold(5) GridSearchCV<br/>scoring = f1_macro"]
        G --> LR["Logistic Regression<br/>class_weight=balanced"]
        G --> RF["Random Forest<br/>class_weight=balanced"]
        G --> XGB["XGBoost<br/>balanced sample weights"]
        LR --> ENS["Soft-voting ensemble<br/>weights 2:1:1 multiclass, equal for binary"]
        RF --> ENS
        XGB --> ENS
    end

    ENS --> H["7 · Feature importance<br/>RF + XGB → feature_importance.csv"]
    ENS --> I["8 · Evaluate on test set<br/>reports + confusion matrices"]
    ENS --> T["Tune binary threshold<br/>out-of-fold CV on train, max macro-F1"]
    H --> J
    I --> J
    T --> J
    J[["9 · Export to models/<br/>models, scaler, label encoders,<br/>feature_metadata.json"]]
```

**Severity bands (BDI total → label)**

| BDI total | Label |
|---|---|
| 0–10 | Normal |
| 11–16 | Mild mood disturbance |
| 17–20 | Borderline clinical depression |
| 21–30 | Moderate depression |
| 31–40 | Severe depression |
| 41–63 | Extreme depression |

> **Known scoring issue in the training data (found 2026-10-18).** `_parse_bdi()` matches keywords inside
> the answer text, so the score-0 statements of BDI item 6 ("I don't feel I am being punished") and item 7
> ("I don't feel disappointed in myself") are scored 3 and 1. This affects 717 answers and changes 185
> severity labels in `Data/cancer_psychology.csv`; the saved models in `models/` were trained on these labels.
> The app itself is not affected at prediction time, because the Assessment page scores answers by option
> position. The thesis analysis (`thesis_analysis/`) applies the corrected scoring; the deployed models have
> not been retrained.

---

## 3. Assessment workflow (`predict_depression.py` / Streamlit app)

```mermaid
flowchart TD
    S([Start]) --> L["load_artefacts()<br/>models, scaler, encoders, metadata"]
    L --> MODE{"Choose mode"}

    MODE -- "Quick (~7 min)" --> Q["load_top_items()<br/>top 10 BDI + top 5 FCRI<br/>from feature_importance.csv"]
    Q --> SAFE["with_safety_item()<br/>always add BDI item 9<br/>(suicidal thoughts)"]
    MODE -- "Full (~25-30 min)" --> FULL["All 21 BDI + all 42 FCRI"]

    SAFE --> COL
    FULL --> COL
    COL["Collect answers<br/>demographics → BDI → FCRI"] --> SR{"BDI item 9 ≥ 2?"}
    SR -- Yes --> WARN["⚠ Show crisis helplines immediately"]
    SR -- No --> FV
    WARN --> FV

    FV["build_feature_vector()"] --> P["predict()"]
    P --> R1["Step 1 · bdi_reference()<br/>raw total, extrapolated to 21 items"]
    P --> R2["Step 2 · fcri_reference()<br/>subscale scores, Severity subscale 0-36"]
    R1 --> V
    R2 --> V
    P --> V["Step 3 · final_verdict()"]
    V --> DISP["Display results<br/>+ follow-up / crisis messages"]
    DISP --> X{"Show SHAP explanation?"}
    X -- Yes --> SH["SHAPAnalyzer.explain_patient()<br/>top contributing features + plots"]
    X -- No --> AG
    SH --> AG{"Another assessment?"}
    AG -- Yes --> MODE
    AG -- No --> E([End])
```

---

## 4. Model prediction

What happens inside `build_feature_vector()` and `predict()` for one patient.

```mermaid
flowchart TD
    IN["Patient answers<br/>BDI · FCRI · demographics"] --> INIT

    subgraph BFV["build_feature_vector()"]
        INIT["Start every feature at its<br/>training mean (feature_means)"] --> FB["Overwrite asked BDI items"]
        FB --> FF["Overwrite asked FCRI items<br/>item 13 reversed: 4 − raw"]
        FF --> FD["Encode demographics with saved LabelEncoders<br/>unseen category → keep training mean"]
        FD --> VEC["Ordered vector matching feature_cols"]
    end

    VEC --> SCL["StandardScaler.transform"]

    SCL --> BIN["Binary voting ensemble<br/>predict_proba"]
    SCL --> MUL["Multiclass voting ensemble<br/>predict_proba"]

    BIN --> TH{"P(depression) ≥ tuned threshold?<br/>(binary_threshold ≈ 0.46)"}
    TH -- Yes --> BP1["binary_pred = 1"]
    TH -- No --> BP0["binary_pred = 0"]

    MUL --> REORD["Re-map class ids → labels<br/>(LabelEncoder is alphabetical)<br/>reorder Normal → Extreme"]
    REORD --> ARG["argmax → severity_label<br/>max prob → confidence %"]

    BP1 --> OUT
    BP0 --> OUT
    ARG --> OUT
    OUT[["pred = {binary_pred, depression_prob,<br/>severity_idx, severity_label,<br/>confidence, multi_proba}"]]
```

In Quick mode, unasked items keep their training-mean value, so the model
always receives a full-length vector without treating missing answers as
"no symptom".

---

## 5. Final verdict and safety rules

`final_verdict()` combines the model's output with the FCRI Severity subscale
(cut-off ≥ 13, Simard & Savard).

```mermaid
flowchart TD
    START["pred + bdi_ref + fcri_ref"] --> MC{"Model severity<br/>≥ Borderline?"}

    MC -- Yes --> F1{"FCRI Severity ≥ 13?"}
    F1 -- Yes --> YES1["🔴 YES<br/>Cancer-related depression detected"]
    F1 -- No --> YES2["🟠 YES<br/>Depression detected<br/>(not driven by fear of recurrence)"]

    MC -- No --> MM{"Model Mild<br/>OR binary_pred = 1?"}
    MM -- Yes --> F2{"FCRI Severity ≥ 13?"}
    F2 -- Yes --> POSS["🟡 POSSIBLE<br/>At risk of cancer-related depression"]
    F2 -- No --> NO["🟢 NO<br/>No cancer-related depression"]
    MM -- No --> NO

    YES1 --> CHK
    YES2 --> CHK
    POSS --> CHK
    NO --> CHK
    CHK["Sanity check: does the rule-based BDI total<br/>(≥ 17) agree with the model?"]
```

Safety messages are shown when **either** the model or the BDI total triggers
them:

```mermaid
flowchart LR
    subgraph FU["needs_follow_up()"]
        A1["verdict ≠ NO"] --> FO["Recommend clinical follow-up"]
        A2["model severity ≥ Borderline"] --> FO
        A3["BDI total ≥ 17"] --> FO
    end

    subgraph CR["needs_crisis_resources()"]
        B1["model severity ≥ Moderate"] --> CRI["Show crisis helplines"]
        B2["BDI total ≥ 21"] --> CRI
        B3["BDI item 9 ≥ 2<br/>(suicidal thoughts)"] --> CRI
    end
```
