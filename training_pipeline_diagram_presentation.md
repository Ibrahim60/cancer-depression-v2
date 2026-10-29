# Training Pipeline

`depression_detection_pipeline.py`: from raw survey answers to saved, ready-to-use models.

```mermaid
flowchart TB
    subgraph P1["PHASE 1 · PREPARE DATA"]
        direction LR
        A[/"<b>Raw survey data</b><br/>cancer_psychology.csv<br/>≈1,531 patients<br/>21 BDI + 42 FCRI + demographics"/]
        B["<b>1 · Clean and convert</b><br/>Remove personal info<br/>Text answers → scores<br/>Drop careless / empty rows<br/>Reverse item 13 · fill gaps"]
        C["<b>2 · Create labels</b><br/>BDI total (0–63)<br/>Binary: total ≥ 11 → depressed<br/>Multiclass: 6 severity bands"]
        D["<b>3 · Feature matrix X</b><br/>21 BDI + 42 FCRI items<br/>Encoded demographics"]
        A --> B --> C --> D
    end

    subgraph P2["PHASE 2 · TRAIN MODELS"]
        direction LR
        E["<b>4 · Split 80 / 20</b><br/>Stratified<br/>One split for both tasks"]
        F["<b>5 · Scale features</b><br/>StandardScaler<br/>Fit on train only"]
        subgraph G["6 · Train — binary and multiclass"]
            direction LR
            M1["Logistic Regression"]
            M2["Random Forest"]
            M3["XGBoost"]
            ENS(["<b>Soft-voting ensemble</b><br/>binary 1:1:1 · multi 2:1:1"])
            M1 & M2 & M3 --> ENS
        end
        E --> F --> G
    end

    subgraph P3["PHASE 3 · EVALUATE AND SAVE"]
        direction LR
        H["<b>7 · Feature importance</b><br/>RF + XGBoost averaged<br/>Top 5 BDI and FCRI items"]
        I["<b>8a · Evaluate on test set</b><br/>Accuracy · Precision · Recall<br/>F1 · ROC-AUC · confusion matrix"]
        J["<b>8b · Tune binary threshold</b><br/>CV on train set<br/>Best macro-F1 cut-off"]
        K[("<b>9 · Export to models/</b><br/>8 models · scaler · encoders<br/>feature_metadata.json")]
        H --> I --> J --> K
    end

    P1 --> P2 --> P3

    classDef input fill:#f1f3f6,stroke:#8a94a6,stroke-dasharray:5 4,color:#14181f
    classDef prep fill:#eef3ff,stroke:#2563eb,stroke-width:2px,color:#14181f
    classDef train fill:#f4efff,stroke:#7c3aed,stroke-width:2px,color:#14181f
    classDef ens fill:#7c3aed,stroke:#7c3aed,color:#ffffff
    classDef evals fill:#eafaf3,stroke:#059669,stroke-width:2px,color:#14181f
    class A input
    class B,C,D prep
    class E,F,M1,M2,M3 train
    class ENS ens
    class H,I,J,K evals
    style P1 fill:none,stroke:#2563eb,stroke-width:2px,color:#2563eb
    style P2 fill:none,stroke:#7c3aed,stroke-width:2px,color:#7c3aed
    style G fill:none,stroke:#7c3aed,stroke-dasharray:5 4,color:#7c3aed
    style P3 fill:none,stroke:#059669,stroke-width:2px,color:#059669
```

| Step | What happens | Key detail |
| --- | --- | --- |
| 1 | Clean and convert | Text answers become BDI 0–3 / FCRI 0–4. Personal info removed. Bad rows dropped. The negated answers of BDI items 6 and 7 were mis-scored until 2026-10-26; the saved models were retrained after the fix (see MODEL_FLOW.md). |
| 2 | Create labels | Binary: BDI total ≥ 11. Multiclass: Normal → Extreme (6 bands). |
| 3 | Feature matrix | BDI + FCRI items + encoded demographics. |
| 4 | Split | 80 / 20 stratified, shared by both tasks. |
| 5 | Scale | StandardScaler fitted on training data only. |
| 6 | Train | LR, RF, XGBoost tuned with GridSearchCV (5-fold, macro-F1, balanced weights), then combined in a soft-voting ensemble. |
| 7 | Feature importance | Top items, so a shorter mobile screening is possible. |
| 8 | Evaluate + threshold | Scored on the unseen test set. Binary cut-off tuned on training CV. |
| 9 | Export | Saved to `models/`, then loaded by `predict_depression.py` and the Streamlit app. |
