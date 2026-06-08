# Depression Detection System for Cancer Patients

A hybrid machine learning framework for detecting depression and its severity in cancer patients using BDI (Beck Depression Inventory) and FCRI (Fear of Cancer Recurrence Inventory) assessments.

## Features

- **Multi-model Ensemble**: Combines Logistic Regression, Random Forest, and XGBoost
- **Binary Classification**: Detects presence/absence of depression
- **Multi-class Classification**: Identifies severity levels (Normal, Mild, Borderline, Moderate, Severe, Extreme)
- **Interactive Screening Tool**: CLI-based assessment with personalized suggestions
- **Demographic-aware**: Adjusts predictions based on age and gender risk factors
- **Feature Importance Analysis**: Identifies top predictive BDI and FCRI items

## Project Structure

```
V2/
├── depression_detection_pipeline.py   # Main ML training pipeline
├── predict_depression.py              # Interactive prediction tool
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── Data/
│   ├── generate_mock_data.py          # Mock data generator
│   └── Psychological Assessment Form for Cancer Patients (BDI & FCRI).csv
└── models/                            # Trained models (generated)
    ├── binary_voting_ensemble_model.joblib
    ├── multiclass_voting_ensemble_model.joblib
    ├── feature_importance.csv
    └── ...
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

---

### Linux / macOS

#### 1. Clone the Repository

```bash
git clone https://github.com/Ibrahim60/cancer-depression-v2.git
cd cancer-depression-v2
```

#### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

#### 3. Activate Virtual Environment

```bash
source venv/bin/activate
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Generate Training Data

```bash
python Data/generate_mock_data.py
```

#### 6. Train the Models

```bash
python depression_detection_pipeline.py
```

#### 7. Run the Prediction Tool

```bash
python predict_depression.py
```

#### Deactivate Virtual Environment (when done)

```bash
deactivate
```

---

### Windows

#### 1. Clone the Repository

```cmd
git clone https://github.com/Ibrahim60/cancer-depression-v2.git
cd cancer-depression-v2
```

#### 2. Create Virtual Environment

```cmd
python -m venv venv
```

#### 3. Activate Virtual Environment

**Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

> **Note**: If you get a PowerShell execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### 4. Install Dependencies

```cmd
pip install -r requirements.txt
```

#### 5. Generate Training Data

```cmd
python Data\generate_mock_data.py
```

#### 6. Train the Models

```cmd
python depression_detection_pipeline.py
```

#### 7. Run the Prediction Tool

```cmd
python predict_depression.py
```

#### Deactivate Virtual Environment (when done)

```cmd
deactivate
```

---

## Usage

### Training Pipeline

The training pipeline (`depression_detection_pipeline.py`) performs:

1. **Data Preprocessing**
   - Removes identifying information (Name, Timestamp)
   - Handles missing values with subscale mean imputation
   - Converts text responses to numeric scores
   - Reverse scores FCRI Item 13

2. **Target Engineering**
   - Calculates BDI total score (0-63)
   - Creates binary target (depression present/absent)
   - Creates multi-class severity labels

3. **Model Training**
   - Trains Logistic Regression, Random Forest, XGBoost
   - Uses 5-fold stratified cross-validation
   - Creates soft voting ensemble

4. **Evaluation & Export**
   - Generates classification reports
   - Saves confusion matrices
   - Exports trained models to `models/` directory

**Expected Output:**
```
======================================================================
HYBRID ML FRAMEWORK FOR DEPRESSION DETECTION IN CANCER PATIENTS
======================================================================

Binary Classification:
  - Voting Ensemble Accuracy: ~99%
  
Multi-class Classification:
  - Voting Ensemble Accuracy: ~99%

Models saved to: models/
```

### Interactive Prediction Tool

The prediction tool (`predict_depression.py`) provides:

1. **Demographic Collection**
   - Gender (Male, Female, Other)
   - Age Group (Under 18, 18-30, 31-45, 46-60, Over 60)

2. **BDI Assessment** (10 key questions)
   - Failure feelings
   - Future outlook
   - Crying frequency
   - Decision making
   - Appetite changes
   - Emotional state
   - Self-perception
   - Appearance concerns
   - Fatigue levels
   - Health worries

3. **FCRI Assessment** (3 key questions)
   - Cancer recurrence worry
   - Risk perception
   - Intrusive thoughts

4. **Results Display**
   - Depression score (0-63)
   - Risk-adjusted score based on demographics
   - Severity classification
   - Personalized suggestions
   - Crisis helpline information (for severe cases)

**Sample Interaction:**
```
============================================================
DEPRESSION SCREENING TOOL FOR CANCER PATIENTS
============================================================

--- Basic Information ---

What is your gender?
  1. Male
  2. Female
  3. Other/Prefer not to say

Your answer (1-3): 2

What is your age group?
  1. Under 18
  2. 18-30
  3. 31-45
  4. 46-60
  5. Over 60

Your answer (1-5): 3

--- Part 1: Emotional Well-being Assessment ---
...

============================================================
ASSESSMENT RESULTS
============================================================

Patient Profile: Female, 31-45

Base Depression Score: 25.0/63
Risk-Adjusted Score: 33.0/63 (factor: 1.32x)

Depression Status: Detected
Severity Level: Moderate depression
```

---

## Depression Severity Levels

| Score Range | Severity | Description |
|-------------|----------|-------------|
| 0-10 | Normal | No significant depression |
| 11-16 | Mild | Mild mood disturbance |
| 17-20 | Borderline | Borderline clinical depression |
| 21-30 | Moderate | Moderate depression symptoms |
| 31-40 | Severe | Severe depression - seek help |
| 41-63 | Extreme | Extreme depression - urgent care needed |

---

## Risk Factors

The system adjusts scores based on demographic risk factors:

| Demographic | Risk Factor | Rationale |
|-------------|-------------|-----------|
| Female | 1.2x | Women have 2x higher depression prevalence |
| Male | 1.0x | Baseline |
| Under 18 | 1.1x | Adolescent vulnerability |
| 18-30 | 1.15x | Young adult cancer challenges |
| 31-45 | 1.1x | Family/career stress |
| 46-60 | 1.05x | Financial concerns |
| Over 60 | 1.0x | Baseline |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21.0 | Numerical computing |
| pandas | ≥1.3.0 | Data manipulation |
| scikit-learn | ≥1.0.0 | ML algorithms |
| xgboost | ≥1.5.0 | Gradient boosting |
| matplotlib | ≥3.4.0 | Plotting |
| seaborn | ≥0.11.0 | Statistical visualization |
| joblib | ≥1.0.0 | Model serialization |

---

## Troubleshooting

### "Command 'python' not found" (Linux)
Use `python3` instead of `python`:
```bash
python3 -m venv venv
python3 depression_detection_pipeline.py
```

### "externally-managed-environment" error (Linux)
Always use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### PowerShell execution policy error (Windows)
Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Models not found error
Ensure you've trained the models first:
```bash
python depression_detection_pipeline.py
```

### Data file not found
Generate the mock data first:
```bash
python Data/generate_mock_data.py
```

---

## Disclaimer

⚠️ **This tool is for screening purposes only and is NOT a clinical diagnosis.**

Please consult a qualified healthcare professional for proper evaluation and treatment recommendations. If you are experiencing thoughts of self-harm, please contact emergency services or a crisis helpline immediately.

**Pakistan Mental Health Helpline:** 0311-7786264

---

## License

This project is developed for academic/research purposes.

---

## Contributors

- Hina (FYP Project)
- Ibrahim60

---

## Acknowledgments

- Beck Depression Inventory (BDI-II)
- Fear of Cancer Recurrence Inventory (FCRI)
