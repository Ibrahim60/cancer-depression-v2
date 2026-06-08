"""
===================================================================================
HYBRID MACHINE LEARNING FRAMEWORK FOR DEPRESSION DETECTION IN CANCER PATIENTS
===================================================================================
Author: ML Pipeline for Cancer Psychology Assessment
Version: 1.0
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# SECTION 0: MOCK DATA GENERATION
# =============================================================================

def generate_mock_data(n_samples=500, output_path=None):
    """Generate synthetic mock data for testing the pipeline."""
    np.random.seed(RANDOM_STATE)
    
    genders = ['Male', 'Female']
    age_groups = ['Under 18', '18-30', '31-45', '46-60', 'Over 60']
    provinces = ['Punjab', 'Sindh', 'KPK', 'Balochistan', 'Gilgit Baltistan']
    cancer_types = ['Breast Cancer', 'Lung Cancer', 'Colorectal Cancer', 'Prostate Cancer']
    treatment_types = ['Chemotherapy', 'Radiotherapy', 'Surgery', 'Combination']
    cancer_statuses = ['In Treatment', 'Recovered / Survivor', 'In Remission']
    diagnosis_durations = ['Less than 6 months', '6 months - 1 year', '1 - 2 years', 'More than 2 years']
    
    # BDI Response texts (0-3 scoring)
    bdi_responses = {i: [f'response_{i}_0', f'response_{i}_1', f'response_{i}_2', f'response_{i}_3'] 
                     for i in range(1, 22)}
    
    fcri_frequency = ['Never', 'Rarely', 'Sometimes', 'Most of the time', 'All the time']
    fcri_intensity = ['Not at all', 'A little', 'Somewhat', 'A lot', 'A great deal']
    
    data = {
        'Timestamp': [datetime.now().strftime('%m/%d/%Y %H:%M:%S') for _ in range(n_samples)],
        'Name': [f'Patient_{str(i).zfill(5)}' for i in range(1, n_samples + 1)],
        'Gender': np.random.choice(genders, n_samples),
        '  Cancer Type  ': np.random.choice(cancer_types, n_samples),
        'Age Group': np.random.choice(age_groups, n_samples, p=[0.02, 0.15, 0.35, 0.35, 0.13]),
        'Povince': np.random.choice(provinces, n_samples),
        '  Duration Since Diagnosis  ': np.random.choice(diagnosis_durations, n_samples),
        '  Current Cancer Status  ': np.random.choice(cancer_statuses, n_samples),
        '  Current Treatment Type  ': np.random.choice(treatment_types, n_samples),
        'Are you currently taking any medication for depression, anxiety, stress, or other emotional/psychological problems?  ': 
            np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'How long have you been taking this medication?  ': 
            np.random.choice(['Less than 1 month', '1–6 months', '6–12 months', 'More than 1 year'], n_samples),
    }
    
    # BDI items (generate numeric directly for simplicity in mock)
    for i in range(1, 22):
        data[f'{i}.'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.25, 0.35, 0.25, 0.15])
    
    # FCRI items (42 items)
    fcri_col_names = [
        '1. Television shows or newspaper articles about cancer or illness',
        '2. An appointment with my doctor or other health professional',
        '3. Medical examinations (e.g. annual check-up, blood tests, X-rays)',
        '4. Conversations about cancer or illness in general',
        '5. Seeing or hearing about someone who is ill',
        '6. Going to a funeral or reading the obituary section of the paper',
        '7. When I feel unwell physically or when I am sick',
        '8. Generally, I avoid situations or things that make me think about the possibility of cancer recurrence',
        '9. I am worried or anxious about the possibility of cancer recurrence',
        '10. I am afraid of cancer recurrence',
        '11. I believe it is normal to be worried or anxious about the possibility of cancer recurrence',
        '12. When I think about the possibility of cancer recurrence, this triggers other unpleasant thoughts',
        '13. I believe that I am cured and that the cancer will not come back',
        '14. In your opinion, are you at risk of having a cancer recurrence?',
        '15. How often do you think about the possibility of cancer recurrence?',
        '16. How much time per day do you spend thinking about the possibility of cancer recurrence?',
        '17. How long have you been thinking about the possibility of cancer recurrence?',
        '18. Worry, fear or anxiety', '19. Sadness, discouragement or disappointment',
        '20. Frustration, anger or outrage', '21. Helplessness or resignation',
        '22. My social or leisure activities', '23. My work or everyday activities',
        '24. My relationships with my partner, my family, or those close to me',
        '25. My ability to make future plans or set life goals',
        '26. My state of mind or my mood', '27. My quality of life in general',
        '28. I feel that I worry excessively about the possibility of cancer recurrence',
        '29. Other people think that I worry excessively about the possibility of cancer recurrence',
        '30. I think that I worry more about the possibility of cancer recurrence than others',
        '31. I call my doctor or other health professional',
        '32. I go to the hospital or clinic for an examination',
        '33. I examine myself to see if I have any physical signs of cancer',
        '34. I try to distract myself', '35. I try not to think about it',
        '36. I pray, meditate or do relaxation', '37. I try to convince myself that everything will be fine',
        '38. I talk to someone about it', '39. I try to understand what is happening and deal with it',
        '40. I try to find a solution', '41. I try to replace this thought with a more pleasant one',
        '42. I tell myself stop it',
    ]
    
    for col in fcri_col_names:
        data[col] = np.random.choice([0, 1, 2, 3, 4], n_samples)
    
    df = pd.DataFrame(data)
    
    # Add mechanical responses (all 0s for FCRI) for ~2% of data
    mechanical_indices = np.random.choice(n_samples, int(n_samples * 0.02), replace=False)
    for idx in mechanical_indices:
        for col in fcri_col_names:
            df.loc[idx, col] = 0
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Mock data saved to: {output_path}")
    
    return df, fcri_col_names


# =============================================================================
# SECTION 1: DATA PREPROCESSING & CLEANING LAYER
# =============================================================================

class ClinicalDataPreprocessor:
    """Comprehensive data preprocessing class for clinical psychometric data."""
    
    FCRI_SUBSCALES = {
        'triggers': list(range(1, 9)),
        'severity': list(range(9, 18)),
        'psychological_distress': list(range(18, 22)),
        'functioning_impairments': list(range(22, 28)),
        'insight': list(range(28, 31)),
        'reassurance': list(range(31, 34)),
        'coping_strategies': list(range(34, 43))
    }
    
    FCRI_SCORE_MAPPINGS = {
        'frequency': {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Most of the time': 3, 'All the time': 4},
        'intensity': {'Not at all': 0, 'A little': 1, 'Somewhat': 2, 'A lot': 3, 'A great deal': 4},
    }
    
    def __init__(self):
        self.bdi_columns = []
        self.fcri_columns = []
        self.demographic_columns = {}
        self.label_encoders = {}
        
    def _identify_columns(self, df):
        """Identify BDI, FCRI, and demographic columns from dataframe."""
        columns = df.columns.tolist()
        
        # BDI columns: exactly "1." through "21."
        self.bdi_columns = [col for col in columns if col.strip() in [f'{i}.' for i in range(1, 22)]]
        
        # FCRI columns: start with "1. ", "2. ", etc. and have descriptive text
        self.fcri_columns = []
        for col in columns:
            col_stripped = col.strip()
            # Check if it starts with number followed by ". " and has more text (FCRI pattern)
            for i in range(1, 43):
                prefix = f'{i}. '
                if col_stripped.startswith(prefix) and len(col_stripped) > len(prefix) + 5:
                    self.fcri_columns.append(col)
                    break
        
        self.demographic_columns = {
            'gender': self._find_column(columns, ['gender']),
            'age_group': self._find_column(columns, ['age group', 'age']),
            'province': self._find_column(columns, ['province', 'povince']),
            'cancer_type': self._find_column(columns, ['cancer type']),
            'treatment_type': self._find_column(columns, ['treatment type', 'current treatment']),
            'cancer_status': self._find_column(columns, ['cancer status', 'current cancer status']),
            'medication': self._find_column(columns, ['medication for depression']),
            'medication_duration': self._find_column(columns, ['how long have you been taking']),
            'diagnosis_duration': self._find_column(columns, ['duration since diagnosis'])
        }
        print(f"Identified {len(self.bdi_columns)} BDI columns, {len(self.fcri_columns)} FCRI columns")
        
    def _find_column(self, columns, search_terms):
        for col in columns:
            for term in search_terms:
                if term.lower() in col.lower().strip():
                    return col
        return None
    
    def _convert_to_numeric(self, df):
        """Convert text responses to numeric if needed."""
        df_copy = df.copy()
        
        # BDI mapping: score based on position in response options (0-3)
        # Keywords checked from highest severity (3) to lowest (0)
        bdi_score_keywords = {
            3: ['so sad', 'unhappy that i can', 'future is hopeless', 'complete failure',
                'dissatisfied or bored', 'guilty all of the time', 'i am being punished',
                'hate myself', 'blame myself for everything', 'would kill myself if i had',
                "can't cry", 'used to be able to cry', 'irritated all the time',
                'lost all of my interest', "can't make decisions at all",
                'believe that i look ugly', "can't do any work", 'several hours earlier',
                'too tired to do anything', 'no appetite at all', 'lost more than fifteen',
                'so worried about my physical', 'lost interest in sex completely'],
            2: ['sad all the time', 'nothing to look forward', 'lot of failures',
                "don't get real satisfaction", 'quite guilty most', 'expect to be punished',
                'disgusted with myself', 'blame myself all the time', 'would like to kill myself',
                'cry all the time', 'quite annoyed', 'lost most of my interest',
                'greater difficulty in making decisions', 'permanent changes in my appearance',
                'push myself very hard', 'wake up 1-2 hours earlier', '1-2 hours earlier',
                'tired from doing almost', 'much worse now', 'lost more than ten pounds',
                'very worried about physical', 'almost no interest in sex'],
            1: ['i feel sad', 'feel discouraged', 'failed more than', "don't enjoy things",
                'feel guilty a good part', 'may be punished', 'disappointed in myself',
                'critical of myself', 'thoughts of killing myself, but', 'cry more now',
                'slightly more irritated', 'less interested in other people', 'put off making decisions',
                'worried that i am looking', 'extra effort to get started', "don't sleep as well",
                'get tired more easily', 'not as good as it used to be', 'lost more than five pounds',
                'worried about physical problems', 'less interested in sex than'],
            0: ['do not feel sad', 'not particularly discouraged', 'as much satisfaction',
                "don't feel particularly guilty", 'no more irritated', 'have not lost interest',
                'make decisions about as well', "don't feel that i look", 'can work about as well',
                'can sleep as well', "don't get more tired", 'no worse than usual',
                "haven't lost much weight", 'no more worried about my health',
                'have not noticed any recent change', "don't feel i am any worse",
                "don't have any thoughts of killing", "don't cry any more than usual"]
        }
        
        def map_bdi_response(response):
            if pd.isna(response):
                return np.nan
            response_lower = str(response).lower().strip().rstrip('.')
            # Check each score level from highest to lowest
            for score in [3, 2, 1, 0]:
                for keyword in bdi_score_keywords[score]:
                    if keyword in response_lower:
                        return score
            # Fallback: return NaN if no match
            return np.nan
        
        # Convert BDI columns (check for string-like dtypes)
        for col in self.bdi_columns:
            if col in df_copy.columns:
                if df_copy[col].dtype in ['object', 'string'] or str(df_copy[col].dtype) == 'str':
                    df_copy[col] = df_copy[col].apply(map_bdi_response)
        
        # FCRI mapping
        fcri_mapping = {
            **self.FCRI_SCORE_MAPPINGS['frequency'],
            **self.FCRI_SCORE_MAPPINGS['intensity'],
            # Additional variants
            'a little at risk': 1, 'somewhat at risk': 2, 'a lot at risk': 3, 'a great deal at risk': 4,
            'not at all at risk': 0, 'a few times a year': 1, 'a few times a month': 2,
            'a few times a week': 3, 'at least once a day': 4, "i don't think about it": 0,
            'a few seconds': 1, 'a few minutes': 2, 'a few hours': 3, 'several hours': 4,
            'a few weeks': 1, 'a few months': 2, 'a few years': 3, 'since my diagnosis': 4
        }
        
        def map_fcri_response(response):
            if pd.isna(response):
                return np.nan
            response_str = str(response).strip().lower()
            for key, value in fcri_mapping.items():
                if key.lower() == response_str:
                    return value
            return np.nan
        
        # Convert FCRI columns (check for string-like dtypes)
        for col in self.fcri_columns:
            if col in df_copy.columns:
                if df_copy[col].dtype in ['object', 'string'] or str(df_copy[col].dtype) == 'str':
                    df_copy[col] = df_copy[col].apply(map_fcri_response)
        
        return df_copy
    
    def _reverse_fcri_item_13(self, df):
        """Reverse scoring for FCRI Item 13 (scale 0-4, reversed = 4 - original)."""
        for col in self.fcri_columns:
            if col.strip().startswith('13.'):
                # Ensure numeric type before reversal
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = 4 - df[col]
                print(f"Reversed scoring for FCRI Item 13")
                break
        return df
    
    def _drop_mechanical_responses(self, df):
        """Drop rows where all FCRI responses are exactly 0."""
        if not self.fcri_columns:
            return df
        fcri_data = df[self.fcri_columns]
        mechanical_mask = (fcri_data == 0).all(axis=1)
        n_mechanical = mechanical_mask.sum()
        if n_mechanical > 0:
            print(f"Dropping {n_mechanical} mechanical response rows")
            df = df[~mechanical_mask].reset_index(drop=True)
        return df
    
    def _drop_excessive_missing(self, df, threshold=0.5):
        """Drop rows where more than threshold of questionnaire items are missing."""
        questionnaire_cols = self.bdi_columns + self.fcri_columns
        if not questionnaire_cols:
            return df
        missing_pct = df[questionnaire_cols].isna().mean(axis=1)
        excessive_mask = missing_pct > threshold
        n_dropped = excessive_mask.sum()
        if n_dropped > 0:
            print(f"Dropping {n_dropped} rows with >{threshold*100}% missing items")
            df = df[~excessive_mask].reset_index(drop=True)
        return df
    
    def _get_fcri_subscale_columns(self, subscale_name):
        """Get FCRI column names for a specific subscale."""
        item_numbers = self.FCRI_SUBSCALES.get(subscale_name, [])
        subscale_cols = []
        for item_num in item_numbers:
            for col in self.fcri_columns:
                if col.strip().startswith(f'{item_num}. ') or col.strip().startswith(f'{item_num}.'):
                    subscale_cols.append(col)
                    break
        return subscale_cols
    
    def _impute_missing_by_subscale(self, df):
        """Impute missing values using participant's subscale mean."""
        df_copy = df.copy()
        
        # Ensure all questionnaire columns are numeric
        for col in self.bdi_columns + self.fcri_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Impute BDI using participant's BDI mean
        if self.bdi_columns:
            bdi_data = df_copy[self.bdi_columns].astype(float)
            bdi_means = bdi_data.mean(axis=1)
            for col in self.bdi_columns:
                mask = df_copy[col].isna()
                if mask.any():
                    df_copy.loc[mask, col] = bdi_means[mask].round()
        
        # Impute FCRI using participant's subscale mean
        for subscale_name in self.FCRI_SUBSCALES:
            subscale_cols = self._get_fcri_subscale_columns(subscale_name)
            if not subscale_cols:
                continue
            # Get only existing columns
            existing_cols = [c for c in subscale_cols if c in df_copy.columns]
            if not existing_cols:
                continue
            subscale_data = df_copy[existing_cols].astype(float)
            subscale_means = subscale_data.mean(axis=1)
            for col in existing_cols:
                mask = df_copy[col].isna()
                if mask.any():
                    df_copy.loc[mask, col] = subscale_means[mask].round()
        return df_copy
    
    def _encode_categorical_variables(self, df):
        """Encode categorical demographic features."""
        df_copy = df.copy()
        encoded_features = []
        
        for var_name, col_name in self.demographic_columns.items():
            if col_name is None or col_name not in df_copy.columns:
                continue
            df_copy[col_name] = df_copy[col_name].fillna('Unknown').astype(str).str.strip()
            le = LabelEncoder()
            df_copy[f'{var_name}_encoded'] = le.fit_transform(df_copy[col_name])
            self.label_encoders[var_name] = le
            encoded_features.append(f'{var_name}_encoded')
        
        return df_copy, encoded_features
    
    def fit_transform(self, df):
        """Complete preprocessing pipeline."""
        print("=" * 60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        df_processed = df.copy()
        initial_rows = len(df_processed)
        print(f"Initial dataset size: {initial_rows} rows")
        
        self._identify_columns(df_processed)
        
        # Drop identifying information
        for col in ['Name', 'Timestamp']:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])
                print(f"Dropped column: {col}")
        
        # Convert to numeric FIRST
        df_processed = self._convert_to_numeric(df_processed)
        
        # Debug: Check conversion success
        if self.bdi_columns:
            bdi_valid = df_processed[self.bdi_columns].notna().mean().mean()
            print(f"BDI conversion success rate: {bdi_valid*100:.1f}%")
        if self.fcri_columns:
            fcri_valid = df_processed[self.fcri_columns].notna().mean().mean()
            print(f"FCRI conversion success rate: {fcri_valid*100:.1f}%")
        
        df_processed = self._reverse_fcri_item_13(df_processed)
        df_processed = self._drop_mechanical_responses(df_processed)
        df_processed = self._drop_excessive_missing(df_processed, threshold=0.5)
        df_processed = self._impute_missing_by_subscale(df_processed)
        df_processed, encoded_features = self._encode_categorical_variables(df_processed)
        
        print(f"Preprocessing complete: {initial_rows} -> {len(df_processed)} rows")
        print("=" * 60)
        
        return df_processed, encoded_features


# =============================================================================
# SECTION 2: TARGET VARIABLE ENGINEERING
# =============================================================================

class TargetVariableEngineer:
    """Engineer target variables from BDI scores."""
    
    SEVERITY_MAPPING = {
        'Normal': (0, 10), 'Mild mood disturbance': (11, 16),
        'Borderline clinical depression': (17, 20), 'Moderate depression': (21, 30),
        'Severe depression': (31, 40), 'Extreme depression': (41, 63)
    }
    
    SEVERITY_LABELS = ['Normal', 'Mild mood disturbance', 'Borderline clinical depression',
                       'Moderate depression', 'Severe depression', 'Extreme depression']
    
    def __init__(self, bdi_columns):
        self.bdi_columns = bdi_columns
        self.severity_encoder = LabelEncoder()
        self.severity_encoder.fit(self.SEVERITY_LABELS)
        
    def calculate_bdi_total(self, df):
        """Calculate total BDI score per patient."""
        bdi_data = df[self.bdi_columns].apply(pd.to_numeric, errors='coerce')
        return bdi_data.sum(axis=1)
    
    def map_to_severity(self, bdi_score):
        """Map BDI score to severity category."""
        if pd.isna(bdi_score):
            return 'Unknown'
        for category, (lower, upper) in self.SEVERITY_MAPPING.items():
            if lower <= bdi_score <= upper:
                return category
        return 'Unknown'
    
    def create_targets(self, df):
        """Create both target variables."""
        print("\n" + "=" * 60)
        print("TARGET VARIABLE ENGINEERING")
        print("=" * 60)
        
        df_copy = df.copy()
        df_copy['bdi_total'] = self.calculate_bdi_total(df_copy)
        
        print(f"BDI Score - Mean: {df_copy['bdi_total'].mean():.2f}, "
              f"Std: {df_copy['bdi_total'].std():.2f}, "
              f"Range: [{df_copy['bdi_total'].min():.0f}, {df_copy['bdi_total'].max():.0f}]")
        
        df_copy['depression_severity'] = df_copy['bdi_total'].apply(self.map_to_severity)
        df_copy['depression_severity_encoded'] = self.severity_encoder.transform(df_copy['depression_severity'])
        df_copy['has_depression'] = (df_copy['bdi_total'] >= 11).astype(int)
        
        print(f"\nBinary Target: {df_copy['has_depression'].value_counts().to_dict()}")
        print(f"Multi-class Target: {df_copy['depression_severity'].value_counts().to_dict()}")
        print("=" * 60)
        
        return df_copy


# =============================================================================
# SECTION 3: MODEL TRAINING & ENSEMBLE PIPELINE
# =============================================================================

class DepressionClassificationPipeline:
    """ML pipeline for depression classification with ensemble methods."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.scaler = StandardScaler()
        
    def _tune_model(self, model, param_grid, X, y, cv, model_name):
        """Tune hyperparameters using GridSearchCV."""
        print(f"\n  Tuning {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=0)
        grid_search.fit(X, y)
        print(f"    Best params: {grid_search.best_params_}")
        print(f"    Best CV F1: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_
    
    def train_classifier(self, X_train, y_train, task='binary'):
        """Train and tune models for classification."""
        print(f"\n{'=' * 60}")
        print(f"TRAINING {task.upper()} CLASSIFICATION MODELS")
        print("=" * 60)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        num_classes = len(np.unique(y_train))
        
        models = {}
        
        # Logistic Regression
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        if task == 'multiclass':
            lr.set_params(multi_class='multinomial')
        lr_params = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
        lr_best, _ = self._tune_model(lr, lr_params, X_train_scaled, y_train, cv, "Logistic Regression")
        models['logistic_regression'] = lr_best
        
        # Random Forest
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
        rf_best, _ = self._tune_model(rf, rf_params, X_train_scaled, y_train, cv, "Random Forest")
        models['random_forest'] = rf_best
        
        # XGBoost
        objective = 'binary:logistic' if task == 'binary' else 'multi:softmax'
        xgb = XGBClassifier(random_state=self.random_state, objective=objective, n_jobs=-1, 
                           use_label_encoder=False, eval_metric='logloss')
        xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
        xgb_best, _ = self._tune_model(xgb, xgb_params, X_train_scaled, y_train, cv, "XGBoost")
        models['xgboost'] = xgb_best
        
        # Voting Ensemble
        print("\n  Creating Voting Classifier Ensemble...")
        voting_clf = VotingClassifier(
            estimators=[('lr', lr_best), ('rf', rf_best), ('xgb', xgb_best)], voting='soft'
        )
        voting_clf.fit(X_train_scaled, y_train)
        models['voting_ensemble'] = voting_clf
        
        self.models[task] = models
        return models, X_train_scaled


# =============================================================================
# SECTION 4: FEATURE IMPORTANCE & MOBILE APP OPTIMIZATION
# =============================================================================

class FeatureImportanceAnalyzer:
    """Analyze feature importance for mobile app optimization."""
    
    def __init__(self, bdi_columns, fcri_columns, feature_names):
        self.bdi_columns = bdi_columns
        self.fcri_columns = fcri_columns
        self.feature_names = feature_names
        
    def extract_feature_importance(self, rf_model, xgb_model):
        """Extract and combine feature importance from RF and XGBoost."""
        rf_imp = rf_model.feature_importances_ / rf_model.feature_importances_.sum()
        xgb_imp = xgb_model.feature_importances_ / xgb_model.feature_importances_.sum()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'rf_importance': rf_imp,
            'xgb_importance': xgb_imp,
            'combined_importance': (rf_imp + xgb_imp) / 2
        }).sort_values('combined_importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def get_top_items(self, importance_df, top_n=5):
        """Get top N most important BDI and FCRI items."""
        top_bdi = importance_df[importance_df['feature'].isin(self.bdi_columns)].head(top_n)
        top_fcri = importance_df[importance_df['feature'].isin(self.fcri_columns)].head(top_n)
        
        print("\n" + "=" * 60)
        print("MOBILE APP OPTIMIZATION - TOP PREDICTIVE ITEMS")
        print("=" * 60)
        print(f"\nTop {top_n} BDI Items:")
        for _, row in top_bdi.iterrows():
            print(f"  {row['feature']} - Importance: {row['combined_importance']:.4f}")
        print(f"\nTop {top_n} FCRI Items:")
        for _, row in top_fcri.iterrows():
            feat = row['feature'][:50] + "..." if len(row['feature']) > 50 else row['feature']
            print(f"  {feat} - Importance: {row['combined_importance']:.4f}")
        print("=" * 60)
        
        return {'top_bdi': top_bdi['feature'].tolist(), 'top_fcri': top_fcri['feature'].tolist()}


# =============================================================================
# SECTION 5: EVALUATION, EXPORT & DEPLOYMENT
# =============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation and export utilities."""
    
    def __init__(self, output_dir='models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_models(self, models, X_test, y_test, scaler, task='binary', class_names=None):
        """Generate evaluation reports for all models."""
        print(f"\n{'=' * 60}")
        print(f"{task.upper()} CLASSIFICATION EVALUATION RESULTS")
        print("=" * 60)
        
        X_test_scaled = scaler.transform(X_test)
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            print(f"\n--- {name.upper()} ---")
            print(f"Accuracy:  {results[name]['accuracy']:.4f}")
            print(f"Precision: {results[name]['precision']:.4f}")
            print(f"Recall:    {results[name]['recall']:.4f}")
            print(f"F1-Score:  {results[name]['f1']:.4f}")
            
            if class_names:
                print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=class_names, zero_division=0)}")
        
        return results
    
    def plot_confusion_matrices(self, models, X_test, y_test, scaler, task='binary', class_names=None):
        """Plot confusion matrices for all models."""
        X_test_scaled = scaler.transform(X_test)
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, model) in zip(axes, models.items()):
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'{name}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{task}_confusion_matrices.png'), dpi=150)
        plt.close()
        print(f"Confusion matrices saved to {self.output_dir}/{task}_confusion_matrices.png")
    
    def export_models(self, models, scaler, preprocessor, task='binary'):
        """Export trained models and preprocessing pipelines."""
        for name, model in models.items():
            filepath = os.path.join(self.output_dir, f'{task}_{name}_model.joblib')
            joblib.dump(model, filepath)
            print(f"Saved: {filepath}")
        
        joblib.dump(scaler, os.path.join(self.output_dir, f'{task}_scaler.joblib'))
        joblib.dump(preprocessor.label_encoders, os.path.join(self.output_dir, 'label_encoders.joblib'))
        print(f"All models and preprocessors exported to {self.output_dir}/")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("HYBRID ML FRAMEWORK FOR DEPRESSION DETECTION IN CANCER PATIENTS")
    print("=" * 70)
    
    # Check for real data or generate mock data
    data_path = 'Data/cancer_psychology.csv'
    
    if os.path.exists(data_path):
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)
        fcri_columns = None  # Will be identified by preprocessor
    else:
        print("\nReal data not found. Generating mock data for demonstration...")
        df, fcri_columns = generate_mock_data(n_samples=500)
    
    print(f"Dataset shape: {df.shape}")
    
    # 1. DATA PREPROCESSING
    preprocessor = ClinicalDataPreprocessor()
    df_processed, encoded_features = preprocessor.fit_transform(df)
    
    # 2. TARGET VARIABLE ENGINEERING
    target_engineer = TargetVariableEngineer(preprocessor.bdi_columns)
    df_final = target_engineer.create_targets(df_processed)
    
    # 3. PREPARE FEATURES AND TARGETS
    feature_cols = preprocessor.bdi_columns + preprocessor.fcri_columns + encoded_features
    feature_cols = [c for c in feature_cols if c in df_final.columns]
    
    X = df_final[feature_cols].fillna(0)
    y_binary = df_final['has_depression']
    y_multi = df_final['depression_severity_encoded']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Train/Test Split (Stratified 80/20)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
    )
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X, y_multi, test_size=0.2, random_state=RANDOM_STATE, stratify=y_multi
    )
    
    # 4. MODEL TRAINING
    pipeline = DepressionClassificationPipeline(random_state=RANDOM_STATE)
    
    # Binary classification
    binary_models, X_train_bin_scaled = pipeline.train_classifier(X_train_bin, y_train_bin, task='binary')
    
    # Multi-class classification
    multi_models, X_train_multi_scaled = pipeline.train_classifier(X_train_multi, y_train_multi, task='multiclass')
    
    # 5. FEATURE IMPORTANCE ANALYSIS
    analyzer = FeatureImportanceAnalyzer(
        preprocessor.bdi_columns, preprocessor.fcri_columns, feature_cols
    )
    importance_df = analyzer.extract_feature_importance(
        binary_models['random_forest'], binary_models['xgboost']
    )
    top_items = analyzer.get_top_items(importance_df, top_n=5)
    
    # Save feature importance
    importance_df.to_csv('models/feature_importance.csv', index=False)
    print("Feature importance saved to models/feature_importance.csv")
    
    # 6. EVALUATION
    evaluator = ModelEvaluator(output_dir='models')
    
    # Binary evaluation
    binary_class_names = ['No Depression', 'Has Depression']
    binary_results = evaluator.evaluate_models(
        binary_models, X_test_bin, y_test_bin, pipeline.scaler, 
        task='binary', class_names=binary_class_names
    )
    evaluator.plot_confusion_matrices(
        binary_models, X_test_bin, y_test_bin, pipeline.scaler,
        task='binary', class_names=binary_class_names
    )
    
    # Multi-class evaluation
    multi_class_names = target_engineer.SEVERITY_LABELS
    # Filter to only present classes
    present_classes = sorted(y_test_multi.unique())
    multi_class_names_filtered = [multi_class_names[i] for i in present_classes]
    
    multi_results = evaluator.evaluate_models(
        multi_models, X_test_multi, y_test_multi, pipeline.scaler,
        task='multiclass', class_names=multi_class_names_filtered
    )
    evaluator.plot_confusion_matrices(
        multi_models, X_test_multi, y_test_multi, pipeline.scaler,
        task='multiclass', class_names=multi_class_names_filtered
    )
    
    # 7. EXPORT MODELS
    evaluator.export_models(binary_models, pipeline.scaler, preprocessor, task='binary')
    evaluator.export_models(multi_models, pipeline.scaler, preprocessor, task='multiclass')
    
    # SUMMARY
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\nBest Binary Classifier: {max(binary_results, key=lambda x: binary_results[x]['f1'])}")
    print(f"Best Multi-class Classifier: {max(multi_results, key=lambda x: multi_results[x]['f1'])}")
    print(f"\nTop 5 BDI items for mobile app: {top_items['top_bdi']}")
    print(f"Top 5 FCRI items for mobile app: {top_items['top_fcri']}")
    print(f"\nAll models and artifacts saved to: models/")
    print("=" * 70)
    
    return {
        'binary_models': binary_models,
        'multi_models': multi_models,
        'feature_importance': importance_df,
        'top_items': top_items,
        'preprocessor': preprocessor,
        'target_engineer': target_engineer
    }


if __name__ == "__main__":
    results = main()
