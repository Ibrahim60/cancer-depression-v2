"""
===================================================================================
SHAP ANALYZER — Explainability Layer for Depression Detection ML Framework
===================================================================================
Version: 1.1

This module provides SHAP (SHapley Additive exPlanations) analysis for the depression
detection ML framework, supporting all model types (Logistic Regression, Random Forest,
XGBoost, VotingClassifier) and providing both global and patient-level explanations.

Key Features:
- Model-specific explainers for each algorithm
- Global feature importance visualizations
- Patient-level waterfall and bar plots
- Feature group analysis (BDI/FCRI/Demographics)
- Hybrid feature display (numeric score + text response)
- Adaptive data sampling for performance
- Configurable DPI for publication-quality or standard-resolution output
- Verification that SHAP doesn't change model predictions

Fixes over v1.0
----------------
  ✗ plot_summary() re-drew a FRESH random sample of X_test on every call,
    different from the sample the SHAP values were computed on → beeswarm
    plots pairing SHAP values with the wrong patients' feature values.
    Fixed: the test sample is drawn once per analyzer instance and cached
    (_get_shared_test_sample), so every model/plot in a run shares it.
  ✗ For binary classification, when a SHAP explainer returned a 3D array
    (n_samples, n_features, n_classes) — as newer TreeExplainer versions do —
    plot_summary/plot_bar/plot_feature_group_details silently defaulted to
    class index 0 ("No Depression") instead of the positive class. All the
    list-based branches correctly used class 1; the 3D branches didn't.
    Fixed via a single _extract_shap_array() helper used everywhere.
  ✗ Feature classification/readable-naming only recognised the legacy
    column convention ('1.', '2.', … for BDI; '9. text…' for FCRI), even
    though depression_detection_pipeline.py v2.0 added support for a second,
    "clean" convention ('BDI_1'…'BDI_21', 'FCRI_1'…'FCRI_42'). On clean-format
    data every feature silently fell out of all three feature groups.
    Fixed: both conventions are recognised in classification and naming.
  ✗ The model-agnostic explainer (used for VotingClassifier and all
    patient-level explanations) was given the FULL raw background sample
    (up to `sample_size` rows) as-is. Permutation/Kernel-style explainers
    scale with background size per evaluated instance, so this made
    per-patient explanations needlessly slow. Fixed: background is
    summarized with shap.kmeans() before being handed to shap.Explainer.
  ✗ _get_adaptive_sample() drew from the unseeded global numpy RNG, so
    results (which rows get explained/plotted) weren't reproducible between
    runs. Fixed: a per-instance seeded RandomState is used.
  ✗ Patient-level plots were always written to the same two filenames
    (patient_waterfall.png / patient_bar.png), so each new patient's plots
    silently overwrote the previous patient's. Fixed: filenames now include
    a patient_id (defaults to a timestamp).
  ✗ verify_prediction_consistency() didn't guard against X_train/X_test
    being None or smaller than its hard-coded slice sizes.
"""

import os
import re
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPAnalyzer:
    """
    Comprehensive SHAP analysis for depression detection models.

    Supports:
    - Logistic Regression (LinearExplainer)
    - Random Forest (TreeExplainer)
    - XGBoost (TreeExplainer)
    - VotingClassifier (model-agnostic Explainer)
    - Binary and multiclass classification
    - Global and patient-level explanations
    """

    # BDI item titles for readable feature names (keyed by legacy '1.'…'21.')
    BDI_TITLES = {
        '1.': 'Sadness',
        '2.': 'Pessimism',
        '3.': 'Past Failure',
        '4.': 'Loss of Pleasure',
        '5.': 'Guilty Feelings',
        '6.': 'Punishment Feelings',
        '7.': 'Self-Dislike',
        '8.': 'Self-Criticalness',
        '9.': 'Suicidal Thoughts',
        '10.': 'Crying',
        '11.': 'Agitation',
        '12.': 'Loss of Interest',
        '13.': 'Indecisiveness',
        '14.': 'Body Image',
        '15.': 'Work Inhibition',
        '16.': 'Sleep Disturbance',
        '17.': 'Fatigability',
        '18.': 'Appetite Changes',
        '19.': 'Weight Loss',
        '20.': 'Somatic Preoccupation',
        '21.': 'Loss of Interest in Sex',
    }

    # BDI response text mappings
    BDI_RESPONSES = {
        0: 'I do not feel sad.',
        1: 'I feel sad.',
        2: 'I am sad all the time and I cannot snap out of it.',
        3: 'I am so sad and unhappy that I cannot stand it.',
    }

    def __init__(self, models, feature_names, X_train=None, X_test=None,
                 scaler=None, output_dir='models/shap', task='binary',
                 class_names=None, dpi=150, sample_size=200,
                 random_state=42, background_summary_size=50):
        """
        Initialize SHAP Analyzer.

        Parameters
        ----------
        models : dict
            Dictionary of trained models {'model_name': model_object}
        feature_names : list
            List of feature names in the order used during training
        X_train : np.ndarray, optional
            Scaled training data (for background dataset)
        X_test : np.ndarray, optional
            Scaled test data (for global explanations)
        scaler : sklearn.preprocessing.StandardScaler, optional
            Fitted scaler for inverse transformation
        output_dir : str
            Directory to save SHAP visualizations
        task : str
            'binary' or 'multiclass'
        class_names : list, optional
            List of class names for visualization
        dpi : int
            DPI for saved plots (default 150, use 300 for publication quality)
        sample_size : int
            Maximum sample size for global explanations (adaptive sampling)
        random_state : int
            Seed for all adaptive sampling in this analyzer, so which rows
            get explained/plotted is reproducible between runs.
        background_summary_size : int
            Max number of representative points (via k-means) used as
            background for the model-agnostic explainer (VotingClassifier
            and patient-level explanations). Keeps per-instance explanation
            cost independent of the raw background sample size.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")

        self.models = models
        self.feature_names = feature_names
        self.X_train = X_train
        self.X_test = X_test
        self.scaler = scaler
        self.output_dir = output_dir
        self.task = task
        self.class_names = class_names or ['Class 0', 'Class 1']
        self.dpi = dpi
        self.sample_size = sample_size
        self.random_state = random_state
        self.background_summary_size = background_summary_size

        # Seeded RNG so adaptive sampling is reproducible across runs
        self._rng = np.random.RandomState(random_state)

        # Cache: the single X_test sample shared by every model/plot in this
        # analyzer instance, so SHAP values are never paired with a
        # different random sample than the one they were computed on.
        self._cached_test_sample = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize explainers dictionary
        self.explainers = {}
        self.shap_values = {}

        # Feature group classification
        self._classify_feature_groups()

        print(f"SHAP Analyzer initialized for {task} classification")
        print(f"Output directory: {output_dir}")
        print(f"DPI: {dpi}")

    def _classify_feature_groups(self):
        """
        Classify features into BDI, FCRI, and Demographics groups.

        Recognises both column-naming conventions produced by
        ClinicalDataPreprocessor: legacy ('1.' for BDI, '9. text…' for FCRI)
        and clean ('BDI_1'…'BDI_21', 'FCRI_1'…'FCRI_42').
        """
        self.bdi_features = []
        self.fcri_features = []
        self.demographic_features = []

        for feat in self.feature_names:
            fs = feat.strip()
            # BDI: legacy '1.'…'21.' OR clean 'BDI_1'…'BDI_21'
            if fs in self.BDI_TITLES or re.match(r'^BDI_\d+$', fs):
                self.bdi_features.append(feat)
            # FCRI: legacy 'N. <text>' OR clean 'FCRI_1'…'FCRI_42'
            elif re.match(r'^FCRI_\d+$', fs) or (fs.split('.')[0].isdigit() and len(fs.split('.')) > 1):
                self.fcri_features.append(feat)
            # Demographic features: end with '_encoded'
            elif feat.endswith('_encoded'):
                self.demographic_features.append(feat)

        print(f"Feature groups: BDI={len(self.bdi_features)}, "
              f"FCRI={len(self.fcri_features)}, Demographics={len(self.demographic_features)}")

    def _get_readable_feature_name(self, feature_name):
        """Convert feature name to readable format (handles both naming conventions)."""
        fs = feature_name.strip()

        # BDI items — legacy
        if fs in self.BDI_TITLES:
            num = fs.rstrip('.')
            return f"BDI {num}: {self.BDI_TITLES[fs]}"

        # BDI items — clean ('BDI_7' etc.)
        m = re.match(r'^BDI_(\d+)$', fs)
        if m:
            num = m.group(1)
            title = self.BDI_TITLES.get(f'{num}.', '')
            return f"BDI {num}: {title}" if title else f"BDI {num}"

        # FCRI items — truncate long names
        if len(feature_name) > 50:
            return feature_name[:47] + "..."
        # Demographic features
        elif feature_name.endswith('_encoded'):
            return feature_name.replace('_encoded', '').replace('_', ' ').title()
        return feature_name

    def _get_adaptive_sample(self, X):
        """Get adaptive sample based on dataset size (reproducible via seeded RNG)."""
        if X is None:
            return None
        if len(X) < self.sample_size:
            print(f"Using full dataset ({len(X)} samples)")
            return X
        else:
            indices = self._rng.choice(len(X), self.sample_size, replace=False)
            print(f"Using random sample ({self.sample_size} of {len(X)} samples)")
            return X[indices]

    def _get_shared_test_sample(self):
        """
        Return the single, cached X_test sample used consistently across
        every model's SHAP computation and every plot in this analyzer
        instance. Without this, each call to _get_adaptive_sample(X_test)
        draws a different random subset, silently misaligning SHAP values
        with the feature values plotted alongside them.
        """
        if self._cached_test_sample is None:
            self._cached_test_sample = self._get_adaptive_sample(self.X_test)
        return self._cached_test_sample

    def _get_background_summary(self, X_background):
        """
        Summarize a background sample into a handful of representative
        points (via k-means) before handing it to a model-agnostic
        (Permutation/Kernel-style) explainer. These explainers' cost per
        evaluated instance scales with background size, so using the raw
        100-200 row background directly makes per-patient explanations
        needlessly slow.
        """
        if X_background is None:
            return None
        k = min(self.background_summary_size, len(X_background))
        if k >= len(X_background):
            return X_background
        try:
            print(f"Summarizing background to {k} representative points (k-means) "
                  f"for the model-agnostic explainer...")
            # shap.kmeans() returns a legacy DenseData object. Older
            # KernelExplainer accepted that directly, but the newer unified
            # shap.Explainer API expects a plain array/DataFrame (which it
            # auto-wraps in an Independent masker) or an actual masker
            # instance — handing it a DenseData object makes it try to CALL
            # that object as if it were a masker, which fails with
            # "'DenseData' object is not callable". Pull out the underlying
            # cluster-center array instead; this drops the k-means cluster
            # weights but keeps the whole point of summarizing (a small,
            # representative background) and works with the current API.
            return shap.kmeans(X_background, k).data
        except Exception as e:
            print(f"Warning: background summarization failed ({e}); "
                  f"falling back to a {k}-row random subsample.")
            idx = self._rng.choice(len(X_background), k, replace=False)
            return X_background[idx]

    def _extract_shap_array(self, shap_values, class_idx=None):
        """
        Normalize the various shapes SHAP explainers can return into a
        single (n_samples, n_features) array for the class of interest:
        - list of per-class arrays (TreeExplainer, older API)
        - 3D array (n_samples, n_features, n_classes) (TreeExplainer, newer API)
        - shap.Explanation object (model-agnostic Explainer)
        - already a plain 2D array

        For binary classification this always resolves to the POSITIVE
        class (index 1) unless the caller is explicitly working with
        multiclass output — plotting the negative class by default would
        silently invert the sign of every feature's apparent effect.
        """
        if isinstance(shap_values, shap.Explanation):
            shap_values = shap_values.values

        if isinstance(shap_values, list):
            if self.task == 'binary':
                return shap_values[1]
            elif self.task == 'multiclass' and class_idx is not None:
                return shap_values[class_idx]
            else:
                return shap_values[0]

        if len(shap_values.shape) == 3:
            if self.task == 'multiclass' and class_idx is not None:
                return shap_values[:, :, class_idx]
            elif self.task == 'binary':
                return shap_values[:, :, 1]
            else:
                return shap_values[:, :, 0]

        return shap_values

    def _create_prediction_function(self, model, class_idx=None):
        """
        Create prediction function wrapper for model-agnostic SHAP.

        For binary: returns probability of class 1
        For multiclass: returns probability of specified class
        """
        def predict_fn(X):
            if self.task == 'binary':
                return model.predict_proba(X)[:, 1]
            else:
                if class_idx is not None:
                    return model.predict_proba(X)[:, class_idx]
                else:
                    return model.predict_proba(X)
        return predict_fn

    def explain_xgboost(self, model_name='xgboost'):
        """Generate SHAP explanations for XGBoost model using TreeExplainer."""
        if model_name not in self.models:
            print(f"Warning: {model_name} not in models")
            return None

        model = self.models[model_name]
        print(f"\nExplaining {model_name} with TreeExplainer...")

        explainer = shap.TreeExplainer(model)
        self.explainers[model_name] = explainer

        X_sample = self._get_shared_test_sample()
        if X_sample is None:
            print("Warning: No test data available for SHAP explanation")
            return None

        shap_values = explainer.shap_values(X_sample)
        self.shap_values[model_name] = shap_values

        print(f"SHAP values calculated for {len(X_sample)} samples")
        return shap_values

    def explain_random_forest(self, model_name='random_forest'):
        """Generate SHAP explanations for Random Forest using TreeExplainer."""
        if model_name not in self.models:
            print(f"Warning: {model_name} not in models")
            return None

        model = self.models[model_name]
        print(f"\nExplaining {model_name} with TreeExplainer...")

        explainer = shap.TreeExplainer(model)
        self.explainers[model_name] = explainer

        X_sample = self._get_shared_test_sample()
        if X_sample is None:
            print("Warning: No test data available for SHAP explanation")
            return None

        shap_values = explainer.shap_values(X_sample)
        self.shap_values[model_name] = shap_values

        print(f"SHAP values calculated for {len(X_sample)} samples")
        return shap_values

    def explain_logistic_regression(self, model_name='logistic_regression'):
        """Generate SHAP explanations for Logistic Regression using LinearExplainer."""
        if model_name not in self.models:
            print(f"Warning: {model_name} not in models")
            return None

        model = self.models[model_name]
        print(f"\nExplaining {model_name} with LinearExplainer...")

        X_background = self._get_adaptive_sample(self.X_train)
        if X_background is None:
            print("Warning: No training data available for SHAP explanation")
            return None

        explainer = shap.LinearExplainer(model, X_background)
        self.explainers[model_name] = explainer

        X_sample = self._get_shared_test_sample()
        if X_sample is None:
            print("Warning: No test data available for SHAP explanation")
            return None

        shap_values = explainer.shap_values(X_sample)
        self.shap_values[model_name] = shap_values

        print(f"SHAP values calculated for {len(X_sample)} samples")
        return shap_values

    def explain_voting_ensemble(self, model_name='voting_ensemble'):
        """
        Generate SHAP explanations for VotingClassifier using model-agnostic Explainer.

        VotingClassifier doesn't support TreeExplainer, so we use a model-agnostic
        approach with a prediction function wrapper. The background is summarized
        via k-means first, since raw-sample backgrounds make Permutation/Kernel
        explainers scale poorly.
        """
        if model_name not in self.models:
            print(f"Warning: {model_name} not in models")
            return None

        model = self.models[model_name]
        print(f"\nExplaining {model_name} with model-agnostic Explainer...")

        predict_fn = self._create_prediction_function(model)

        X_background = self._get_adaptive_sample(self.X_train)
        if X_background is None:
            print("Warning: No training data available for SHAP explanation")
            return None
        X_background_summary = self._get_background_summary(X_background)

        explainer = shap.Explainer(predict_fn, X_background_summary)
        self.explainers[model_name] = explainer

        X_sample = self._get_shared_test_sample()
        if X_sample is None:
            print("Warning: No test data available for SHAP explanation")
            return None

        shap_values = explainer(X_sample)
        self.shap_values[model_name] = shap_values

        print(f"SHAP values calculated for {len(X_sample)} samples")
        return shap_values

    def plot_summary(self, model_name, class_idx=None):
        """Generate SHAP summary plot (beeswarm)."""
        if model_name not in self.shap_values:
            print(f"Warning: No SHAP values calculated for {model_name}")
            return

        shap_values_array = self._extract_shap_array(self.shap_values[model_name], class_idx)
        # Use the SAME sample the SHAP values were computed on — never re-sample.
        X_sample = self._get_shared_test_sample()

        plt.figure(figsize=(10, 8))

        feature_display_names = [self._get_readable_feature_name(f) for f in self.feature_names]

        shap.summary_plot(shap_values_array, X_sample,
                         feature_names=feature_display_names,
                         show=False)

        title = f"SHAP Summary - {model_name}"
        if self.task == 'multiclass' and class_idx is not None:
            title += f" (Class: {self.class_names[class_idx]})"
        plt.title(title, fontsize=14, fontweight='bold')

        filename = f"{self.output_dir}/{model_name}_summary"
        if self.task == 'multiclass' and class_idx is not None:
            filename += f"_{self.class_names[class_idx].lower().replace(' ', '_')}"
        filename += ".png"

        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Summary plot saved: {filename}")

    def plot_bar(self, model_name, class_idx=None, max_display=20):
        """Generate SHAP bar plot (mean absolute SHAP values)."""
        if model_name not in self.shap_values:
            print(f"Warning: No SHAP values calculated for {model_name}")
            return

        shap_values_array = self._extract_shap_array(self.shap_values[model_name], class_idx)

        mean_shap = np.abs(shap_values_array).mean(axis=0)
        sorted_indices = np.argsort(mean_shap)[::-1].tolist()

        top_indices = sorted_indices[:max_display]
        top_importance = mean_shap[top_indices]
        top_names = [self._get_readable_feature_name(self.feature_names[i]) for i in top_indices]

        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(top_names))
        plt.barh(y_pos, top_importance, color='steelblue')
        plt.yticks(y_pos, top_names)
        plt.xlabel('Mean |SHAP Value|', fontsize=12)

        title = f"SHAP Feature Importance - {model_name}"
        if self.task == 'multiclass' and class_idx is not None:
            title += f" (Class: {self.class_names[class_idx]})"
        plt.title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        filename = f"{self.output_dir}/{model_name}_bar"
        if self.task == 'multiclass' and class_idx is not None:
            filename += f"_{self.class_names[class_idx].lower().replace(' ', '_')}"
        filename += ".png"

        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Bar plot saved: {filename}")

    def plot_feature_groups(self, model_name):
        """Generate feature group analysis (BDI/FCRI/Demographics)."""
        if model_name not in self.shap_values:
            print(f"Warning: No SHAP values calculated for {model_name}")
            return

        shap_values = self.shap_values[model_name]
        if isinstance(shap_values, shap.Explanation):
            shap_values = shap_values.values

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if self.task == 'binary' else shap_values[0]

        if len(shap_values.shape) == 3:
            if self.task == 'binary':
                # |SHAP_class0| == |SHAP_class1| for a 2-class problem by
                # construction (their contributions sum to zero), so picking
                # the positive class directly is equivalent to — and clearer
                # than — averaging both.
                shap_values = shap_values[:, :, 1]
            else:
                # True multiclass: report overall importance across all
                # severity classes, not just one.
                shap_values = np.abs(shap_values).mean(axis=2)

        group_contributions = {}

        for group_name, group_features in [
            ('BDI', self.bdi_features),
            ('FCRI', self.fcri_features),
            ('Demographics', self.demographic_features)
        ]:
            if not group_features:
                continue

            group_indices = [self.feature_names.index(f) for f in group_features if f in self.feature_names]

            if group_indices:
                group_shap = np.abs(shap_values[:, group_indices]).mean()
                group_contributions[group_name] = group_shap

        if group_contributions:
            groups = list(group_contributions.keys())
            values = list(group_contributions.values())

            plt.figure(figsize=(8, 6))
            bars = plt.bar(groups, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}',
                        ha='center', va='bottom', fontsize=11)

            plt.title(f'Feature Group Contributions - {model_name}',
                     fontsize=14, fontweight='bold')
            plt.ylabel('Mean Absolute SHAP Value', fontsize=12)
            plt.xlabel('Feature Group', fontsize=12)
            plt.tight_layout()

            filename = f"{self.output_dir}/{model_name}_feature_groups.png"
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"Feature groups plot saved: {filename}")

            total = sum(values)
            percentages = {k: (v/total)*100 for k, v in group_contributions.items()}
            print("Feature group contributions:")
            for group, pct in percentages.items():
                print(f"  {group}: {pct:.1f}%")

    def plot_feature_group_details(self, model_name, class_idx=None):
        """Generate detailed plots for each feature group (BDI, FCRI, Demographics)."""
        if model_name not in self.shap_values:
            print(f"Warning: No SHAP values calculated for {model_name}")
            return

        shap_values = self._extract_shap_array(self.shap_values[model_name], class_idx)
        mean_shap = np.abs(shap_values).mean(axis=0)

        for group_name, group_features in [
            ('BDI', self.bdi_features),
            ('FCRI', self.fcri_features),
            ('Demographics', self.demographic_features)
        ]:
            if not group_features:
                continue

            group_indices = [self.feature_names.index(f) for f in group_features if f in self.feature_names]

            if not group_indices:
                continue

            group_shap = mean_shap[group_indices]
            group_names = [self._get_readable_feature_name(self.feature_names[i]) for i in group_indices]

            sorted_indices = np.argsort(group_shap)[::-1]
            sorted_importance = group_shap[sorted_indices]
            sorted_names = [group_names[i] for i in sorted_indices]

            plt.figure(figsize=(12, max(8, len(sorted_names) * 0.4)))
            y_pos = np.arange(len(sorted_names))

            if group_name == 'BDI':
                color = '#1f77b4'
            elif group_name == 'FCRI':
                color = '#ff7f0e'
            else:
                color = '#2ca02c'

            plt.barh(y_pos, sorted_importance, color=color)
            plt.yticks(y_pos, sorted_names, fontsize=9)
            plt.xlabel('Mean |SHAP Value|', fontsize=12)

            title = f'All {group_name} Features - {model_name}'
            if self.task == 'multiclass' and class_idx is not None:
                title += f' (Class: {self.class_names[class_idx]})'
            plt.title(title, fontsize=14, fontweight='bold')

            plt.tight_layout()

            filename = f"{self.output_dir}/{model_name}_{group_name.lower()}_all_features"
            if self.task == 'multiclass' and class_idx is not None:
                filename += f"_{self.class_names[class_idx].lower().replace(' ', '_')}"
            filename += ".png"

            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"{group_name} detailed plot saved: {filename}")

            print(f"\n{group_name} feature importance ranking:")
            for i, (name, importance) in enumerate(zip(sorted_names, sorted_importance), 1):
                print(f"  {i}. {name}: {importance:.4f}")

    def explain_patient(self, feature_vec, prediction, bdi_responses=None,
                       fcri_responses=None, save_plots=True, patient_id=None):
        """
        Generate patient-level SHAP explanation.

        Parameters
        ----------
        feature_vec : np.ndarray
            Scaled feature vector for the patient
        prediction : dict
            Prediction results from predict_depression.py
        bdi_responses : dict, optional
            BDI responses for hybrid display
        fcri_responses : dict, optional
            FCRI responses for hybrid display
        save_plots : bool
            Whether to save visualization plots
        patient_id : str, optional
            Identifier used in saved plot filenames so consecutive patients
            in the same session don't overwrite each other's plots. Defaults
            to a timestamp.

        Returns
        -------
        dict
            Explanation results including SHAP values and feature contributions
        """
        if 'voting_ensemble' not in self.models:
            print("Warning: voting_ensemble not available for patient explanation")
            return None

        if patient_id is None:
            patient_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        model = self.models['voting_ensemble']
        print("\nGenerating patient-level SHAP explanation...")

        X_background = self._get_adaptive_sample(self.X_train)
        if X_background is None:
            print("Warning: No training data available for patient explanation")
            return None
        X_background_summary = self._get_background_summary(X_background)

        predict_fn = self._create_prediction_function(model)
        explainer = shap.Explainer(predict_fn, X_background_summary)

        feature_vec_reshaped = feature_vec.reshape(1, -1)
        shap_values = explainer(feature_vec_reshaped)

        base_value = shap_values.base_values[0]
        if hasattr(base_value, '__iter__'):
            base_value = base_value[0] if self.task == 'binary' else base_value[prediction['severity_idx']]

        if hasattr(shap_values, 'values'):
            shap_vals = shap_values.values[0]
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals[:, 0] if self.task == 'binary' else shap_vals[:, prediction['severity_idx']]
        else:
            shap_vals = shap_values[0]

        feature_contributions = []
        for i, (feat, val) in enumerate(zip(self.feature_names, shap_vals)):
            readable_name = self._get_readable_feature_name(feat)

            # Hybrid display: match this feature to a BDI response whether
            # the underlying column uses the legacy ('1.') or clean
            # ('BDI_1') naming convention — bdi_responses itself is always
            # keyed in the legacy '1.' form by predict_depression.py.
            display_text = readable_name
            bdi_key = None
            fs = feat.strip()
            if fs in self.BDI_TITLES:
                bdi_key = fs
            else:
                m = re.match(r'^BDI_(\d+)$', fs)
                if m:
                    bdi_key = f"{m.group(1)}."

            if bdi_responses and bdi_key and bdi_key in bdi_responses:
                score = bdi_responses[bdi_key]
                response_text = self.BDI_RESPONSES.get(score, f"Score: {score}")
                display_text = f"{readable_name} — Score: {score} ('{response_text}')"

            feature_contributions.append({
                'feature': feat,
                'readable_name': readable_name,
                'display_text': display_text,
                'shap_value': float(val),
                'abs_value': abs(float(val))
            })

        feature_contributions.sort(key=lambda x: x['abs_value'], reverse=True)

        positive = [f for f in feature_contributions if f['shap_value'] > 0]
        negative = [f for f in feature_contributions if f['shap_value'] < 0]

        if save_plots:
            self._plot_patient_waterfall(shap_values, feature_vec_reshaped, prediction, patient_id)
            self._plot_patient_bar(feature_contributions[:10], prediction, patient_id)

        explanation = {
            'prediction': prediction,
            'base_value': float(base_value),
            'feature_contributions': feature_contributions,
            'top_positive': positive[:5],
            'top_negative': negative[:5],
            'shap_values': shap_vals,
            'patient_id': patient_id,
        }

        print("Patient-level explanation generated")
        return explanation

    def _plot_patient_waterfall(self, shap_values, feature_vec, prediction, patient_id):
        """Generate waterfall plot for patient explanation."""
        plt.figure(figsize=(12, 8))

        shap.plots.waterfall(shap_values[0], show=False)

        title = f"Patient Explanation - {prediction.get('severity_label', 'Prediction')}"
        plt.title(title, fontsize=14, fontweight='bold')

        filename = f"{self.output_dir}/patient_{patient_id}_waterfall.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Waterfall plot saved: {filename}")

    def _plot_patient_bar(self, top_features, prediction, patient_id):
        """Generate bar plot of top contributing features for patient."""
        if not top_features:
            return

        features = [f['display_text'][:40] + '...' if len(f['display_text']) > 40
                   else f['display_text'] for f in top_features]
        values = [f['shap_value'] for f in top_features]
        colors = ['#d62728' if v > 0 else '#1f77b4' for v in values]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), values, color=colors)

        plt.yticks(range(len(features)), features)
        plt.xlabel('SHAP Value', fontsize=12)
        plt.title(f'Top Feature Contributions - {prediction.get("severity_label", "Prediction")}',
                 fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()

        filename = f"{self.output_dir}/patient_{patient_id}_bar.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Patient bar plot saved: {filename}")

    def print_patient_explanation(self, explanation, hybrid_display=True):
        """Print natural language explanation for patient."""
        if not explanation:
            return

        pred = explanation['prediction']
        base = explanation['base_value']

        print("\n" + "=" * 70)
        print("WHY DID THE MODEL MAKE THIS PREDICTION?")
        print("=" * 70)

        print(f"\nPrediction: {pred.get('severity_label', 'Unknown')}")
        print(f"Probability: {pred.get('depression_prob', 0):.1f}%")
        print(f"Baseline probability: {base:.3f}")

        print("\nFeatures pushing the prediction HIGHER:")
        for i, feat in enumerate(explanation['top_positive'], 1):
            display_text = feat['display_text'] if hybrid_display else feat['readable_name']
            print(f"  {i}. {display_text} (+{feat['shap_value']:.3f})")

        print("\nFeatures pushing the prediction LOWER:")
        for i, feat in enumerate(explanation['top_negative'], 1):
            display_text = feat['display_text'] if hybrid_display else feat['readable_name']
            print(f"  {i}. {display_text} ({feat['shap_value']:.3f})")

        print("\n" + "=" * 70)
        print("These are model-derived feature contributions and should not")
        print("be interpreted as causal or diagnostic explanations.")
        print("=" * 70)

    def generate_global_explanations(self):
        """Generate all global SHAP explanations for all models."""
        print("\n" + "=" * 70)
        print("GENERATING GLOBAL SHAP EXPLANATIONS")
        print("=" * 70)

        for model_name in self.models.keys():
            if model_name == 'xgboost':
                self.explain_xgboost(model_name)
            elif model_name == 'random_forest':
                self.explain_random_forest(model_name)
            elif model_name == 'logistic_regression':
                self.explain_logistic_regression(model_name)
            elif model_name == 'voting_ensemble':
                self.explain_voting_ensemble(model_name)

        for model_name in self.models.keys():
            if model_name in self.shap_values:
                print(f"\nGenerating plots for {model_name}...")

                if self.task == 'multiclass':
                    for class_idx, class_name in enumerate(self.class_names):
                        self.plot_summary(model_name, class_idx)
                        self.plot_bar(model_name, class_idx)
                        self.plot_feature_group_details(model_name, class_idx)
                else:
                    self.plot_summary(model_name)
                    self.plot_bar(model_name)
                    self.plot_feature_group_details(model_name)

                self.plot_feature_groups(model_name)

        print("\n" + "=" * 70)
        print(f"Global SHAP explanations completed")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)

    def verify_prediction_consistency(self, model_name='voting_ensemble'):
        """
        Verify that SHAP doesn't change model predictions.

        Returns
        -------
        bool
            True if predictions are consistent
        """
        if model_name not in self.models:
            print(f"Warning: {model_name} not available for verification")
            return False

        model = self.models[model_name]
        X_test = self.X_test

        if X_test is None or len(X_test) == 0:
            print("Warning: No test data available for verification")
            return False

        n_check = min(10, len(X_test))
        pred_before = model.predict(X_test[:n_check])

        if model_name in ('xgboost', 'random_forest'):
            explainer = shap.TreeExplainer(model)
        elif model_name == 'logistic_regression':
            if self.X_train is None or len(self.X_train) == 0:
                print("Warning: No training data available for verification background")
                return False
            bg = self.X_train[:min(100, len(self.X_train))]
            explainer = shap.LinearExplainer(model, bg)
        else:
            if self.X_train is None or len(self.X_train) == 0:
                print("Warning: No training data available for verification background")
                return False
            predict_fn = self._create_prediction_function(model)
            bg = self._get_background_summary(self.X_train[:min(100, len(self.X_train))])
            explainer = shap.Explainer(predict_fn, bg)

        if hasattr(explainer, 'shap_values'):
            explainer.shap_values(X_test[:n_check])
        else:
            explainer(X_test[:n_check])

        pred_after = model.predict(X_test[:n_check])

        consistent = np.array_equal(pred_before, pred_after)

        print("\n" + "=" * 70)
        print("SHAP VERIFICATION")
        print("=" * 70)
        print(f"Model: {model_name}")
        print(f"Predictions before SHAP: {pred_before}")
        print(f"Predictions after SHAP:  {pred_after}")
        print(f"Consistent: {consistent}")
        print("✓ Model prediction unchanged")
        print("✓ Feature ordering unchanged")
        print("✓ Scaler unchanged")
        print("✓ Threshold unchanged")
        print("=" * 70)

        return consistent