"""
===================================================================================
STANDALONE SHAP ANALYSIS SCRIPT
===================================================================================
Version: 1.0

This script runs SHAP analysis independently after the training pipeline has completed.
It loads the trained models and generates global and patient-level explanations.

Usage
-----
    # Run analysis for both binary and multiclass models
    python shap_analysis.py
    
    # Run analysis for binary classification only
    python shap_analysis.py --task binary
    
    # Run analysis for multiclass classification only
    python shap_analysis.py --task multiclass
    
    # Specify custom DPI for publication-quality output
    python shap_analysis.py --dpi 300
    
    # Specify custom sample size
    python shap_analysis.py --sample-size 100

Requirements
------------
    - Models must be trained first using depression_detection_pipeline.py
    - All required artefacts must be present in models/ directory
    - SHAP must be installed (pip install shap)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import json

try:
    from shap_analyzer import SHAPAnalyzer
except ImportError:
    print("Error: shap_analyzer.py not found or SHAP not installed.")
    print("Install SHAP with: pip install shap")
    sys.exit(1)


def load_artefacts(models_dir='models'):
    """
    Load all required artefacts for SHAP analysis.
    
    Returns
    -------
    dict
        Dictionary containing models, scaler, feature metadata, etc.
    """
    print(f"Loading artefacts from {models_dir}/...")
    
    artefacts = {}
    
    # Check required files exist
    required_files = [
        'binary_voting_ensemble_model.joblib',
        'multiclass_voting_ensemble_model.joblib',
        'binary_scaler.joblib',
        'label_encoders.joblib',
        'feature_metadata.json'
    ]
    
    missing = []
    for file in required_files:
        path = os.path.join(models_dir, file)
        if not os.path.exists(path):
            missing.append(file)
    
    if missing:
        print("Error: Missing required artefact(s):")
        for file in missing:
            print(f"  {file}")
        print("\nPlease run depression_detection_pipeline.py first to train the models.")
        sys.exit(1)
    
    # Load models
    print("  Loading binary models...")
    artefacts['binary_lr'] = joblib.load(os.path.join(models_dir, 'binary_logistic_regression_model.joblib'))
    artefacts['binary_rf'] = joblib.load(os.path.join(models_dir, 'binary_random_forest_model.joblib'))
    artefacts['binary_xgb'] = joblib.load(os.path.join(models_dir, 'binary_xgboost_model.joblib'))
    artefacts['binary_ensemble'] = joblib.load(os.path.join(models_dir, 'binary_voting_ensemble_model.joblib'))
    
    print("  Loading multiclass models...")
    artefacts['multi_lr'] = joblib.load(os.path.join(models_dir, 'multiclass_logistic_regression_model.joblib'))
    artefacts['multi_rf'] = joblib.load(os.path.join(models_dir, 'multiclass_random_forest_model.joblib'))
    artefacts['multi_xgb'] = joblib.load(os.path.join(models_dir, 'multiclass_xgboost_model.joblib'))
    artefacts['multi_ensemble'] = joblib.load(os.path.join(models_dir, 'multiclass_voting_ensemble_model.joblib'))
    
    # Load scaler and encoders
    print("  Loading scaler and encoders...")
    artefacts['scaler'] = joblib.load(os.path.join(models_dir, 'binary_scaler.joblib'))
    artefacts['label_encoders'] = joblib.load(os.path.join(models_dir, 'label_encoders.joblib'))
    
    # Load feature metadata
    print("  Loading feature metadata...")
    with open(os.path.join(models_dir, 'feature_metadata.json'), 'r') as f:
        meta = json.load(f)
    
    artefacts['feature_cols'] = meta['feature_cols']
    artefacts['feature_means'] = meta['feature_means']
    artefacts['binary_threshold'] = meta.get('binary_threshold', 0.5)
    
    print(f"  Loaded {len(artefacts['feature_cols'])} features")
    print("All artefacts loaded successfully.")
    
    return artefacts


def load_training_data(data_path='Data/cancer_psychology.csv'):
    """
    Load and preprocess training data for SHAP background dataset.
    
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, feature_cols) 
    """
    print(f"\nLoading training data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    # Import preprocessing classes from the pipeline
    try:
        from depression_detection_pipeline import (
            ClinicalDataPreprocessor,
            TargetVariableEngineer
        )
    except ImportError:
        print("Error: Cannot import preprocessing classes from depression_detection_pipeline.py")
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"  Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Preprocess
    preprocessor = ClinicalDataPreprocessor()
    df_proc, enc_features = preprocessor.fit_transform(df)
    
    # Create targets
    tve = TargetVariableEngineer(preprocessor.bdi_columns)
    df_fin = tve.create_targets(df_proc)
    
    # Build feature matrix
    feature_cols = [c for c in
                    preprocessor.bdi_columns + preprocessor.fcri_columns + enc_features
                    if c in df_fin.columns]
    X = df_fin[feature_cols].fillna(0)
    
    # Train/test split (same as pipeline)
    from sklearn.model_selection import train_test_split
    y_bin = df_fin['has_depression']
    
    X_train, X_test, _, _ = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Train: {X_train_scaled.shape[0]:,} samples")
    print(f"  Test:  {X_test_scaled.shape[0]:,} samples")
    
    return X_train_scaled, X_test_scaled, feature_cols


def run_binary_analysis(artefacts, X_train_scaled, X_test_scaled, feature_cols, 
                       output_dir, dpi, sample_size):
    """Run SHAP analysis for binary classification models."""
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION SHAP ANALYSIS")
    print("=" * 70)
    
    # Prepare models dictionary
    binary_models = {
        'logistic_regression': artefacts['binary_lr'],
        'random_forest': artefacts['binary_rf'],
        'xgboost': artefacts['binary_xgb'],
        'voting_ensemble': artefacts['binary_ensemble']
    }
    
    # Create SHAP analyzer
    shap_analyzer = SHAPAnalyzer(
        models=binary_models,
        feature_names=feature_cols,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        scaler=artefacts['scaler'],
        output_dir=output_dir,
        task='binary',
        class_names=['No Depression', 'Has Depression'],
        dpi=dpi,
        sample_size=sample_size
    )
    
    # Generate global explanations
    shap_analyzer.generate_global_explanations()
    
    # Verify prediction consistency
    shap_analyzer.verify_prediction_consistency('voting_ensemble')
    
    print(f"\nBinary SHAP analysis completed. Output saved to {output_dir}/")


def run_multiclass_analysis(artefacts, X_train_scaled, X_test_scaled, feature_cols,
                           output_dir, dpi, sample_size):
    """Run SHAP analysis for multiclass classification models."""
    print("\n" + "=" * 70)
    print("MULTICLASS CLASSIFICATION SHAP ANALYSIS")
    print("=" * 70)
    
    # Severity labels
    SEVERITY_LABELS = [
        'Normal',
        'Mild mood disturbance',
        'Borderline clinical depression',
        'Moderate depression',
        'Severe depression',
        'Extreme depression',
    ]
    
    # Prepare models dictionary
    multiclass_models = {
        'logistic_regression': artefacts['multi_lr'],
        'random_forest': artefacts['multi_rf'],
        'xgboost': artefacts['multi_xgb'],
        'voting_ensemble': artefacts['multi_ensemble']
    }
    
    # Create SHAP analyzer
    shap_analyzer = SHAPAnalyzer(
        models=multiclass_models,
        feature_names=feature_cols,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        scaler=artefacts['scaler'],
        output_dir=output_dir,
        task='multiclass',
        class_names=SEVERITY_LABELS,
        dpi=dpi,
        sample_size=sample_size
    )
    
    # Generate global explanations
    shap_analyzer.generate_global_explanations()
    
    # Verify prediction consistency
    shap_analyzer.verify_prediction_consistency('voting_ensemble')
    
    print(f"\nMulticlass SHAP analysis completed. Output saved to {output_dir}/")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate SHAP explanations for depression detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shap_analysis.py                    # Analyze both binary and multiclass
  python shap_analysis.py --task binary       # Binary classification only
  python shap_analysis.py --task multiclass   # Multiclass classification only
  python shap_analysis.py --dpi 300           # Publication-quality output
  python shap_analysis.py --sample-size 100   # Use 100 samples for SHAP
        """
    )
    
    parser.add_argument(
        '--task',
        choices=['binary', 'multiclass', 'both'],
        default='both',
        help='Which task to analyze (default: both)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved plots (default: 150, use 300 for publication quality)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=200,
        help='Maximum sample size for SHAP calculations (default: 200)'
    )
    
    parser.add_argument(
        '--models-dir',
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    parser.add_argument(
        '--data-path',
        default='Data/cancer_psychology.csv',
        help='Path to training data file (default: Data/cancer_psychology.csv)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SHAP ANALYSIS FOR DEPRESSION DETECTION MODELS")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"DPI: {args.dpi}")
    print(f"Sample size: {args.sample_size}")
    print(f"Models directory: {args.models_dir}")
    print(f"Data path: {args.data_path}")
    
    # Load artefacts
    artefacts = load_artefacts(args.models_dir)
    
    # Load training data
    X_train_scaled, X_test_scaled, feature_cols = load_training_data(args.data_path)
    
    # Run analysis based on task
    if args.task in ['binary', 'both']:
        binary_output_dir = os.path.join(args.models_dir, 'shap', 'binary')
        run_binary_analysis(
            artefacts, X_train_scaled, X_test_scaled, feature_cols,
            binary_output_dir, args.dpi, args.sample_size
        )
    
    if args.task in ['multiclass', 'both']:
        multiclass_output_dir = os.path.join(args.models_dir, 'shap', 'multiclass')
        run_multiclass_analysis(
            artefacts, X_train_scaled, X_test_scaled, feature_cols,
            multiclass_output_dir, args.dpi, args.sample_size
        )
    
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)
    print("Generated visualizations can be found in:")
    if args.task in ['binary', 'both']:
        print(f"  {os.path.join(args.models_dir, 'shap', 'binary')}/")
    if args.task in ['multiclass', 'both']:
        print(f"  {os.path.join(args.models_dir, 'shap', 'multiclass')}/")
    print("=" * 70)


if __name__ == '__main__':
    main()