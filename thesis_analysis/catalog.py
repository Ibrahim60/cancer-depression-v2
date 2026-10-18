"""Single registry of every thesis figure and table.

Numbering, file names, captions and target thesis sections come from here, so
they stay stable no matter which stage produced the file. ``{subset}`` in a
key means one entry per analysis subset (suffix ``_all`` / ``_excl_generated``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Item:
    key: str
    chapter: int
    slug: str
    caption: str
    section: str


FIGURES: list[Item] = [
    # ---- Chapter 3: Methodology ------------------------------------------------
    Item("methodology_flowchart", 3, "methodology_flowchart",
         "Overall research methodology of the hybrid machine-learning framework.", "3.1 Research design"),
    Item("curation_flow", 3, "data_curation_flow",
         "Participant flow from the raw form export to the analysis datasets.", "3.2 Data collection and curation"),
    Item("bdi_mapping", 3, "bdi_severity_mapping",
         "Mapping of the BDI total score to the binary target and the six severity classes.", "3.3 Target definition"),
    Item("preprocessing", 3, "preprocessing_pipeline",
         "Data preprocessing pipeline; steps inside the dashed box are re-fitted on every training fold.", "3.4 Data preprocessing"),
    Item("hybrid_framework", 3, "hybrid_framework_architecture",
         "Architecture of the proposed hybrid framework (feature selection followed by ensemble learning).", "3.6 Proposed hybrid framework"),
    Item("nested_cv", 3, "nested_cross_validation",
         "Repeated stratified nested cross-validation scheme used for tuning and evaluation.", "3.7 Model validation"),
    Item("web_architecture", 3, "web_system_architecture",
         "Architecture of the web-based decision-support system.", "3.9 Web-based decision-support interface"),
    # ---- Chapter 4: Results ----------------------------------------------------
    Item("class_distribution", 4, "class_distributions",
         "Distribution of the binary depression target and the six BDI severity classes.", "4.1 Participant characteristics"),
    Item("score_distribution", 4, "score_distributions",
         "Distributions of BDI total and FCRI total scores (dashed lines: BDI severity cut-offs).", "4.1 Participant characteristics"),
    Item("fcri_by_severity", 4, "fcri_by_severity",
         "FCRI total and subscale scores across BDI severity classes.", "4.2 Statistical analysis"),
    Item("correlation_heatmap", 4, "correlation_heatmap",
         "Spearman correlations between BDI total, FCRI total and FCRI subscales.", "4.2 Statistical analysis"),
    Item("fcri_bdi_scatter", 4, "fcri_vs_bdi_scatter",
         "Relationship between FCRI total and BDI total scores.", "4.2 Statistical analysis"),
    Item("logistic_forest", 4, "logistic_regression_forest",
         "Adjusted odds ratios (95% CI) of being classified as depressed (multivariable logistic regression).", "4.2 Statistical analysis"),
    Item("featureset_comparison", 4, "featureset_comparison",
         "Macro F1 of every model on feature sets A-D (mean ± SD over 25 outer folds).", "4.3 Feature-set experiments and target leakage"),
    Item("model_fs_comparison", 4, "model_fs_comparison",
         "Macro F1 of models and feature-selection methods on feature set B.", "4.4 Hybrid framework performance"),
    Item("roc_curves", 4, "roc_curves_binary",
         "ROC curves of all models for binary depression detection (feature set B, pooled out-of-fold predictions).", "4.4 Hybrid framework performance"),
    Item("pr_curves", 4, "pr_curves_binary",
         "Precision-recall curves for binary depression detection (feature set B).", "4.4 Hybrid framework performance"),
    Item("confusion_binary", 4, "confusion_binary",
         "Confusion matrices (counts and row-normalised) of the hybrid model for binary detection.", "4.4 Hybrid framework performance"),
    Item("confusion_severity", 4, "confusion_severity",
         "Confusion matrices (counts and row-normalised) of the hybrid model for severity classification.", "4.4 Hybrid framework performance"),
    Item("cd_diagram", 4, "critical_difference",
         "Critical-difference diagrams (Friedman + Nemenyi) comparing models on feature set B.", "4.5 Statistical comparison of models"),
    Item("calibration", 4, "calibration_curve",
         "Calibration curve of the best binary model (out-of-fold predictions).", "4.6 Calibration, learning curves and errors"),
    Item("learning_curve", 4, "learning_curve",
         "Learning curves of the hybrid models.", "4.6 Calibration, learning curves and errors"),
    Item("error_analysis", 4, "error_analysis",
         "Misclassification rate by distance of the BDI total from the nearest class boundary.", "4.6 Calibration, learning curves and errors"),
    Item("imbalance_ablation", 4, "imbalance_and_ablation",
         "Effect of imbalance handling and of removing feature groups / preprocessing steps.", "4.7 Ablation study"),
    Item("fs_stability", 4, "fs_stability",
         "Selection frequency of the most frequently selected features across outer folds.", "4.8 Feature selection"),
    Item("shap_beeswarm", 4, "shap_beeswarm",
         "SHAP summary (beeswarm) plot of the hybrid binary model on feature set B.", "4.9 Explainability"),
    Item("shap_bar", 4, "shap_bar",
         "Mean absolute SHAP values of the top features (binary and severity models, feature set B).", "4.9 Explainability"),
    Item("shap_dependence", 4, "shap_dependence",
         "SHAP dependence plots of the five most important features (binary model, feature set B).", "4.9 Explainability"),
    Item("shap_waterfall", 4, "shap_waterfall",
         "SHAP waterfall plots explaining individual predictions for three participants.", "4.9 Explainability"),
    Item("shap_groups", 4, "shap_feature_groups",
         "Share of total mean |SHAP| contributed by each feature group (feature sets A and B).", "4.9 Explainability"),
]

TABLES: list[Item] = [
    Item("variables", 3, "variables_and_feature_groups",
         "Study variables, measurement scales and feature groups.", "3.2 Data collection and curation"),
    Item("feature_sets", 3, "feature_sets",
         "Feature sets used in the experiments.", "3.5 Experimental design"),
    Item("search_space", 3, "hyperparameter_search_space",
         "Models and hyper-parameter search spaces tuned in the inner cross-validation loop.", "3.6 Proposed hybrid framework"),
    Item("data_quality", 4, "data_quality",
         "Data-quality checks on the raw form export.", "4.1 Participant characteristics"),
    Item("characteristics_{subset}", 4, "participant_characteristics",
         "Participant characteristics overall and by depression status.", "4.1 Participant characteristics"),
    Item("reliability", 4, "reliability",
         "Internal consistency (Cronbach's alpha) of the BDI and the FCRI.", "4.1 Participant characteristics"),
    Item("bivariate_binary_{subset}", 4, "bivariate_tests_binary",
         "Association of FCRI, demographic and clinical variables with depression status.", "4.2 Statistical analysis"),
    Item("severity_tests_{subset}", 4, "severity_tests",
         "Differences across the six severity classes (Kruskal-Wallis / chi-square).", "4.2 Statistical analysis"),
    Item("dunn_{subset}", 4, "dunn_posthoc",
         "Dunn post-hoc comparisons of FCRI total between severity classes (Holm-adjusted).", "4.2 Statistical analysis"),
    Item("spearman_{subset}", 4, "spearman_fcri_bdi",
         "Spearman correlation of FCRI total and subscales with BDI total.", "4.2 Statistical analysis"),
    Item("logistic_{subset}", 4, "logistic_regression",
         "Univariate and multivariable logistic regression for depression (FCRI + demographic + clinical).", "4.2 Statistical analysis"),
    Item("ordinal_{subset}", 4, "ordinal_regression",
         "Ordinal logistic regression of severity class on FCRI and covariates.", "4.2 Statistical analysis"),
    Item("featureset_results", 4, "featureset_results",
         "Performance on feature sets A-D (mean ± SD across outer folds, 95% bootstrap CI).", "4.3 Feature-set experiments and target leakage"),
    Item("binary_results_{subset}", 4, "binary_results",
         "Binary detection performance on feature set B for all models and feature-selection methods.", "4.4 Hybrid framework performance"),
    Item("severity_results_{subset}", 4, "severity_results",
         "Severity classification performance on feature set B for all models and feature-selection methods.", "4.4 Hybrid framework performance"),
    Item("per_class_{subset}", 4, "per_class_report",
         "Per-class precision, recall and F1 of the hybrid severity model.", "4.4 Hybrid framework performance"),
    Item("significance_{subset}", 4, "significance_tests",
         "Statistical comparison of the hybrid model with comparators (Holm-adjusted).", "4.5 Statistical comparison of models"),
    Item("friedman", 4, "friedman_tests",
         "Friedman test and average ranks of the models.", "4.5 Statistical comparison of models"),
    Item("error_analysis", 4, "error_analysis",
         "Characteristics of correctly and incorrectly classified participants.", "4.6 Calibration, learning curves and errors"),
    Item("imbalance", 4, "imbalance_comparison",
         "Class weighting versus SMOTE versus no imbalance handling.", "4.7 Ablation study"),
    Item("ablation", 4, "ablation",
         "Ablation of feature groups and preprocessing steps (primary subset).", "4.7 Ablation study"),
    Item("sensitivity", 4, "sensitivity_analyses",
         "Sensitivity analyses: BDI >= 17 binary cut-off and five-class severity.", "4.7 Ablation study"),
    Item("fs_stability", 4, "fs_stability",
         "Feature-selection stability (Nogueira index) and number of selected features.", "4.8 Feature selection"),
    Item("selected_features_{subset}", 4, "selected_features",
         "Features most frequently selected by the hybrid model's selection method.", "4.8 Feature selection"),
    Item("key_factors_{subset}", 4, "key_factors_comparison",
         "Key factors identified by SHAP, statistical tests and feature-selection frequency.", "4.9 Explainability"),
]


def _expand(items: list[Item], subsets: list[str]) -> list[Item]:
    out = []
    for it in items:
        if "{subset}" in it.key:
            for s in subsets:
                out.append(Item(it.key.replace("{subset}", s), it.chapter, f"{it.slug}_{s}",
                                it.caption, it.section))
        else:
            out.append(it)
    return out


def numbered(items: list[Item], subsets: list[str], prefix: str) -> dict[str, dict]:
    """key -> {number, name, caption, section}; numbering restarts per chapter."""
    out, counter = {}, {}
    for it in _expand(items, subsets):
        counter[it.chapter] = counter.get(it.chapter, 0) + 1
        n = counter[it.chapter]
        out[it.key] = {"number": f"{it.chapter}.{n}",
                       "name": f"ch{it.chapter}_{prefix}{n:02d}_{it.slug}",
                       "caption": it.caption, "section": it.section, "chapter": it.chapter}
    return out


def figure(key: str, cfg: dict) -> dict:
    reg = numbered(FIGURES, cfg["run_subsets"], "fig")
    if key not in reg:
        raise KeyError(f"Figure key '{key}' not in catalog.FIGURES")
    return reg[key]


def table(key: str, cfg: dict) -> dict:
    reg = numbered(TABLES, cfg["run_subsets"], "tab")
    if key not in reg:
        raise KeyError(f"Table key '{key}' not in catalog.TABLES")
    return reg[key]


def save_table(df: pd.DataFrame, key: str, cfg: dict, tables_dir: Path, note: str = "") -> Path:
    """Write a catalogued table to CSV (+ a sidecar note used in the Word file)."""
    meta = table(key, cfg)
    path = tables_dir / f"{meta['name']}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    note_path = tables_dir / f"{meta['name']}.note.txt"
    if note:
        note_path.write_text(note, encoding="utf-8")
    elif note_path.exists():
        note_path.unlink()
    return path


def subset_label(cfg: dict, subset: str) -> str:
    return cfg["subsets"][subset]["label"]
