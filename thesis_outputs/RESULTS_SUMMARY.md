# Thesis results summary

Generated 2026-10-19 16:18 from git commit `9bafade-dirty` with `python -m thesis_analysis.run_all`. Do not edit numbers by hand; re-run the pipeline.

## 1. Figures and tables

Files are in `thesis_outputs/figures/` (PNG 300 dpi + SVG) and `thesis_outputs/tables/` (CSV; all tables are also compiled in `thesis_tables.docx`). Captions are suggestions.

### Figures

| No. | File | Suggested caption | Thesis section |
|---|---|---|---|
| Figure 3.1 | `ch3_fig01_methodology_flowchart.png` | Overall research methodology of the hybrid machine-learning framework. | 3.1 Research design |
| Figure 3.2 | `ch3_fig02_data_curation_flow.png` | Participant flow from the raw form export to the analysis datasets. | 3.2 Data collection and curation |
| Figure 3.3 | `ch3_fig03_bdi_severity_mapping.png` | Mapping of the BDI total score to the binary target and the six severity classes. | 3.3 Target definition |
| Figure 3.4 | `ch3_fig04_preprocessing_pipeline.png` | Data preprocessing pipeline; steps inside the dashed box are re-fitted on every training fold. | 3.4 Data preprocessing |
| Figure 3.5 | `ch3_fig05_hybrid_framework_architecture.png` | Architecture of the proposed hybrid framework (feature selection followed by ensemble learning). | 3.6 Proposed hybrid framework |
| Figure 3.6 | `ch3_fig06_nested_cross_validation.png` | Repeated stratified nested cross-validation scheme used for tuning and evaluation. | 3.7 Model validation |
| Figure 3.7 | `ch3_fig07_web_system_architecture.png` | Architecture of the web-based decision-support system. | 3.9 Web-based decision-support interface |
| Figure 4.1 | `ch4_fig01_class_distributions.png` | Distribution of the binary depression target and the six BDI severity classes. | 4.1 Participant characteristics |
| Figure 4.2 | `ch4_fig02_score_distributions.png` | Distributions of BDI total and FCRI total scores (dashed lines: BDI severity cut-offs). | 4.1 Participant characteristics |
| Figure 4.3 | `ch4_fig03_fcri_by_severity.png` | FCRI total and subscale scores across BDI severity classes. | 4.2 Statistical analysis |
| Figure 4.4 | `ch4_fig04_correlation_heatmap.png` | Spearman correlations between BDI total, FCRI total and FCRI subscales. | 4.2 Statistical analysis |
| Figure 4.5 | `ch4_fig05_fcri_vs_bdi_scatter.png` | Relationship between FCRI total and BDI total scores. | 4.2 Statistical analysis |
| Figure 4.6 | `ch4_fig06_logistic_regression_forest.png` | Adjusted odds ratios (95% CI) of being classified as depressed (multivariable logistic regression). | 4.2 Statistical analysis |
| Figure 4.7 | `ch4_fig07_featureset_comparison.png` | Macro F1 of every model on feature sets A-D (mean ± SD over 25 outer folds). | 4.3 Feature-set experiments and target leakage |
| Figure 4.8 | `ch4_fig08_model_fs_comparison.png` | Macro F1 of models and feature-selection methods on feature set B. | 4.4 Hybrid framework performance |
| Figure 4.9 | `ch4_fig09_roc_curves_binary.png` | ROC curves of all models for binary depression detection (feature set B, pooled out-of-fold predictions). | 4.4 Hybrid framework performance |
| Figure 4.10 | `ch4_fig10_pr_curves_binary.png` | Precision-recall curves for binary depression detection (feature set B). | 4.4 Hybrid framework performance |
| Figure 4.11 | `ch4_fig11_confusion_binary.png` | Confusion matrices (counts and row-normalised) of the hybrid model for binary detection. | 4.4 Hybrid framework performance |
| Figure 4.12 | `ch4_fig12_confusion_severity.png` | Confusion matrices (counts and row-normalised) of the hybrid model for severity classification. | 4.4 Hybrid framework performance |
| Figure 4.13 | `ch4_fig13_critical_difference.png` | Critical-difference diagrams (Friedman + Nemenyi) comparing models on feature set B. | 4.5 Statistical comparison of models |
| Figure 4.14 | `ch4_fig14_calibration_curve.png` | Calibration curve of the best binary model (out-of-fold predictions). | 4.6 Calibration, learning curves and errors |
| Figure 4.15 | `ch4_fig15_learning_curve.png` | Learning curves of the hybrid models. | 4.6 Calibration, learning curves and errors |
| Figure 4.16 | `ch4_fig16_error_analysis.png` | Misclassification rate by distance of the BDI total from the nearest class boundary. | 4.6 Calibration, learning curves and errors |
| Figure 4.17 | `ch4_fig17_imbalance_and_ablation.png` | Effect of imbalance handling and of removing feature groups / preprocessing steps. | 4.7 Ablation study |
| Figure 4.18 | `ch4_fig18_fs_stability.png` | Selection frequency of the most frequently selected features across outer folds. | 4.8 Feature selection |
| Figure 4.19 | `ch4_fig19_shap_beeswarm.png` | SHAP summary (beeswarm) plot of the hybrid binary model on feature set B. | 4.9 Explainability |
| Figure 4.20 | `ch4_fig20_shap_bar.png` | Mean absolute SHAP values of the top features (binary and severity models, feature set B). | 4.9 Explainability |
| Figure 4.21 | `ch4_fig21_shap_dependence.png` | SHAP dependence plots of the five most important features (binary model, feature set B). | 4.9 Explainability |
| Figure 4.22 | `ch4_fig22_shap_waterfall.png` | SHAP waterfall plots explaining individual predictions for three participants. | 4.9 Explainability |
| Figure 4.23 | `ch4_fig23_shap_feature_groups.png` | Share of total mean |SHAP| contributed by each feature group (feature sets A and B). | 4.9 Explainability |

Appendix figures (20; `figures/appendix/`): secondary-subset versions of the figures above and per-class SHAP plots.

### Tables

| No. | File | Suggested caption | Thesis section |
|---|---|---|---|
| Table 3.1 | `ch3_tab01_variables_and_feature_groups.csv` | Study variables, measurement scales and feature groups. | 3.2 Data collection and curation |
| Table 3.2 | `ch3_tab02_feature_sets.csv` | Feature sets used in the experiments. | 3.5 Experimental design |
| Table 3.3 | `ch3_tab03_hyperparameter_search_space.csv` | Models and hyper-parameter search spaces tuned in the inner cross-validation loop. | 3.6 Proposed hybrid framework |
| Table 4.1 | `ch4_tab01_data_quality.csv` | Data-quality checks on the raw form export. | 4.1 Participant characteristics |
| Table 4.2 | `ch4_tab02_participant_characteristics_all.csv` | Participant characteristics overall and by depression status. (All rows) | 4.1 Participant characteristics |
| Table 4.3 | `ch4_tab03_participant_characteristics_excl_generated.csv` | Participant characteristics overall and by depression status. (Excluding generator-matched rows) | 4.1 Participant characteristics |
| Table 4.4 | `ch4_tab04_reliability.csv` | Internal consistency (Cronbach's alpha) of the BDI and the FCRI. | 4.1 Participant characteristics |
| Table 4.5 | `ch4_tab05_bivariate_tests_binary_all.csv` | Association of FCRI, demographic and clinical variables with depression status. (All rows) | 4.2 Statistical analysis |
| Table 4.6 | `ch4_tab06_bivariate_tests_binary_excl_generated.csv` | Association of FCRI, demographic and clinical variables with depression status. (Excluding generator-matched rows) | 4.2 Statistical analysis |
| Table 4.7 | `ch4_tab07_severity_tests_all.csv` | Differences across the six severity classes (Kruskal-Wallis / chi-square). (All rows) | 4.2 Statistical analysis |
| Table 4.8 | `ch4_tab08_severity_tests_excl_generated.csv` | Differences across the six severity classes (Kruskal-Wallis / chi-square). (Excluding generator-matched rows) | 4.2 Statistical analysis |
| Table 4.9 | `ch4_tab09_dunn_posthoc_all.csv` | Dunn post-hoc comparisons of FCRI total between severity classes (Holm-adjusted). (All rows) | 4.2 Statistical analysis |
| Table 4.10 | `ch4_tab10_dunn_posthoc_excl_generated.csv` | Dunn post-hoc comparisons of FCRI total between severity classes (Holm-adjusted). (Excluding generator-matched rows) | 4.2 Statistical analysis |
| Table 4.11 | `ch4_tab11_spearman_fcri_bdi_all.csv` | Spearman correlation of FCRI total and subscales with BDI total. (All rows) | 4.2 Statistical analysis |
| Table 4.12 | `ch4_tab12_spearman_fcri_bdi_excl_generated.csv` | Spearman correlation of FCRI total and subscales with BDI total. (Excluding generator-matched rows) | 4.2 Statistical analysis |
| Table 4.13 | `ch4_tab13_logistic_regression_all.csv` | Univariate and multivariable logistic regression for depression (FCRI + demographic + clinical). (All rows) | 4.2 Statistical analysis |
| Table 4.14 | `ch4_tab14_logistic_regression_excl_generated.csv` | Univariate and multivariable logistic regression for depression (FCRI + demographic + clinical). (Excluding generator-matched rows) | 4.2 Statistical analysis |
| Table 4.15 | `ch4_tab15_ordinal_regression_all.csv` | Ordinal logistic regression of severity class on FCRI and covariates. (All rows) | 4.2 Statistical analysis |
| Table 4.16 | `ch4_tab16_ordinal_regression_excl_generated.csv` | Ordinal logistic regression of severity class on FCRI and covariates. (Excluding generator-matched rows) | 4.2 Statistical analysis |
| Table 4.17 | `ch4_tab17_featureset_results.csv` | Performance on feature sets A-D (mean ± SD across outer folds, 95% bootstrap CI). | 4.3 Feature-set experiments and target leakage |
| Table 4.18 | `ch4_tab18_binary_results_all.csv` | Binary detection performance on feature set B for all models and feature-selection methods. (All rows) | 4.4 Hybrid framework performance |
| Table 4.19 | `ch4_tab19_binary_results_excl_generated.csv` | Binary detection performance on feature set B for all models and feature-selection methods. (Excluding generator-matched rows) | 4.4 Hybrid framework performance |
| Table 4.20 | `ch4_tab20_severity_results_all.csv` | Severity classification performance on feature set B for all models and feature-selection methods. (All rows) | 4.4 Hybrid framework performance |
| Table 4.21 | `ch4_tab21_severity_results_excl_generated.csv` | Severity classification performance on feature set B for all models and feature-selection methods. (Excluding generator-matched rows) | 4.4 Hybrid framework performance |
| Table 4.22 | `ch4_tab22_per_class_report_all.csv` | Per-class precision, recall and F1 of the hybrid severity model. (All rows) | 4.4 Hybrid framework performance |
| Table 4.23 | `ch4_tab23_per_class_report_excl_generated.csv` | Per-class precision, recall and F1 of the hybrid severity model. (Excluding generator-matched rows) | 4.4 Hybrid framework performance |
| Table 4.24 | `ch4_tab24_significance_tests_all.csv` | Statistical comparison of the hybrid model with comparators (Holm-adjusted). (All rows) | 4.5 Statistical comparison of models |
| Table 4.25 | `ch4_tab25_significance_tests_excl_generated.csv` | Statistical comparison of the hybrid model with comparators (Holm-adjusted). (Excluding generator-matched rows) | 4.5 Statistical comparison of models |
| Table 4.26 | `ch4_tab26_friedman_tests.csv` | Friedman test and average ranks of the models. | 4.5 Statistical comparison of models |
| Table 4.27 | `ch4_tab27_error_analysis.csv` | Characteristics of correctly and incorrectly classified participants. | 4.6 Calibration, learning curves and errors |
| Table 4.28 | `ch4_tab28_imbalance_comparison.csv` | Class weighting versus SMOTE versus no imbalance handling. | 4.7 Ablation study |
| Table 4.29 | `ch4_tab29_ablation.csv` | Ablation of feature groups and preprocessing steps (primary subset). | 4.7 Ablation study |
| Table 4.30 | `ch4_tab30_sensitivity_analyses.csv` | Sensitivity analyses: BDI ≥ 17 binary cut-off and five-class severity. | 4.7 Ablation study |
| Table 4.31 | `ch4_tab31_fs_stability.csv` | Feature-selection stability (Nogueira index) and number of selected features. | 4.8 Feature selection |
| Table 4.32 | `ch4_tab32_selected_features_all.csv` | Features most frequently selected by the hybrid model's selection method. (All rows) | 4.8 Feature selection |
| Table 4.33 | `ch4_tab33_selected_features_excl_generated.csv` | Features most frequently selected by the hybrid model's selection method. (Excluding generator-matched rows) | 4.8 Feature selection |
| Table 4.34 | `ch4_tab34_key_factors_comparison_all.csv` | Key factors identified by SHAP, statistical tests and feature-selection frequency. (All rows) | 4.9 Explainability |
| Table 4.35 | `ch4_tab35_key_factors_comparison_excl_generated.csv` | Key factors identified by SHAP, statistical tests and feature-selection frequency. (Excluding generator-matched rows) | 4.9 Explainability |

## 2. Key findings

Generated from the result files; re-run `--stage report` after any re-analysis. Values are mean ± SD over the outer folds of repeated nested cross-validation unless stated otherwise.

### 2.1 Data

- The raw export contained 1532 responses; after quality filters and exclusions 1530 participants remained.
- 561 rows (36.6%) exactly match the output of a mock-data generator found in the project's git history (name and all 21 BDI answers identical). All analyses are therefore reported for all rows and for the subset that excludes these rows. **The provenance of the dataset must be confirmed before results are used in the thesis.**
- 717 BDI item-6/7 answers were re-scored to 0 because the original keyword parser scored the negated statements ("I don't feel ...") as symptomatic.
- Consistency checks flagged 19 male participants reporting ovarian/cervical cancer, 5 female participants reporting prostate cancer, 68 participants under 18 years and 26 participants who gave the same answer to all 21 BDI items.

- All rows (n = 1530): 1411 depressed and 119 not depressed (BDI ≥ 11); severity classes: Normal 119, Mild mood disturbance 133, Borderline clinical depression 138, Moderate depression 424, Severe depression 317, Extreme depression 399.
- Excluding generator-matched rows (n = 969): 950 depressed and 19 not depressed (BDI ≥ 11); severity classes: Normal 19, Mild mood disturbance 47, Borderline clinical depression 58, Moderate depression 253, Severe depression 225, Extreme depression 367.

### 2.2 Statistical analysis

- All rows: FCRI total correlated with BDI total (Spearman ρ = 0.849, p < 0.001). 15 of 16 bivariate tests were significant after Holm correction. In the multivariable logistic model the adjusted odds ratio of depression per SD of FCRI total was 99.01 (95% CI 41.51–236.14, p < 0.001), with 3.61 minority-class events per model parameter (below the common minimum of 10, so the multivariable estimates are unstable).
- Excluding generator-matched rows: FCRI total correlated with BDI total (Spearman ρ = 0.802, p < 0.001). 10 of 16 bivariate tests were significant after Holm correction. In the multivariable logistic model the adjusted odds ratio of depression per SD of FCRI total was 5.22 (95% CI 2.38–11.45, p < 0.001), with 0.58 minority-class events per model parameter (below the common minimum of 10, so the multivariable estimates are unstable).

### 2.3 Feature sets and target leakage

- All rows, binary depression detection — best macro F1 per feature set: set A: 0.949 ± 0.025 (Soft Voting); set B: 0.882 ± 0.037 (Soft Voting); set C: 0.875 ± 0.034 (Soft Voting); set D: 0.634 ± 0.046 (Soft Voting).
- All rows, six-class severity classification — best macro F1 per feature set: set A: 0.861 ± 0.021 (Stacking (LR meta)); set B: 0.604 ± 0.029 (Stacking (LR meta)); set C: 0.612 ± 0.034 (Stacking (LR meta)); set D: 0.319 ± 0.022 (XGBoost).
- Excluding generator-matched rows, binary depression detection — best macro F1 per feature set: set A: 0.937 ± 0.077 (Soft Voting); set B: 0.626 ± 0.094 (Soft Voting); set C: 0.576 ± 0.091 (XGBoost); set D: 0.572 ± 0.069 (Random Forest).
- Excluding generator-matched rows, six-class severity classification — best macro F1 per feature set: set A: 0.796 ± 0.049 (Stacking (LR meta)); set B: 0.467 ± 0.041 (Stacking (LR meta)); set C: 0.455 ± 0.036 (Stacking (LR meta)); set D: 0.332 ± 0.026 (Soft Voting).

Feature set A contains the BDI items whose sum defines both targets, so its performance measures how well a model re-computes the label, not its ability to detect depression from other information. Feature set B (FCRI + demographic + clinical, no BDI) is the honest test of the thesis claim that fear of cancer recurrence can be used to detect depression.

### 2.4 Hybrid framework

- All rows, binary depression detection: the hybrid model (Boruta (RF) + Soft Voting) reached macro F1 0.882 ± 0.037 (95% CI 0.847–0.914); balanced accuracy 0.897 ± 0.038, sensitivity 0.977 ± 0.011, specificity 0.817 ± 0.075, ROC-AUC 0.956 ± 0.025, MCC 0.766 ± 0.072, Brier 0.033 ± 0.006. The best individual learner on set B was XGBoost with No FS (macro F1 0.876 ± 0.030).
- All rows, six-class severity classification: the hybrid model (Boruta (RF) + Stacking (LR meta)) reached macro F1 0.604 ± 0.029 (95% CI 0.571–0.636); balanced accuracy 0.608 ± 0.030, quadratic weighted κ 0.857 ± 0.029, macro OvR ROC-AUC 0.892 ± 0.015. The best individual learner on set B was SVM (RBF) with Boruta (RF) (macro F1 0.573 ± 0.024).
- Excluding generator-matched rows, binary depression detection: the hybrid model (Chi-square + Soft Voting) reached macro F1 0.626 ± 0.094 (95% CI 0.530–0.731); balanced accuracy 0.638 ± 0.106, sensitivity 0.985 ± 0.010, specificity 0.290 ± 0.214, ROC-AUC 0.907 ± 0.058, MCC 0.262 ± 0.195, Brier 0.025 ± 0.005. The best individual learner on set B was XGBoost with Chi-square (macro F1 0.603 ± 0.084).
- Excluding generator-matched rows, six-class severity classification: the hybrid model (L1 logistic regression + Stacking (LR meta)) reached macro F1 0.458 ± 0.030 (95% CI 0.413–0.499); balanced accuracy 0.490 ± 0.042, quadratic weighted κ 0.689 ± 0.047, macro OvR ROC-AUC 0.864 ± 0.022. The best individual learner on set B was Random Forest with No FS (macro F1 0.456 ± 0.051).

Paired comparisons (corrected resampled t-test, Holm-adjusted within task):

- All rows, binary depression detection: the hybrid was significantly better than 2 and significantly worse than 0 of the 5 other models trained with the same feature-selection method; versus the same ensemble without feature selection Δ macro F1 = +0.010 (p = 0.865).
  Feature set B vs A (same ensemble, no FS): 0.872 vs 0.949, Δ = -0.077 (p = 0.001).
  Feature set B vs C (same ensemble, no FS): 0.872 vs 0.875, Δ = -0.003 (p = 0.865).
  Feature set B vs D (same ensemble, no FS): 0.872 vs 0.634, Δ = +0.238 (p < 0.001).
- All rows, six-class severity classification: the hybrid was significantly better than 2 and significantly worse than 0 of the 5 other models trained with the same feature-selection method; versus the same ensemble without feature selection Δ macro F1 = +0.000 (p = 1.000).
  Feature set B vs A (same ensemble, no FS): 0.604 vs 0.861, Δ = -0.257 (p < 0.001).
  Feature set B vs C (same ensemble, no FS): 0.604 vs 0.612, Δ = -0.008 (p = 1.000).
  Feature set B vs D (same ensemble, no FS): 0.604 vs 0.319, Δ = +0.285 (p < 0.001).
- Excluding generator-matched rows, binary depression detection: the hybrid was significantly better than 0 and significantly worse than 0 of the 5 other models trained with the same feature-selection method; versus the same ensemble without feature selection Δ macro F1 = +0.032 (p = 1.000).
  Feature set B vs A (same ensemble, no FS): 0.594 vs 0.937, Δ = -0.343 (p < 0.001).
  Feature set B vs C (same ensemble, no FS): 0.594 vs 0.573, Δ = +0.021 (p = 1.000).
  Feature set B vs D (same ensemble, no FS): 0.594 vs 0.563, Δ = +0.031 (p = 1.000).
- Excluding generator-matched rows, six-class severity classification: the hybrid was significantly better than 2 and significantly worse than 0 of the 5 other models trained with the same feature-selection method; versus the same ensemble without feature selection Δ macro F1 = -0.009 (p = 1.000).
  Feature set B vs A (same ensemble, no FS): 0.467 vs 0.796, Δ = -0.329 (p < 0.001).
  Feature set B vs C (same ensemble, no FS): 0.467 vs 0.455, Δ = +0.012 (p = 1.000).
  Feature set B vs D (same ensemble, no FS): 0.467 vs 0.315, Δ = +0.152 (p < 0.001).

### 2.5 Ablation, imbalance handling and sensitivity analyses

- binary depression detection: 2 of 12 variants changed macro F1 significantly — Without FCRI items (set D) Δ = -0.250 (p < 0.001); Set D without medication variable Δ = -0.275 (p < 0.001).
- six-class severity classification: 2 of 12 variants changed macro F1 significantly — Without FCRI items (set D) Δ = -0.283 (p < 0.001); Set D without medication variable Δ = -0.308 (p < 0.001).
- All rows, binary target at BDI ≥ 17: best macro F1 0.843 ± 0.019 (Soft Voting).
- All rows, five-class severity: best macro F1 0.627 ± 0.021 (Stacking (LR meta)).
- Excluding generator-matched rows, binary target at BDI ≥ 17: best macro F1 0.672 ± 0.049 (Soft Voting).
- Excluding generator-matched rows, five-class severity: best macro F1 0.546 ± 0.044 (Stacking (LR meta)).

### 2.6 Calibration and learning curves

- Brier score of the hybrid binary model (pooled out-of-fold): 0.033.
- Binary detection: cross-validated macro F1 rose from 0.831 with 244 training participants to 0.879 with 1224 (training score 0.991).
- Severity (6 classes): cross-validated macro F1 rose from 0.494 with 244 training participants to 0.612 with 1224 (training score 0.791).

### 2.7 Key factors and the contribution of FCR

- All rows, binary depression detection — most frequently selected by Boruta (RF): FCRI 1: Television shows or newspaper articl… (1.00); FCRI 10: I am afraid of cancer recurrence (1.00); FCRI 11: I believe it is normal to be worried… (1.00); FCRI 13: I believe that I am cured and that t… (1.00); FCRI 14: In your opinion, are you at risk of… (1.00).
- All rows, six-class severity classification — most frequently selected by Boruta (RF): FCRI 1: Television shows or newspaper articl… (1.00); FCRI 10: I am afraid of cancer recurrence (1.00); FCRI 11: I believe it is normal to be worried… (1.00); FCRI 12: When I think about the possibility o… (1.00); FCRI 13: I believe that I am cured and that t… (1.00).
- Excluding generator-matched rows, binary depression detection — most frequently selected by Chi-square: Cancer status (1.00); Province (1.00); Time since diagnosis (1.00); FCRI 14: In your opinion, are you at risk of… (0.96); FCRI 1: Television shows or newspaper articl… (0.95).
- Excluding generator-matched rows, six-class severity classification — most frequently selected by L1 logistic regression: FCRI 1: Television shows or newspaper articl… (1.00); FCRI 14: In your opinion, are you at risk of… (1.00); FCRI 16: How much time per day do you spend t… (1.00); FCRI 13: I believe that I am cured and that t… (1.00); Province (1.00).
- All rows: with BDI items available (set A) BDI items account for 67% of total mean |SHAP| of the binary model; without BDI (set B) FCRI items account for 100%, demographic variables for 0% and clinical variables for 0%.
  Top SHAP features (binary, set B): FCRI 14: In your opinion, are you at risk of… (0.033); FCRI 9: I am worried or anxious about the po… (0.024); FCRI 1: Television shows or newspaper articl… (0.023); FCRI 16: How much time per day do you spend t… (0.022); FCRI 13: I believe that I am cured and that t… (0.019).
- Excluding generator-matched rows: with BDI items available (set A) BDI items account for 73% of total mean |SHAP| of the binary model; without BDI (set B) FCRI items account for 52%, demographic variables for 14% and clinical variables for 34%.
  Top SHAP features (binary, set B): Cancer status (0.023); Time since diagnosis (0.018); Province (0.015); FCRI 1: Television shows or newspaper articl… (0.009); FCRI 9: I am worried or anxious about the po… (0.008).


## 3. Draft methodology text (as implemented)

**Data.** Responses were collected with a Google Form containing the 21-item Beck Depression Inventory
(BDI; first-edition item wording, each item scored 0–3, total 0–63), the 42-item Fear of Cancer Recurrence
Inventory (FCRI; items scored 0–4, total 0–168, seven subscales), and demographic (gender, age group,
province) and clinical questions (cancer type, time since diagnosis, cancer status, current treatment and
current psychotropic medication). Names and timestamps were removed before analysis.

**Scoring and curation.** Text responses were converted to item scores with the scoring rules of the
application's preprocessing module; the parser was corrected so that the negated statements of BDI items 6
and 7 are scored 0. FCRI item 13 was reverse-scored (4 − raw). Respondents with all FCRI items equal to zero
or with more than 50% of questionnaire items missing were excluded. Remaining missing FCRI items were imputed
with the participant's own subscale mean (participants with fewer than half of a subscale answered would be
excluded). Free-text cancer types were harmonised and respondents reporting no cancer were excluded. Because
part of the dataset matched mock-generator output in the project history, every analysis was repeated on
the subset excluding those rows.

**Targets.** Depression status was defined as BDI total ≥ 11. Severity was defined by the
BDI interpretation bands Normal (0–10); Mild mood disturbance (11–16); Borderline clinical depression (17–20); Moderate depression (21–30); Severe depression (31–40); Extreme depression (41–63). Sensitivity analyses used BDI ≥ 17 and a five-class
scheme merging the two lowest classes.

**Statistical analysis.** Categorical variables were compared with χ² tests (Fisher's exact test for sparse
2 × 2 tables and a Monte-Carlo χ² test with 10,000 permutations for sparse larger tables; effect size Cramér's V).
Continuous scores were compared with Welch's t-test when both groups passed the Shapiro–Wilk test (α = 0.05)
and otherwise with the Mann–Whitney U test (effect size rank-biserial r). Differences across severity classes
used the Kruskal–Wallis test (ε²) or one-way ANOVA (η²) with Dunn's post-hoc test. Associations between FCRI
scores and BDI total used Spearman's ρ with Bonett–Wright confidence intervals. Univariate and multivariable
logistic regression (FCRI total plus all demographic and clinical variables; continuous predictors
standardised; categorical levels with fewer than 10 participants combined or excluded) and proportional-odds
ordinal regression of severity were fitted with statsmodels. P-values were adjusted with the Holm method
within each table. Internal consistency was assessed with Cronbach's α (bootstrap 95% CI).

**Machine-learning design.** Four feature sets were compared: A, all features (BDI, FCRI, demographic,
clinical); B, FCRI + demographic + clinical; C, FCRI only; D, demographic + clinical. All preprocessing was
fitted inside each training fold with scikit-learn/imbalanced-learn pipelines: one-hot encoding of categorical
variables, Min–Max scaling to [0, 1], optional SMOTE (k = 5), feature
selection and the classifier. Feature-selection methods were mutual information and χ² filters (k ∈ [10, 20, 30]),
recursive feature elimination with logistic regression (step 1, k ∈ [10, 20, 30]), L1-penalised logistic
regression (C ∈ [0.01, 0.05, 0.1, 0.5]; one-vs-rest for severity) and Boruta (20 iterations,
random forest with 150 trees, Bonferroni-corrected binomial test at α = 0.05).
Base learners were logistic regression, random forest, XGBoost and an RBF support-vector machine, with
balanced class weights (balanced sample weights for XGBoost). Hyper-parameter search spaces: Logistic Regression: C ∈ [0.001, 0.01, 0.1, 1, 10, 100]; Random Forest: n_estimators ∈ [300], max_depth ∈ [3, 5, 10, None], min_samples_leaf ∈ [1, 2, 5], max_features ∈ ['sqrt', 0.3]; XGBoost: n_estimators ∈ [100, 300], max_depth ∈ [2, 3, 5], learning_rate ∈ [0.03, 0.1, 0.3], subsample ∈ [0.8, 1.0], colsample_bytree ∈ [0.5, 0.8, 1.0]; SVM (RBF): C ∈ [0.1, 1, 10, 100], gamma ∈ ['scale', 0.01, 0.1].
Ensembles were soft voting with equal weights and stacking with a logistic-regression meta-learner trained on
3-fold out-of-fold probabilities, both built from the tuned base pipelines of the same outer fold.
The hybrid model was defined as the feature-selection method and ensemble with the highest mean macro F1 on
feature set B.

**Validation.** Models were evaluated with repeated stratified nested cross-validation: an outer loop of
5 folds × 5 repeats and an inner 3-fold stratified loop with randomised search
(10 candidates, scoring: macro F1). No separate hold-out set was used because the minority classes are
too small to give stable hold-out estimates; every participant is predicted out-of-fold once per repeat.
Ablation, imbalance and sensitivity analyses used 2 outer repeats. Random seed: 42.

**Evaluation.** Binary metrics: accuracy, balanced accuracy, precision, recall (sensitivity), specificity, F1,
macro F1, ROC-AUC, PR-AUC (for both classes), Matthews correlation coefficient and Brier score. Multiclass
metrics: accuracy, balanced accuracy, macro and weighted F1, Cohen's κ, quadratic weighted κ and macro
one-vs-rest ROC-AUC. Results are mean ± SD over outer folds with 95% percentile bootstrap confidence intervals
(1000 resamples of participants from the pooled out-of-fold predictions). Models were compared with
the Nadeau–Bengio corrected resampled t-test and the Wilcoxon signed-rank test on paired fold scores, McNemar's
test on paired out-of-fold predictions (Holm-adjusted), and the Friedman test with Nemenyi critical difference.

**Explainability.** The hybrid model of each task was refitted on the whole subset with the most frequent
nested-CV hyper-parameters and explained with permutation SHAP on the raw input variables
(background 50 participants, 300 explained participants stratified by class).

**Software.** Python 3.13.0; numpy 2.5.4, pandas 3.0.6, scipy 1.18.1,
scikit-learn 1.9.1, imbalanced-learn 0.14.2, xgboost 3.4.1,
statsmodels 0.15.0, scikit-posthocs 0.17.1, shap 0.54.1,
matplotlib 3.11.2, python-docx 1.2.0; diagrams with Graphviz.


## 4. How to read the results

- **Macro F1** averages the F1 score of every class with equal weight, so the small "not depressed" and
  "normal" classes count as much as the large ones. It is the primary metric because accuracy is dominated by
  the majority class (a model that labels everyone "depressed" already reaches the majority-class share).
- **Balanced accuracy** is the mean of per-class recall; 0.5 (binary) or 1/6 (severity) is chance level.
- **Sensitivity / specificity**: proportion of depressed / non-depressed participants correctly identified.
- **ROC-AUC** measures ranking quality independent of the threshold; **PR-AUC (not depressed)** is more
  informative for the rare class — compare it with the dashed line (its prevalence).
- **MCC** ranges from −1 to 1 (0 = chance) and is robust to imbalance. **Quadratic weighted κ** rewards
  predictions close to the true severity class (an Extreme case predicted as Severe is penalised less than one
  predicted as Normal).
- **Brier score** is the mean squared error of predicted probabilities (lower is better); the calibration curve
  shows whether, e.g., participants given a 0.8 probability are depressed about 80% of the time.
- **Mean ± SD** describes variation over the 25 outer folds; the **95% CI** describes uncertainty of the pooled
  estimate. Overlapping CIs do not by themselves prove equivalence — use the paired tests.
- **Corrected t-test / Wilcoxon / McNemar**: p < 0.05 after Holm adjustment means the difference is unlikely to
  be explained by fold-to-fold variation. The corrected t-test accounts for overlapping training sets.
- **Critical-difference diagram**: models joined by a thick bar are not significantly different (Nemenyi).
- **Feature-set figure**: the gap between set A and set B is the share of performance that comes from
  re-computing the BDI-based label; set B shows what FCR and context variables contribute on their own.
- **Selection frequency / Nogueira stability**: how consistently a feature is chosen across training folds;
  stability near 1 means the selected subset hardly changes.
- **SHAP**: each dot is a participant; positive values push the prediction towards depression. Group shares
  show how much of the model's total attribution comes from each block of variables. SHAP explains the model,
  not causal effects.
- **Confusion matrices** show counts from the first repetition of the outer CV (each participant once) and
  row-normalised recall; most severity errors fall in adjacent classes (see error analysis).


## 5. Limitations

- **Data provenance.** 561 of 1532 rows match mock-generator output in the repository history; the remaining rows could not be verified from the repository alone. Results on the two subsets differ and should be interpreted accordingly.
- **Small minority classes.** Not-depressed participants: All rows n = 119, Excluding generator-matched rows n = 19. Estimates for this class (specificity, PR-AUC) have wide confidence intervals.
- **Regression power.** Events per parameter in the multivariable logistic model: All rows 3.61, Excluding generator-matched rows 0.58; values below 10 indicate over-fitting risk and unstable odds ratios.
- **Target definition and leakage.** Both targets are deterministic functions of the BDI items; any model that sees BDI items (set A) re-computes the label. Only sets B–D test prediction from other information.
- **Construct overlap.** FCRI items on sadness (19) and mood (26) overlap with depressive symptoms; the ablation without these items shows their influence.
- **Self-report, cross-sectional, single source.** Both instruments are self-report questionnaires completed at one time point through one online form; common-method variance can inflate the FCRI–BDI association, and no external validation cohort was available.
- **Instrument version and cut-offs.** The BDI wording and severity bands correspond to the first-edition BDI interpretation (0–10 normal … 41–63 extreme); the binary cut-off of 11 classifies mild mood disturbance as depression. Results at BDI ≥ 17 are given as a sensitivity analysis.
- **Data inconsistencies.** Sex-specific cancer types reported by participants of the other gender and respondents under 18 were retained (see the data-quality table).
- **Explainability.** SHAP values were computed on a stratified sample with a model refitted on all data; they describe associations learned by the model and do not imply causation.

## 6. To decide or verify with your supervisor

1. Which rows are genuine participants (provenance of the generator-matched rows and of the remainder).
2. Which subset is primary for the thesis, and whether the other is reported as a sensitivity analysis.
3. Binary cut-off (BDI ≥ 11 vs ≥ 17) and whether six severity classes are reportable given the class sizes.
4. Exclusion of participants under 18 and handling of the gender/cancer-type inconsistencies.
5. Whether to describe the instrument as BDI-I/BDI-IA (item wording and cut-offs) rather than BDI-II.
6. Whether the Streamlit app should switch to an FCRI-based model (it currently uses BDI + FCRI models).
