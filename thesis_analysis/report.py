"""Thesis-writing support: thesis_tables.docx and thesis_outputs/RESULTS_SUMMARY.md.

Every number in RESULTS_SUMMARY.md is read from result files produced by the
earlier stages; the text around the numbers is templated. Sentences that
interpret results are phrased conditionally on the numbers (e.g. whether a
difference is significant after Holm correction), never hard-coded.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import catalog
from .experiments import MODEL_LABELS
from .feature_selection import FS_LABELS
from .utils import Paths, get_logger, git_commit, library_versions
from .tables import build_docx

TASK_NAME = {"binary": "binary depression detection", "severity6": "six-class severity classification"}


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _ms(d: dict, nd: int = 3) -> str:
    return f"{d['mean']:.{nd}f} ± {d['sd']:.{nd}f}"


def _ci(d: dict) -> str:
    lo, hi = d.get("ci", [np.nan, np.nan])
    return "" if lo is None or np.isnan(lo) else f" (95% CI {lo:.3f}–{hi:.3f})"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ----------------------------------------------------------------------------
def figure_table_lists(cfg: dict[str, Any], paths: Paths) -> str:
    lines = ["## 1. Figures and tables", "",
             "Files are in `thesis_outputs/figures/` (PNG 300 dpi + SVG) and `thesis_outputs/tables/` "
             "(CSV; all tables are also compiled in `thesis_tables.docx`). Captions are suggestions.", "",
             "### Figures", "", "| No. | File | Suggested caption | Thesis section |", "|---|---|---|---|"]
    figs = catalog.numbered(catalog.FIGURES, cfg["run_subsets"], "fig")
    missing = []
    for key, m in figs.items():
        exists = (paths.figures / f"{m['name']}.png").exists()
        if not exists:
            missing.append(m["name"])
            continue
        lines.append(f"| Figure {m['number']} | `{m['name']}.png` | {m['caption']} | {m['section']} |")
    app = sorted(p.name for p in paths.appendix.glob("*.png"))
    lines += ["", f"Appendix figures ({len(app)}; `figures/appendix/`): secondary-subset versions of the "
              "figures above and per-class SHAP plots.", ""]
    lines += ["### Tables", "", "| No. | File | Suggested caption | Thesis section |", "|---|---|---|---|"]
    tabs = catalog.numbered(catalog.TABLES, cfg["run_subsets"], "tab")
    for key, m in tabs.items():
        if not (paths.tables / f"{m['name']}.csv").exists():
            missing.append(m["name"])
            continue
        sub = next((s for s in cfg["run_subsets"] if key.endswith(f"_{s}")), None)
        cap = m["caption"] + (f" ({catalog.subset_label(cfg, sub)})" if sub else "")
        lines.append(f"| Table {m['number']} | `{m['name']}.csv` | {cap} | {m['section']} |")
    if missing:
        lines += ["", "**Missing outputs** (stage not run or failed): " + ", ".join(f"`{x}`" for x in missing)]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
def key_findings(cfg: dict[str, Any], paths: Paths) -> str:
    R = paths.results
    kr = _load(R / "key_results.json")
    stats = _load(R / "statistics_summary.json")
    counts = _load(R / "class_counts.json")
    quality = _load(R / "data_quality.json")
    fs = _load(R / "fs_summary.json")
    shap = _load(R / "shap_summary.json")
    pm = cfg["cv"]["primary_metric"]
    out = ["## 2. Key findings", "",
           "Generated from the result files; re-run `--stage report` after any re-analysis. Values are "
           "mean ± SD over the outer folds of repeated nested cross-validation unless stated otherwise.", ""]
    if quality:
        q = quality["quality"]
        out += ["### 2.1 Data", "",
                f"- The raw export contained {q['raw_rows']} responses; after quality filters and exclusions "
                f"{counts[cfg['primary_subset']]['n'] if counts else '–'} participants remained.",
                f"- {q['generator_matched_rows']} rows ({100 * q['generator_matched_rows'] / q['raw_rows']:.1f}%) "
                "exactly match the output of a mock-data generator found in the project's git history "
                "(name and all 21 BDI answers identical). All analyses are therefore reported for all rows and "
                "for the subset that excludes these rows. **The provenance of the dataset must be confirmed "
                "before results are used in the thesis.**",
                f"- {q['bdi_items_6_7_rescored']} BDI item-6/7 answers were re-scored to 0 because the original "
                "keyword parser scored the negated statements (\"I don't feel ...\") as symptomatic.",
                f"- Consistency checks flagged {q['male_with_female_specific_cancer']} male participants reporting "
                f"ovarian/cervical cancer, {q['female_with_prostate_cancer']} female participants reporting prostate "
                f"cancer, {q['under_18']} participants under 18 years and {q['bdi_straightlining']} participants who "
                "gave the same answer to all 21 BDI items.", ""]
    if counts:
        for sub, c in counts.items():
            b = {str(k): v for k, v in c["binary"].items()}
            sev = ", ".join(f"{k} {v}" for k, v in c["severity6"].items())
            out.append(f"- {catalog.subset_label(cfg, sub)} (n = {c['n']}): {b['1']} depressed and {b['0']} not "
                       f"depressed (BDI ≥ {cfg['targets']['binary_cutoff']}); severity classes: {sev}.")
        out.append("")
    if stats:
        out += ["### 2.2 Statistical analysis", ""]
        for sub, s in stats.items():
            rho = s["spearman_fcri_total_bdi"]
            lr = s["logistic"]
            lrdf = pd.read_csv(R / f"logistic_raw_{sub}.csv") if (R / f"logistic_raw_{sub}.csv").exists() else None
            adj = ""
            if lrdf is not None:
                m = lrdf[(lrdf.Model == "Multivariable") & (lrdf.key == "fcri_total")]
                if len(m) and not np.isnan(m.iloc[0]["OR"]):
                    r = m.iloc[0]
                    adj = (f" In the multivariable logistic model the adjusted odds ratio of depression per SD of "
                           f"FCRI total was {r.OR:.2f} (95% CI {r['CI low']:.2f}–{r['CI high']:.2f}, {_p(r.p)}), "
                           f"with {lr.get('events_per_variable')} minority-class events per model parameter"
                           + (" (below the common minimum of 10, so the multivariable estimates are unstable)"
                              if lr.get("events_per_variable", 99) < 10 else "") + ".")
            out.append(f"- {catalog.subset_label(cfg, sub)}: FCRI total correlated with BDI total "
                       f"(Spearman ρ = {rho['rho']:.3f}, {_p(rho['p'])}). "
                       f"{s['n_significant_bivariate_holm']} of {s['n_bivariate_tests']} bivariate tests were "
                       f"significant after Holm correction.{adj}")
        out.append("")
    if kr:
        out += ["### 2.3 Feature sets and target leakage", ""]
        for sub in cfg["run_subsets"]:
            if sub not in kr:
                continue
            for task in ["binary", "severity6"]:
                bf = kr[sub][task]["best_by_feature_set"]
                parts = [f"set {k}: {v['mean']:.3f} ± {v['sd']:.3f} ({MODEL_LABELS[v['model']]})" for k, v in bf.items()]
                out.append(f"- {catalog.subset_label(cfg, sub)}, {TASK_NAME[task]} — best macro F1 per feature set: "
                           + "; ".join(parts) + ".")
        out += ["", "Feature set A contains the BDI items whose sum defines both targets, so its performance "
                "measures how well a model re-computes the label, not its ability to detect depression from "
                "other information. Feature set B (FCRI + demographic + clinical, no BDI) is the honest test of "
                "the thesis claim that fear of cancer recurrence can be used to detect depression.", ""]
        out += ["### 2.4 Hybrid framework", ""]
        for sub in cfg["run_subsets"]:
            if sub not in kr:
                continue
            for task in ["binary", "severity6"]:
                h = kr[sub][task]["hybrid"]
                m = h["metrics"]
                bi = kr[sub][task]["best_individual_on_key_set"]
                if task == "binary":
                    extra = (f"balanced accuracy {_ms(m['balanced_accuracy'])}, sensitivity {_ms(m['recall'])}, "
                             f"specificity {_ms(m['specificity'])}, ROC-AUC {_ms(m['roc_auc'])}, "
                             f"MCC {_ms(m['mcc'])}, Brier {_ms(m['brier'])}")
                else:
                    extra = (f"balanced accuracy {_ms(m['balanced_accuracy'])}, quadratic weighted κ "
                             f"{_ms(m['qwk'])}, macro OvR ROC-AUC {_ms(m['roc_auc_ovr'])}")
                out.append(f"- {catalog.subset_label(cfg, sub)}, {TASK_NAME[task]}: the hybrid model "
                           f"({FS_LABELS[h['fs']]} + {MODEL_LABELS[h['model']]}) reached macro F1 "
                           f"{_ms(m[pm])}{_ci(m[pm])}; {extra}. The best individual learner on set "
                           f"{cfg['key_feature_set']} was {MODEL_LABELS[bi['model']]} with {FS_LABELS[bi['fs']]} "
                           f"(macro F1 {bi['mean']:.3f} ± {bi['sd']:.3f}).")
        sig = pd.DataFrame(kr.get("significance", []))
        if len(sig):
            out += ["", "Paired comparisons (corrected resampled t-test, Holm-adjusted within task):", ""]
            alpha = cfg["cv"]["alpha"]
            for (sub, task), g in sig.groupby(["subset", "task"]):
                ra, rb = g.a.str.rsplit(":", n=1).str[0], g.b.str.rsplit(":", n=1).str[0]
                same = g[ra == rb]
                better = same[(same["diff"] > 0) & (same.p_corrected_t_holm < alpha)]
                worse = same[(same["diff"] < 0) & (same.p_corrected_t_holm < alpha)]
                txt = (f"- {catalog.subset_label(cfg, sub)}, {TASK_NAME[task]}: the hybrid was significantly "
                       f"better than {len(better)} and significantly worse than {len(worse)} of the "
                       f"{len(same)} other models trained with the same feature-selection method")
                nofs = g[(ra != rb) & (rb.str.split("__").str[0] == "featuresets") & (ra.str.split("__").str[0] == "fs")]
                for r in nofs.itertuples():
                    txt += (f"; versus the same ensemble without feature selection Δ macro F1 = {r.diff:+.3f} "
                            f"({_p(r.p_corrected_t_holm)})")
                out.append(txt + ".")
                fsets = g[(ra != rb) & (ra.str.split("__").str[0] == "featuresets")]
                for r in fsets.itertuples():
                    other = r.b.split("__")[3]
                    out.append(f"  Feature set {cfg['key_feature_set']} vs {other} (same ensemble, no FS): "
                               f"{r.mean_a:.3f} vs {r.mean_b:.3f}, Δ = {r.diff:+.3f} ({_p(r.p_corrected_t_holm)}).")
        out.append("")
        abl = pd.DataFrame(kr.get("ablation", []))
        if len(abl):
            out += ["### 2.5 Ablation, imbalance handling and sensitivity analyses", ""]
            for task, g in abl.groupby("task"):
                sig_ = g[g.p_holm < cfg["cv"]["alpha"]]
                desc = "; ".join(f"{r.label} Δ = {r.delta:+.3f} ({_p(r.p_holm)})" for r in sig_.itertuples())
                out.append(f"- {TASK_NAME[task]}: {len(sig_)} of {len(g)} variants changed macro F1 significantly"
                           + (f" — {desc}." if desc else "."))
            for sub in cfg["run_subsets"]:
                for t in ("binary17", "severity5"):
                    k = kr.get(sub, {}).get(f"sensitivity_{t}")
                    if k:
                        lab = "binary target at BDI ≥ 17" if t == "binary17" else "five-class severity"
                        out.append(f"- {catalog.subset_label(cfg, sub)}, {lab}: best macro F1 "
                                   f"{k['mean']:.3f} ± {k['sd']:.3f} ({MODEL_LABELS[k['best_model']]}).")
            out.append("")
        cal = kr.get(cfg["primary_subset"], {}).get("binary", {}).get("calibration")
        lc = kr.get("learning_curve")
        if cal or lc:
            out += ["### 2.6 Calibration and learning curves", ""]
            if cal:
                out.append(f"- Brier score of the hybrid binary model (pooled out-of-fold): {cal['brier']:.3f}.")
            if lc:
                for k, v in lc.items():
                    out.append(f"- {k}: cross-validated macro F1 rose from {v['test_mean'][0]:.3f} with "
                               f"{v['sizes'][0]} training participants to {v['test_mean'][-1]:.3f} with "
                               f"{v['sizes'][-1]} (training score {v['train_mean'][-1]:.3f}).")
            out.append("")
    if fs or shap:
        out += ["### 2.7 Key factors and the contribution of FCR", ""]
        if fs:
            for sub, d in fs.items():
                for task, v in d.items():
                    top = list(v["top10"].items())[:5]
                    out.append(f"- {catalog.subset_label(cfg, sub)}, {TASK_NAME[task]} — most frequently selected "
                               f"by {FS_LABELS[v['method']]}: " + "; ".join(f"{k} ({x:.2f})" for k, x in top) + ".")
        if shap:
            for sub, d in shap.items():
                gs = d["group_shares"]
                bkey = f"Binary, set {cfg['key_feature_set']}"
                akey = "Binary, set A"
                if bkey in gs and akey in gs:
                    out.append(f"- {catalog.subset_label(cfg, sub)}: with BDI items available (set A) BDI items "
                               f"account for {100 * gs[akey].get('bdi', 0):.0f}% of total mean |SHAP| of the binary "
                               f"model; without BDI (set {cfg['key_feature_set']}) FCRI items account for "
                               f"{100 * gs[bkey].get('fcri', 0):.0f}%, demographic variables for "
                               f"{100 * gs[bkey].get('demographic', 0):.0f}% and clinical variables for "
                               f"{100 * gs[bkey].get('clinical', 0):.0f}%.")
                top = list(d["top_binary"].items())[:5]
                out.append(f"  Top SHAP features (binary, set {cfg['key_feature_set']}): "
                           + "; ".join(f"{k} ({v:.3f})" for k, v in top) + ".")
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------
def methodology_text(cfg: dict[str, Any], paths: Paths) -> str:
    v = library_versions()
    cv = cfg["cv"]
    g = cfg["models"]["grids"]
    fsc = cfg["feature_selection"]
    tg = cfg["targets"]
    bands = "; ".join(f"{a} ({lo}–{hi})" for a, lo, hi in tg["severity_bounds"])
    grids = "; ".join(f"{MODEL_LABELS[m]}: " + ", ".join(f"{k} ∈ {vals}" for k, vals in g[m].items())
                      for m in cfg["models"]["base"])
    return f"""## 3. Draft methodology text (as implemented)

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

**Targets.** Depression status was defined as BDI total ≥ {tg['binary_cutoff']}. Severity was defined by the
BDI interpretation bands {bands}. Sensitivity analyses used BDI ≥ {tg['binary_cutoff_sensitivity']} and a five-class
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
variables, Min–Max scaling to [0, 1], optional SMOTE (k = {cfg['imbalance']['smote_k_neighbors']}), feature
selection and the classifier. Feature-selection methods were mutual information and χ² filters (k ∈ {fsc['k_grid']}),
recursive feature elimination with logistic regression (step 1, k ∈ {fsc['k_grid']}), L1-penalised logistic
regression (C ∈ {fsc['l1_C_grid']}; one-vs-rest for severity) and Boruta ({fsc['boruta']['n_iter']} iterations,
random forest with {fsc['boruta']['n_estimators']} trees, Bonferroni-corrected binomial test at α = {fsc['boruta']['alpha']}).
Base learners were logistic regression, random forest, XGBoost and an RBF support-vector machine, with
balanced class weights (balanced sample weights for XGBoost). Hyper-parameter search spaces: {grids}.
Ensembles were soft voting with equal weights and stacking with a logistic-regression meta-learner trained on
{cv['stacking_cv']}-fold out-of-fold probabilities, both built from the tuned base pipelines of the same outer fold.
The hybrid model was defined as the feature-selection method and ensemble with the highest mean macro F1 on
feature set {cfg['key_feature_set']}.

**Validation.** Models were evaluated with repeated stratified nested cross-validation: an outer loop of
{cv['outer_splits']} folds × {cv['outer_repeats']} repeats and an inner {cv['inner_splits']}-fold stratified loop with randomised search
({cv['n_iter']} candidates, scoring: macro F1). No separate hold-out set was used because the minority classes are
too small to give stable hold-out estimates; every participant is predicted out-of-fold once per repeat.
Ablation, imbalance and sensitivity analyses used {cv['ablation_repeats']} outer repeats. Random seed: {cfg['seed']}.

**Evaluation.** Binary metrics: accuracy, balanced accuracy, precision, recall (sensitivity), specificity, F1,
macro F1, ROC-AUC, PR-AUC (for both classes), Matthews correlation coefficient and Brier score. Multiclass
metrics: accuracy, balanced accuracy, macro and weighted F1, Cohen's κ, quadratic weighted κ and macro
one-vs-rest ROC-AUC. Results are mean ± SD over outer folds with 95% percentile bootstrap confidence intervals
({cv['n_bootstrap']} resamples of participants from the pooled out-of-fold predictions). Models were compared with
the Nadeau–Bengio corrected resampled t-test and the Wilcoxon signed-rank test on paired fold scores, McNemar's
test on paired out-of-fold predictions (Holm-adjusted), and the Friedman test with Nemenyi critical difference.

**Explainability.** The hybrid model of each task was refitted on the whole subset with the most frequent
nested-CV hyper-parameters and explained with permutation SHAP on the raw input variables
(background {cfg['shap']['background_size']} participants, {cfg['shap']['explain_size']} explained participants stratified by class).

**Software.** Python {v['python']}; numpy {v['numpy']}, pandas {v['pandas']}, scipy {v['scipy']},
scikit-learn {v['scikit-learn']}, imbalanced-learn {v['imbalanced-learn']}, xgboost {v['xgboost']},
statsmodels {v['statsmodels']}, scikit-posthocs {v['scikit-posthocs']}, shap {v['shap']},
matplotlib {v['matplotlib']}, python-docx {v['python-docx']}; diagrams with Graphviz.
"""


GUIDE = """## 4. How to read the results

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
"""


def limitations(cfg: dict[str, Any], paths: Paths) -> str:
    q = (_load(paths.results / "data_quality.json") or {}).get("quality", {})
    counts = _load(paths.results / "class_counts.json") or {}
    stats = _load(paths.results / "statistics_summary.json") or {}
    lines = ["## 5. Limitations", ""]
    if q:
        lines.append(f"- **Data provenance.** {q['generator_matched_rows']} of {q['raw_rows']} rows match mock-generator "
                     "output in the repository history; the remaining rows could not be verified from the repository "
                     "alone. Results on the two subsets differ and should be interpreted accordingly.")
    if counts:
        mins = {s: min(int(v) for v in c["binary"].values()) for s, c in counts.items()}
        lines.append("- **Small minority classes.** Not-depressed participants: " +
                     ", ".join(f"{catalog.subset_label(cfg, s)} n = {n}" for s, n in mins.items()) +
                     ". Estimates for this class (specificity, PR-AUC) have wide confidence intervals.")
    epv = {s: v["logistic"].get("events_per_variable") for s, v in stats.items()}
    if epv:
        lines.append("- **Regression power.** Events per parameter in the multivariable logistic model: " +
                     ", ".join(f"{catalog.subset_label(cfg, s)} {e}" for s, e in epv.items()) +
                     "; values below 10 indicate over-fitting risk and unstable odds ratios.")
    lines += [
        "- **Target definition and leakage.** Both targets are deterministic functions of the BDI items; any model "
        "that sees BDI items (set A) re-computes the label. Only sets B–D test prediction from other information.",
        "- **Construct overlap.** FCRI items on sadness (19) and mood (26) overlap with depressive symptoms; the "
        "ablation without these items shows their influence.",
        "- **Self-report, cross-sectional, single source.** Both instruments are self-report questionnaires completed "
        "at one time point through one online form; common-method variance can inflate the FCRI–BDI association, "
        "and no external validation cohort was available.",
        "- **Instrument version and cut-offs.** The BDI wording and severity bands correspond to the first-edition "
        "BDI interpretation (0–10 normal … 41–63 extreme); the binary cut-off of 11 classifies mild mood "
        "disturbance as depression. Results at BDI ≥ 17 are given as a sensitivity analysis.",
        "- **Data inconsistencies.** Sex-specific cancer types reported by participants of the other gender and "
        "respondents under 18 were retained (see the data-quality table).",
        "- **Explainability.** SHAP values were computed on a stratified sample with a model refitted on all data; "
        "they describe associations learned by the model and do not imply causation.",
    ]
    return "\n".join(lines)


DECISIONS = """## 6. To decide or verify with your supervisor

1. Which rows are genuine participants (provenance of the generator-matched rows and of the remainder).
2. Which subset is primary for the thesis, and whether the other is reported as a sensitivity analysis.
3. Binary cut-off (BDI ≥ 11 vs ≥ 17) and whether six severity classes are reportable given the class sizes.
4. Exclusion of participants under 18 and handling of the gender/cancer-type inconsistencies.
5. Whether to describe the instrument as BDI-I/BDI-IA (item wording and cut-offs) rather than BDI-II.
6. Whether the Streamlit app should switch to an FCRI-based model (it currently uses BDI + FCRI models).
"""


def run(ds, cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    log = get_logger()
    docx_path = build_docx(cfg, paths)
    parts = [
        "# Thesis results summary", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from git commit `{git_commit()}` with "
        "`python -m thesis_analysis.run_all`. Do not edit numbers by hand; re-run the pipeline.", "",
        figure_table_lists(cfg, paths), "", key_findings(cfg, paths), "", methodology_text(cfg, paths), "",
        GUIDE, "", limitations(cfg, paths), "", DECISIONS,
    ]
    out = paths.out / "RESULTS_SUMMARY.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    log.info("Results summary -> %s", out)
    return {"summary": str(out), "docx": str(docx_path)}
