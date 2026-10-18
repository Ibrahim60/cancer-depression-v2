"""Metrics, confidence intervals, statistical comparison of models, and model selection.

Inputs are the out-of-fold (OOF) probabilities saved by ``experiments.run_spec``:
for every outer repeat r each participant has exactly one OOF prediction.

* Fold scores   : metric on each outer test fold (R x S values) -> mean +/- SD.
* 95% CI        : percentile bootstrap over participants of the pooled OOF
                  predictions (each bootstrap draw uses one repeat in rotation,
                  so the CI reflects both sampling and CV-split variability).
* Comparisons   : Nadeau-Bengio corrected resampled t-test and Wilcoxon
                  signed-rank on paired fold scores (Holm-adjusted);
                  McNemar on paired OOF predictions of repeat 1;
                  Friedman + Nemenyi over models across the R x S folds.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from .experiments import ALL_MODELS

BINARY_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
                  "f1_macro", "roc_auc", "pr_auc", "pr_auc_minority", "mcc", "brier"]
MULTI_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "kappa", "qwk",
                 "roc_auc_ovr", "mcc"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "precision": "Precision",
    "recall": "Recall (sensitivity)", "specificity": "Specificity", "f1": "F1 (depressed)",
    "f1_macro": "Macro F1", "f1_weighted": "Weighted F1", "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC (depressed)", "pr_auc_minority": "PR-AUC (not depressed)", "mcc": "MCC",
    "brier": "Brier score", "kappa": "Cohen's kappa", "qwk": "Quadratic weighted kappa",
    "roc_auc_ovr": "Macro OvR ROC-AUC",
}
LOWER_IS_BETTER = {"brier"}


def metric_names(n_classes: int) -> list[str]:
    return BINARY_METRICS if n_classes == 2 else MULTI_METRICS


def compute_metrics(y: np.ndarray, proba: np.ndarray, n_classes: int | None = None) -> dict[str, float]:
    """All thesis metrics for one set of predictions (class = argmax probability)."""
    y = np.asarray(y)
    K = n_classes or proba.shape[1]
    pred = proba.argmax(axis=1)
    out: dict[str, float] = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
    }
    if K == 2:
        p1 = proba[:, 1]
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out.update({
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": _safe(lambda: roc_auc_score(y, p1)),
            "pr_auc": _safe(lambda: average_precision_score(y, p1)),
            "pr_auc_minority": _safe(lambda: average_precision_score(1 - y, 1 - p1)),
            "brier": brier_score_loss(y, p1),
        })
    else:
        out.update({
            "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
            "kappa": cohen_kappa_score(y, pred),
            "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
            "roc_auc_ovr": _safe(lambda: roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                                       labels=list(range(K)))),
        })
    return {k: float(v) for k, v in out.items()}


def _safe(fn):
    try:
        return fn()
    except ValueError:
        return np.nan


def fold_scores(run: dict[str, Any], model: str) -> pd.DataFrame:
    """Metric per outer fold (rows = repeat x fold)."""
    y, fid, P = run["y"], run["fold_id"], run["proba"][model]
    K = P.shape[2]
    rows = []
    for r in range(fid.shape[0]):
        for f in np.unique(fid[r]):
            if f < 0:
                continue
            idx = fid[r] == f
            m = compute_metrics(y[idx], P[r, idx], K)
            rows.append({"repeat": r, "fold": int(f), **m})
    return pd.DataFrame(rows)


def bootstrap_ci(run: dict[str, Any], model: str, n_boot: int, seed: int,
                 alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI of each metric over participants (pooled OOF predictions)."""
    y, P = run["y"], run["proba"][model]
    R, n, K = P.shape
    rng = np.random.default_rng(seed)
    names = metric_names(K)
    draws = {k: [] for k in names}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < K:
            continue
        m = compute_metrics(y[idx], P[b % R, idx], K)
        for k in names:
            draws[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for k, v in draws.items()}


def pooled_metrics(run: dict[str, Any], model: str) -> dict[str, float]:
    """Metrics on pooled OOF predictions, averaged over repeats."""
    y, P = run["y"], run["proba"][model]
    per = [compute_metrics(y, P[r], P.shape[2]) for r in range(P.shape[0])]
    return pd.DataFrame(per).mean().to_dict()


def summarise_run(run: dict[str, Any], cfg: dict[str, Any], with_ci: bool = True,
                  models: list[str] | None = None) -> pd.DataFrame:
    """Long table: run x model x metric with mean, SD (fold level) and bootstrap CI."""
    rows = []
    spec = run["meta"]["spec"]
    for m in models or ALL_MODELS:
        fs = fold_scores(run, m)
        ci = bootstrap_ci(run, m, cfg["cv"]["n_bootstrap"], cfg["seed"], cfg["cv"]["alpha"]) if with_ci else {}
        K = run["proba"][m].shape[2]
        for k in metric_names(K):
            rows.append({"run_id": run["id"], "experiment": spec["experiment"], "task": spec["task"],
                         "subset": spec["subset"], "feature_set": spec["feature_set"],
                         "fs": spec["fs"], "imbalance": spec["imbalance"], "variant": spec["variant"],
                         "model": m, "metric": k, "mean": fs[k].mean(), "sd": fs[k].std(ddof=1),
                         "ci_low": ci.get(k, (np.nan, np.nan))[0],
                         "ci_high": ci.get(k, (np.nan, np.nan))[1], "n_folds": len(fs)})
    return pd.DataFrame(rows)


def oof_predictions(run: dict[str, Any], model: str, repeat: int = 0) -> np.ndarray:
    return run["proba"][model][repeat].argmax(axis=1)


# ----------------------------------------------------------------------------
# Statistical comparison
# ----------------------------------------------------------------------------
def corrected_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold scores."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), k - 1)
    return float(t), float(p)


def mcnemar_test(y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> tuple[float, float, int, int]:
    """McNemar test on paired predictions (correct/incorrect). Exact when discordant < 25."""
    ca, cb = pa == y, pb == y
    b01 = int(np.sum(ca & ~cb))
    b10 = int(np.sum(~ca & cb))
    table = [[int(np.sum(ca & cb)), b01], [b10, int(np.sum(~ca & ~cb))]]
    exact = (b01 + b10) < 25
    res = mcnemar(table, exact=exact, correction=not exact)
    return float(res.statistic), float(res.pvalue), b01, b10


def compare_pairs(runs: dict[str, dict[str, Any]], pairs: list[tuple[str, str, str, str]],
                  metric: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Compare (run_a, model_a) vs (run_b, model_b) on paired folds; Holm-adjusted p-values.

    ``pairs`` items: (run_id_a, model_a, run_id_b, model_b). Runs must share CV splits.
    """
    rows = []
    for ra, ma, rb, mb in pairs:
        A, B = runs[ra], runs[rb]
        fa, fb = fold_scores(A, ma), fold_scores(B, mb)
        k = min(len(fa), len(fb))
        a, b = fa[metric].to_numpy()[:k], fb[metric].to_numpy()[:k]
        n = len(A["y"])
        n_test = n / cfg["cv"]["outer_splits"]
        t, p_t = corrected_ttest(a, b, n - n_test, n_test)
        try:
            w = stats.wilcoxon(a, b, zero_method="zsplit")
            p_w = float(w.pvalue)
        except ValueError:
            p_w = 1.0
        if np.array_equal(A["y"], B["y"]):
            _, p_mc, b01, b10 = mcnemar_test(A["y"], oof_predictions(A, ma), oof_predictions(B, mb))
        else:
            p_mc, b01, b10 = np.nan, -1, -1
        rows.append({"a": f"{ra}:{ma}", "b": f"{rb}:{mb}", "metric": metric,
                     "mean_a": a.mean(), "mean_b": b.mean(), "diff": a.mean() - b.mean(),
                     "t_corrected": t, "p_corrected_t": p_t, "p_wilcoxon": p_w,
                     "p_mcnemar": p_mc, "a_only_correct": b01, "b_only_correct": b10, "n_folds": k})
    df = pd.DataFrame(rows)
    if len(df):
        for col in ["p_corrected_t", "p_wilcoxon", "p_mcnemar"]:
            ok = df[col].notna()
            if ok.any():
                df.loc[ok, col + "_holm"] = multipletests(df.loc[ok, col], method="holm")[1]
    return df


def friedman_nemenyi(score_matrix: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """Friedman test across models (columns) over folds (rows) + Nemenyi critical difference.

    Higher scores are better; rank 1 = best.
    """
    import scikit_posthocs as sp
    M = score_matrix.dropna()
    k, N = M.shape[1], M.shape[0]
    stat, p = stats.friedmanchisquare(*[M[c].to_numpy() for c in M.columns])
    ranks = M.rank(axis=1, ascending=False).mean()
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    nem = sp.posthoc_nemenyi_friedman(M.to_numpy())
    nem.index = nem.columns = list(M.columns)
    return {"statistic": float(stat), "p": float(p), "avg_ranks": ranks.sort_values(),
            "cd": float(cd), "k": k, "N": N, "nemenyi_p": nem}


# ----------------------------------------------------------------------------
# Model selection
# ----------------------------------------------------------------------------
def choose_hybrid(summary: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Hybrid = (FS method, ensemble) with the best mean primary metric on the key feature set.

    Candidates: every FS method in experiment 'fs' plus 'none' (experiment 'featuresets',
    key set) crossed with the two ensembles. Ties are broken by the lower SD.
    """
    pm = cfg["cv"]["primary_metric"]
    key = cfg["key_feature_set"]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    s = summary[(summary.metric == pm) & (summary.feature_set == key)
                & (summary.experiment.isin(["fs", "featuresets"]))
                & (summary.model.isin(["voting", "stacking"]))]
    for (task, sub), g in s.groupby(["task", "subset"]):
        g = g.sort_values(["mean", "sd"], ascending=[False, True])
        best = g.iloc[0]
        out.setdefault(task, {})[sub] = {"fs": best["fs"], "model": best["model"],
                                         "run_id": best["run_id"], pm: float(best["mean"]),
                                         "sd": float(best["sd"])}
    return out


def best_model_overall(summary: pd.DataFrame, cfg: dict[str, Any], task: str, subset: str,
                       feature_set: str | None = None) -> pd.Series:
    """Best (run, model) by mean primary metric for a task/subset (optionally one feature set)."""
    pm = cfg["cv"]["primary_metric"]
    s = summary[(summary.metric == pm) & (summary.task == task) & (summary.subset == subset)
                & (summary.experiment.isin(["fs", "featuresets"]))]
    if feature_set:
        s = s[s.feature_set == feature_set]
    return s.sort_values(["mean", "sd"], ascending=[False, True]).iloc[0]
