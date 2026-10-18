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


# ----------------------------------------------------------------------------
# Fast NumPy implementation (identical results; used for bootstrap resampling)
# ----------------------------------------------------------------------------
def _auc(y01: np.ndarray, s: np.ndarray) -> float:
    P = y01.sum()
    N = len(y01) - P
    if P == 0 or N == 0:
        return np.nan
    r = stats.rankdata(s)
    return float((r[y01 == 1].sum() - P * (P + 1) / 2) / (P * N))


def _ap(y01: np.ndarray, s: np.ndarray) -> float:
    P = y01.sum()
    if P == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    s, y01 = s[o], y01[o]
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y01)[last]
    fps = (last + 1) - tps
    prec = tps / (tps + fps)
    rec = tps / P
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def _kappa(C: np.ndarray, weights: str | None) -> float:
    K = C.shape[0]
    n = C.sum()
    exp = np.outer(C.sum(1), C.sum(0)) / n
    if weights is None:
        W = 1 - np.eye(K)
    else:
        i, j = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
        W = (i - j) ** 2
    den = (W * exp).sum()
    return float(1 - (W * C).sum() / den) if den > 0 else np.nan


def _mcc(C: np.ndarray) -> float:
    t, p = C.sum(1), C.sum(0)
    c, s = np.trace(C), C.sum()
    den = np.sqrt((s ** 2 - (p ** 2).sum()) * (s ** 2 - (t ** 2).sum()))
    return float((c * s - (t * p).sum()) / den) if den > 0 else 0.0


def fast_metrics(y: np.ndarray, proba: np.ndarray, K: int) -> dict[str, float]:
    """Same metrics as ``compute_metrics`` computed from the confusion matrix with NumPy."""
    y = np.asarray(y)
    pred = proba.argmax(axis=1)
    C = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    tp = np.diag(C)
    support, predicted = C.sum(1), C.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(support > 0, tp / support, 0.0)
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = support > 0
    out = {"accuracy": tp.sum() / C.sum(), "balanced_accuracy": rec[present].mean(),
           "f1_macro": f1.mean(), "mcc": _mcc(C)}
    if K == 2:
        p1 = proba[:, 1]
        out.update({"precision": prec[1], "recall": rec[1],
                    "specificity": rec[0] if support[0] else np.nan, "f1": f1[1],
                    "roc_auc": _auc(y, p1), "pr_auc": _ap(y, p1), "pr_auc_minority": _ap(1 - y, 1 - p1),
                    "brier": float(np.mean((p1 - y) ** 2))})
    else:
        out.update({"f1_weighted": float((f1 * support).sum() / support.sum()),
                    "kappa": _kappa(C, None), "qwk": _kappa(C, "quadratic"),
                    "roc_auc_ovr": float(np.mean([_auc((y == k).astype(int), proba[:, k]) for k in range(K)]))})
    return {k: float(v) for k, v in out.items()}


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
            m = fast_metrics(y[idx], P[r, idx], K)
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
        m = fast_metrics(y[idx], P[b % R, idx], K)
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

    Candidates: every FS method of experiment 'fs' crossed with the two ensembles (the
    no-FS ensembles are comparators, not candidates). Ties are broken by the lower SD.
    """
    pm = cfg["cv"]["primary_metric"]
    key = cfg["key_feature_set"]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    s = summary[(summary.metric == pm) & (summary.feature_set == key)
                & (summary.experiment == "fs") & (summary.fs != "none")
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


# ============================================================================
# Stage entry point
# ============================================================================
SHORT_SEV = ["Normal", "Mild", "Border.", "Moder.", "Severe", "Extreme"]
VARIANT_LABELS = {
    "no_demographic": "Without demographic variables",
    "no_clinical": "Without clinical variables",
    "no_medication": "Without medication variable",
    "no_mood_overlap_items": "Without FCRI items 19 & 26 (mood overlap)",
    "no_scaling": "Without Min-Max scaling",
    "ordinal_encoding": "Label (ordinal) encoding instead of one-hot",
    "no_fs": "Without feature selection",
    "smote": "SMOTE instead of class weights",
    "none": "No imbalance handling",
    "fcri_only": "FCRI items only (set C)",
    "no_fcri": "Without FCRI items (set D)",
    "D_no_medication": "Set D without medication variable",
}


def _rid(exp, task, sub, fset, fs="none", imb="class_weight", variant="default"):
    return "__".join([exp, task, sub, fset, fs, imb, variant])


def _ms(mean, sd, nd=3):
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"


def _cell(summary, run_id, model, metric, nd=3):
    r = summary[(summary.run_id == run_id) & (summary.model == model) & (summary.metric == metric)]
    if r.empty:
        return "–"
    r = r.iloc[0]
    return _ms(r["mean"], r["sd"], nd)


def _ci(summary, run_id, model, metric):
    r = summary[(summary.run_id == run_id) & (summary.model == model) & (summary.metric == metric)]
    if r.empty or np.isnan(r.iloc[0]["ci_low"]):
        return "–"
    return f"{r.iloc[0]['ci_low']:.3f}–{r.iloc[0]['ci_high']:.3f}"


def restrict_repeats(run: dict[str, Any], R: int) -> dict[str, Any]:
    """View of a run limited to its first R outer repeats (fold-matched with shorter runs)."""
    return {**run, "fold_id": run["fold_id"][:R], "proba": {m: p[:R] for m, p in run["proba"].items()}}


def per_class_report(run, model, class_names):
    """Per-class precision/recall/F1 on pooled OOF predictions, mean ± SD over repeats."""
    from sklearn.metrics import precision_recall_fscore_support
    y, P = run["y"], run["proba"][model]
    vals = []
    for r in range(P.shape[0]):
        p, rc, f, s = precision_recall_fscore_support(y, P[r].argmax(1), labels=list(range(len(class_names))),
                                                      zero_division=0)
        vals.append(np.vstack([p, rc, f, s]))
    V = np.stack(vals)
    rows = []
    for k, name in enumerate(class_names):
        rows.append({"Class": name, "Support (n)": int(V[0, 3, k]),
                     "Precision": _ms(V[:, 0, k].mean(), V[:, 0, k].std(ddof=1)),
                     "Recall": _ms(V[:, 1, k].mean(), V[:, 1, k].std(ddof=1)),
                     "F1": _ms(V[:, 2, k].mean(), V[:, 2, k].std(ddof=1))})
    return pd.DataFrame(rows)


def _boundary_distance(bdi: np.ndarray, cuts: list[float]) -> np.ndarray:
    c = np.asarray(cuts)[None, :]
    return np.min(np.abs(bdi[:, None] - c), axis=1)


def error_analysis(run, model, frame, cfg, task):
    """Correct vs misclassified participants (first repetition of the outer CV)."""
    y = run["y"]
    pred = run["proba"][model][0].argmax(1)
    wrong = pred != y
    bdi = frame["bdi_total"].to_numpy(float)
    if task == "binary":
        cuts = [cfg["targets"]["binary_cutoff"] - 0.5]
    else:
        cuts = [lo - 0.5 for _, lo, _ in cfg["targets"]["severity_bounds"][1:]]
    dist = _boundary_distance(bdi, cuts)

    def miqr(x):
        q1, q2, q3 = np.percentile(x, [25, 50, 75])
        return f"{q2:.1f} ({q1:.1f}–{q3:.1f})"

    rows = []
    for g, mask in (("Correct", ~wrong), ("Misclassified", wrong)):
        if mask.sum() == 0:
            continue
        r = {"Task": "Binary" if task == "binary" else "Severity (6)", "Group": g, "n": int(mask.sum()),
             "BDI total": miqr(bdi[mask]), "FCRI total": miqr(frame["fcri_total"].to_numpy()[mask]),
             "Distance to nearest cut-off (BDI points)": miqr(dist[mask])}
        if task != "binary":
            adj = (np.abs(pred - y) == 1)[mask].mean() if g == "Misclassified" else np.nan
            r["Errors to adjacent class (%)"] = "" if np.isnan(adj) else f"{100 * adj:.1f}"
        rows.append(r)
    bins = [(0, 1), (1.5, 3), (3.5, 5), (5.5, 9), (9.5, 99)]
    labs = ["≤1", "2–3", "4–5", "6–9", "≥10"]
    by = []
    for (lo, hi), lab in zip(bins, labs):
        m = (dist >= lo) & (dist <= hi)
        if m.sum():
            by.append({"distance": lab, "n": int(m.sum()), "error_rate": float(wrong[m].mean())})
    conf = []
    if task != "binary":
        cm = confusion_matrix(y, pred, labels=list(range(len(SHORT_SEV))))
        np.fill_diagonal(cm, 0)
        for i, j in zip(*np.unravel_index(np.argsort(-cm, axis=None)[:5], cm.shape)):
            conf.append({"true": int(i), "pred": int(j), "n": int(cm[i, j])})
    return pd.DataFrame(rows), pd.DataFrame(by), conf


def _learning_curves(ds, cfg, hybrid, paths):
    """Learning curves of the hybrid models (modal nested-CV hyper-parameters), primary subset."""
    from sklearn.model_selection import StratifiedKFold, learning_curve
    from .experiments import RunSpec, build_final_model, load_run
    sub = cfg["primary_subset"]
    curves = {}
    for task, title in (("binary", "Binary detection"), ("severity6", "Severity (6 classes)")):
        h = hybrid[task][sub]
        run_ = load_run(h["run_id"], paths)
        spec = RunSpec(**run_["meta"]["spec"])
        est, X, y, _ = build_final_model(ds, spec, h["model"], run_, cfg)
        sizes, tr, te = learning_curve(
            est, X, y, train_sizes=np.linspace(0.2, 1.0, 5),
            cv=StratifiedKFold(cfg["cv"]["outer_splits"], shuffle=True, random_state=cfg["seed"]),
            scoring=cfg["cv"]["primary_metric"], n_jobs=cfg["n_jobs"], shuffle=True,
            random_state=cfg["seed"])
        curves[title] = {"sizes": sizes, "train": tr, "test": te}
    return curves


def run(ds, cfg: dict[str, Any], paths) -> dict[str, Any]:
    """Compute every performance table, significance test and result figure."""
    from joblib import Parallel, delayed
    from . import catalog
    from . import figures as F
    from .data import task_spec
    from .experiments import BASE_MODELS, MODEL_LABELS, load_run
    from .feature_selection import FS_LABELS
    from .utils import get_logger, load_json, save_json

    log = get_logger()
    idx = load_json(paths.results / "run_index.json")
    hybrid = load_json(paths.results / "hybrid_selection.json")
    ids = idx["main"] + idx["followup"]
    runs = {i: load_run(i, paths) for i in ids}
    log.info("Summarising %d runs with bootstrap CIs ...", len(ids))
    parts = Parallel(n_jobs=cfg["n_jobs"])(delayed(summarise_run)(runs[i], cfg, True) for i in ids)
    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(paths.results / "summary_all_runs.csv", index=False)
    pm = cfg["cv"]["primary_metric"]
    key = cfg["key_feature_set"]
    prim = cfg["primary_subset"]
    Ra = cfg["cv"]["ablation_repeats"]
    key_results: dict[str, Any] = {"hybrid": hybrid}
    ML = {m: MODEL_LABELS[m] for m in ALL_MODELS}

    # ---- Feature-set comparison ------------------------------------------------
    rows = []
    for sub in cfg["run_subsets"]:
        for task in ["binary", "severity6"]:
            auc = "roc_auc" if task == "binary" else "roc_auc_ovr"
            for fset in cfg["feature_sets"]:
                rid = _rid("featuresets", task, sub, fset)
                for m in ALL_MODELS:
                    rows.append({"Subset": catalog.subset_label(cfg, sub),
                                 "Task": "Binary" if task == "binary" else "Severity (6)",
                                 "Feature set": fset, "Model": ML[m],
                                 "Macro F1": _cell(summary, rid, m, pm), "95% CI": _ci(summary, rid, m, pm),
                                 "Balanced accuracy": _cell(summary, rid, m, "balanced_accuracy"),
                                 "ROC-AUC": _cell(summary, rid, m, auc), "MCC": _cell(summary, rid, m, "mcc")})
    catalog.save_table(pd.DataFrame(rows), "featureset_results", cfg, paths.tables,
                       note="Mean ± SD over 5 × 5 outer folds; 95% CI: participant-level bootstrap of pooled "
                            "out-of-fold predictions. No feature selection. ROC-AUC is macro one-vs-rest for "
                            "severity. Set A contains the BDI items from which the labels are computed.")
    for sub in cfg["run_subsets"]:
        name = catalog.figure("featureset_comparison", cfg)["name"]
        F.featureset_bars(summary, cfg, paths, sub, name if sub == prim else f"{name}_{sub}", sub != prim)
        name = catalog.figure("model_fs_comparison", cfg)["name"]
        F.model_fs_heatmap(summary, cfg, paths, sub, name if sub == prim else f"{name}_{sub}", sub != prim)

    # ---- Model x FS tables -------------------------------------------------------
    for sub in cfg["run_subsets"]:
        for task, tkey in (("binary", "binary_results"), ("severity6", "severity_results")):
            rows = []
            h = hybrid[task][sub]
            for fs in ["none"] + cfg["feature_selection"]["methods"]:
                rid = _rid("featuresets" if fs == "none" else "fs", task, sub, key, fs)
                for m in ALL_MODELS:
                    r = {"FS method": FS_LABELS[fs], "Model": ML[m],
                         "Hybrid": "✓" if (rid == h["run_id"] and m == h["model"]) else "",
                         "Macro F1": _cell(summary, rid, m, pm), "95% CI": _ci(summary, rid, m, pm),
                         "Balanced accuracy": _cell(summary, rid, m, "balanced_accuracy")}
                    if task == "binary":
                        r.update({"Sensitivity": _cell(summary, rid, m, "recall"),
                                  "Specificity": _cell(summary, rid, m, "specificity"),
                                  "Precision": _cell(summary, rid, m, "precision"),
                                  "F1 (depressed)": _cell(summary, rid, m, "f1"),
                                  "Accuracy": _cell(summary, rid, m, "accuracy"),
                                  "ROC-AUC": _cell(summary, rid, m, "roc_auc"),
                                  "PR-AUC (not depressed)": _cell(summary, rid, m, "pr_auc_minority"),
                                  "PR-AUC (depressed)": _cell(summary, rid, m, "pr_auc"),
                                  "MCC": _cell(summary, rid, m, "mcc"),
                                  "Brier": _cell(summary, rid, m, "brier")})
                    else:
                        r.update({"Accuracy": _cell(summary, rid, m, "accuracy"),
                                  "Weighted F1": _cell(summary, rid, m, "f1_weighted"),
                                  "Cohen κ": _cell(summary, rid, m, "kappa"),
                                  "Quadratic weighted κ": _cell(summary, rid, m, "qwk"),
                                  "Macro OvR ROC-AUC": _cell(summary, rid, m, "roc_auc_ovr"),
                                  "MCC": _cell(summary, rid, m, "mcc")})
                    rows.append(r)
            catalog.save_table(pd.DataFrame(rows), f"{tkey}_{sub}", cfg, paths.tables,
                               note=f"{catalog.subset_label(cfg, sub)}; feature set {key} (FCRI + demographic + "
                                    "clinical). Mean ± SD over 25 outer folds; 95% CI for macro F1 by bootstrap. "
                                    "✓ marks the hybrid model (best FS method + ensemble by mean macro F1).")

    # ---- Hybrid-specific outputs -------------------------------------------------
    sig_all, fried_rows, cd_res, err_rows = [], [], {}, []
    for sub in cfg["run_subsets"]:
        frame = ds.subset(sub, cfg)
        appendix = sub != prim
        sfx = "" if not appendix else f"_{sub}"
        err_plot = {}
        for task in ["binary", "severity6"]:
            h = hybrid[task][sub]
            run_h = runs[h["run_id"]]
            hm = h["model"]
            tsk = task_spec(task, ds)
            if not np.array_equal(run_h["row_id"], frame["row_id"].to_numpy()):
                raise RuntimeError(f"Saved predictions of {h['run_id']} do not match the current dataset "
                                   "order. Re-run:  python -m thesis_analysis.run_all --stage train --force")
            hs = summary[(summary.run_id == h["run_id"]) & (summary.model == hm)].set_index("metric")
            key_results.setdefault(sub, {})[task] = {
                "hybrid": {"fs": h["fs"], "model": hm, "run_id": h["run_id"],
                           "metrics": {k: {"mean": float(v["mean"]), "sd": float(v["sd"]),
                                           "ci": [float(v["ci_low"]), float(v["ci_high"])]}
                                       for k, v in hs.iterrows()}}}
            pairs = [(h["run_id"], hm, h["run_id"], m) for m in ALL_MODELS if m != hm]
            nofs = _rid("featuresets", task, sub, key)
            if h["run_id"] != nofs:
                pairs += [(h["run_id"], hm, nofs, hm)]
            for fset in [s for s in cfg["feature_sets"] if s != key]:
                pairs.append((nofs, hm, _rid("featuresets", task, sub, fset), hm))
            cmp_ = compare_pairs(runs, pairs, pm, cfg)
            cmp_.insert(0, "task", task)
            cmp_.insert(0, "subset", sub)
            sig_all.append(cmp_)
            M = pd.DataFrame({m: fold_scores(run_h, m)[pm].to_numpy() for m in ALL_MODELS})
            fr = friedman_nemenyi(M, cfg["cv"]["alpha"])
            fr["title"] = F.TASK_LABELS[task]
            cd_res.setdefault(sub, []).append(fr)
            fried_rows.append({"Subset": catalog.subset_label(cfg, sub), "Task": F.TASK_LABELS[task],
                               "FS method": FS_LABELS[h["fs"]], "Friedman χ²": round(fr["statistic"], 2),
                               "p": f"{fr['p']:.3g}", "Critical difference": round(fr["cd"], 2),
                               "Average ranks (1 = best)": "; ".join(f"{F.SHORT_MODEL[m]} {r:.2f}"
                                                                     for m, r in fr["avg_ranks"].items())})
            fr["nemenyi_p"].to_csv(paths.results / f"nemenyi__{task}__{sub}.csv")
            key_results[sub][task]["friedman"] = {"stat": fr["statistic"], "p": fr["p"], "cd": fr["cd"],
                                                  "ranks": fr["avg_ranks"].to_dict()}
            pred = run_h["proba"][hm][0].argmax(1)
            if task == "binary":
                F.confusion_pair(run_h["y"], pred, tsk["class_names"], cfg, paths,
                                 catalog.figure("confusion_binary", cfg)["name"] + sfx, appendix)
                F.roc_pr_curves(run_h, cfg, paths, catalog.figure("roc_curves", cfg)["name"] + sfx,
                                catalog.figure("pr_curves", cfg)["name"] + sfx, appendix)
                cal = F.calibration_plot(run_h["y"], run_h["proba"][hm][:, :, 1],
                                         f"{F.SHORT_MODEL[hm]} + {FS_LABELS[h['fs']]}", cfg, paths,
                                         catalog.figure("calibration", cfg)["name"] + sfx, appendix)
                key_results[sub][task]["calibration"] = cal
            else:
                F.confusion_pair(run_h["y"], pred, tsk["class_names"], cfg, paths,
                                 catalog.figure("confusion_severity", cfg)["name"] + sfx, appendix,
                                 short_names=SHORT_SEV)
                pc = per_class_report(run_h, hm, tsk["class_names"])
                catalog.save_table(pc, f"per_class_{sub}", cfg, paths.tables,
                                   note=f"{catalog.subset_label(cfg, sub)}. Hybrid model "
                                        f"({MODEL_LABELS[hm]} + {FS_LABELS[h['fs']]}); mean ± SD over the "
                                        "5 repetitions of pooled out-of-fold predictions.")
            et, eb, conf = error_analysis(run_h, hm, frame, cfg, task)
            et.insert(0, "Subset", catalog.subset_label(cfg, sub))
            err_rows.append(et)
            eb.to_csv(paths.results / f"error_by_distance__{task}__{sub}.csv", index=False)
            key_results[sub][task]["top_confusions"] = conf
            key_results[sub][task]["error_by_distance"] = eb.to_dict(orient="records")
            err_plot[F.TASK_LABELS[task]] = eb
        F.error_plot(err_plot, cfg, paths, catalog.figure("error_analysis", cfg)["name"] + sfx, appendix)
        F.cd_diagram(cd_res[sub], cfg, paths, catalog.figure("cd_diagram", cfg)["name"] + sfx, appendix)

    sig = pd.concat(sig_all, ignore_index=True)
    sig.to_csv(paths.results / "significance_full.csv", index=False)
    key_results["significance"] = sig.to_dict(orient="records")

    def nice(x: str) -> str:
        rid, m = x.rsplit(":", 1)
        parts = rid.split("__")
        return f"{F.SHORT_MODEL[m]} | set {parts[3]} | {FS_LABELS[parts[4]]}"

    for sub in cfg["run_subsets"]:
        s = sig[sig.subset == sub]
        out = pd.DataFrame({
            "Task": s.task.map(F.TASK_LABELS), "Model A": s.a.map(nice), "Model B": s.b.map(nice),
            "Macro F1 A": s.mean_a.round(3), "Macro F1 B": s.mean_b.round(3), "Difference": s["diff"].round(3),
            "Corrected t-test p (Holm)": s.p_corrected_t_holm.map(lambda p: f"{p:.3g}"),
            "Wilcoxon p (Holm)": s.p_wilcoxon_holm.map(lambda p: f"{p:.3g}"),
            "McNemar p (Holm)": s.p_mcnemar_holm.map(lambda p: "–" if pd.isna(p) else f"{p:.3g}")})
        catalog.save_table(out, f"significance_{sub}", cfg, paths.tables,
                           note=f"{catalog.subset_label(cfg, sub)}. Paired over the same 25 outer folds. "
                                "Corrected t-test: Nadeau & Bengio (2003). McNemar: paired out-of-fold "
                                "predictions of the first repetition. Holm adjustment within each task.")
    catalog.save_table(pd.DataFrame(fried_rows), "friedman", cfg, paths.tables,
                       note="Friedman test over the 25 outer folds of the hybrid FS configuration; Nemenyi "
                            "critical difference at α = 0.05. Fold scores are not fully independent, so p-values "
                            "are approximate.")
    catalog.save_table(pd.concat(err_rows, ignore_index=True), "error_analysis", cfg, paths.tables,
                       note="Hybrid models, first repetition of the outer CV. Values are median (IQR).")

    log.info("Learning curves ...")
    curves = _learning_curves(ds, cfg, hybrid, paths)
    F.learning_curve_plot(curves, cfg, paths, catalog.figure("learning_curve", cfg)["name"], False)
    key_results["learning_curve"] = {k: {"sizes": v["sizes"].tolist(), "test_mean": v["test"].mean(1).tolist(),
                                         "test_sd": v["test"].std(1).tolist(),
                                         "train_mean": v["train"].mean(1).tolist()} for k, v in curves.items()}

    # ---- Imbalance, ablation, sensitivity -------------------------------------------
    imb_rows, abl_rows, sens_rows, delta_rows = [], [], [], []
    strat_label = {"class_weight": "Class weights", "smote": "SMOTE", "none": "None"}
    for sub in cfg["run_subsets"]:
        for task in ["binary", "severity6"]:
            h = hybrid[task][sub]
            base = restrict_repeats(runs[h["run_id"]], Ra)
            for m in ALL_MODELS:
                r = {"Subset": catalog.subset_label(cfg, sub), "Task": F.TASK_LABELS[task], "Model": ML[m]}
                for imb in cfg["imbalance"]["strategies"]:
                    run_ = base if imb == cfg["imbalance"]["default"] else runs[
                        _rid("imbalance", task, sub, key, h["fs"], imb)]
                    fsr = fold_scores(run_, m)
                    r[f"{strat_label[imb]}: macro F1"] = _ms(fsr[pm].mean(), fsr[pm].std(ddof=1))
                    r[f"{strat_label[imb]}: balanced acc."] = _ms(fsr["balanced_accuracy"].mean(),
                                                                  fsr["balanced_accuracy"].std(ddof=1))
                imb_rows.append(r)
    catalog.save_table(pd.DataFrame(imb_rows), "imbalance", cfg, paths.tables,
                       note=f"Feature set {key} with the hybrid FS method; {Ra} × {cfg['cv']['outer_splits']} "
                            "outer folds (class-weight results restricted to the same folds). SMOTE is applied "
                            "to training folds only.")
    for task in ["binary", "severity6"]:
        h = hybrid[task][prim]
        hm = h["model"]
        base = restrict_repeats(runs[h["run_id"]], Ra)
        bfs = fold_scores(base, hm)[pm].to_numpy()
        variants = [(v, runs[_rid("ablation", task, prim, key, h["fs"], "class_weight", v)]) for v in
                    ["no_demographic", "no_clinical", "no_medication", "no_mood_overlap_items", "no_scaling",
                     "ordinal_encoding"]]
        if h["fs"] != "none":
            variants.append(("no_fs", restrict_repeats(runs[_rid("featuresets", task, prim, key)], Ra)))
        variants += [("smote", runs[_rid("imbalance", task, prim, key, h["fs"], "smote")]),
                     ("none", runs[_rid("imbalance", task, prim, key, h["fs"], "none")]),
                     ("fcri_only", restrict_repeats(runs[_rid("featuresets", task, prim, "C")], Ra)),
                     ("no_fcri", restrict_repeats(runs[_rid("featuresets", task, prim, "D")], Ra)),
                     ("D_no_medication", runs[_rid("ablation", task, prim, "D", "none", "class_weight",
                                                   "D_no_medication")])]
        n = len(runs[h["run_id"]]["y"])
        nt = n / cfg["cv"]["outer_splits"]
        tmp = []
        for v, rr in variants:
            vs = fold_scores(rr, hm)[pm].to_numpy()[:len(bfs)]
            t, p = corrected_ttest(vs, bfs, n - nt, nt)
            tmp.append({"task": task, "variant": v, "label": VARIANT_LABELS[v], "mean": vs.mean(),
                        "sd": vs.std(ddof=1), "delta": vs.mean() - bfs.mean(),
                        "delta_sd": (vs - bfs).std(ddof=1), "p": p})
        tmp = pd.DataFrame(tmp)
        tmp["p_holm"] = multipletests(tmp.p, method="holm")[1]
        tmp["baseline_mean"] = bfs.mean()
        delta_rows.append(tmp)
        abl_rows.append(pd.DataFrame({
            "Task": F.TASK_LABELS[task], "Configuration": ["Hybrid (reference)"] + list(tmp.label),
            "Macro F1": [_ms(bfs.mean(), bfs.std(ddof=1))] + [_ms(a, b) for a, b in zip(tmp["mean"], tmp["sd"])],
            "Δ vs hybrid": [""] + [f"{d:+.3f}" for d in tmp.delta],
            "Corrected t-test p (Holm)": [""] + [f"{p:.3g}" for p in tmp.p_holm]}))
    deltas = pd.concat(delta_rows, ignore_index=True)
    deltas.to_csv(paths.results / "ablation_deltas.csv", index=False)
    catalog.save_table(pd.concat(abl_rows, ignore_index=True), "ablation", cfg, paths.tables,
                       note=f"{catalog.subset_label(cfg, prim)}; ensemble model of the hybrid; {Ra} × "
                            f"{cfg['cv']['outer_splits']} outer folds, paired with the hybrid on the same folds.")
    F.delta_bars(deltas, cfg, paths, catalog.figure("imbalance_ablation", cfg)["name"], False)
    key_results["ablation"] = deltas.to_dict(orient="records")
    for sub in cfg["run_subsets"]:
        for task, src in (("binary17", "binary"), ("severity5", "severity6")):
            rid = _rid("sensitivity", task, sub, key, hybrid[src][sub]["fs"])
            for m in ALL_MODELS:
                sens_rows.append({"Subset": catalog.subset_label(cfg, sub), "Analysis": F.TASK_LABELS[task],
                                  "Model": ML[m], "Macro F1": _cell(summary, rid, m, pm),
                                  "Balanced accuracy": _cell(summary, rid, m, "balanced_accuracy"),
                                  "MCC": _cell(summary, rid, m, "mcc"),
                                  "AUC": _cell(summary, rid, m, "roc_auc" if task == "binary17" else "roc_auc_ovr")})
            ss = summary[(summary.run_id == rid) & (summary.metric == pm)].sort_values("mean", ascending=False)
            key_results[sub][f"sensitivity_{task}"] = {"best_model": ss.iloc[0]["model"],
                                                       "mean": float(ss.iloc[0]["mean"]),
                                                       "sd": float(ss.iloc[0]["sd"])}
    catalog.save_table(pd.DataFrame(sens_rows), "sensitivity", cfg, paths.tables,
                       note=f"Feature set {key} with the hybrid FS method of the corresponding main task; "
                            f"{Ra} × {cfg['cv']['outer_splits']} outer folds.")
    for sub in cfg["run_subsets"]:
        for task in ["binary", "severity6"]:
            for fset in cfg["feature_sets"]:
                b = best_model_overall(summary, cfg, task, sub, fset)
                key_results[sub][task].setdefault("best_by_feature_set", {})[fset] = {
                    "run_id": b["run_id"], "model": b["model"], "fs": b["fs"],
                    "mean": float(b["mean"]), "sd": float(b["sd"]),
                    "ci": [float(b["ci_low"]), float(b["ci_high"])]}
            bb = summary[(summary.metric == pm) & (summary.task == task) & (summary.subset == sub)
                         & (summary.feature_set == key) & summary.experiment.isin(["fs", "featuresets"])
                         & summary.model.isin(BASE_MODELS)].sort_values("mean", ascending=False).iloc[0]
            key_results[sub][task]["best_individual_on_key_set"] = {
                "model": bb["model"], "fs": bb["fs"], "mean": float(bb["mean"]), "sd": float(bb["sd"])}
    save_json(key_results, paths.results / "key_results.json")
    return {"n_runs": len(ids)}
