"""Feature-selection transformers used *inside* every training fold.

All selectors are scikit-learn compatible and are placed in the model
pipeline after encoding/scaling, so selection is re-fitted on each training
fold and never sees the corresponding test fold.

Expensive score computations (mutual information, RFE ranking, L1 path,
Boruta) are memoised per process on a hash of the training data, so the
inner hyper-parameter search does not repeat them for every candidate.

Also provides selection-stability summaries (selection frequency and the
Nogueira et al. 2018 stability index).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectorMixin, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_CACHE_MAX = 64


def _key(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
            h.update(str(p.shape).encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()


def _cached(key: str, fn):
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    val = fn()
    _CACHE[key] = val
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return val


class _BaseSelector(SelectorMixin, BaseEstimator):
    """Common plumbing: subclasses set ``self.support_``."""

    def _get_support_mask(self):
        return self.support_

    def _top_k(self, scores: np.ndarray, k: int) -> np.ndarray:
        k = int(min(max(k, 1), len(scores)))
        order = np.argsort(-scores, kind="stable")
        mask = np.zeros(len(scores), dtype=bool)
        mask[order[:k]] = True
        return mask


class MISelector(_BaseSelector):
    """Filter: keep the k features with the highest mutual information with the target."""

    def __init__(self, k: int = 20, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.scores_ = _cached(_key("mi", X, y, self.random_state),
                               lambda: mutual_info_classif(X, y, random_state=self.random_state))
        self.support_ = self._top_k(self.scores_, self.k)
        self.n_features_in_ = X.shape[1]
        return self


class Chi2Selector(_BaseSelector):
    """Filter: keep the k features with the highest chi-square statistic (non-negative inputs)."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if (X < 0).any():
            raise ValueError("Chi2Selector needs non-negative features (use MinMax scaling).")
        stat, _ = chi2(X, np.asarray(y))
        self.scores_ = np.nan_to_num(stat)
        self.support_ = self._top_k(self.scores_, self.k)
        self.n_features_in_ = X.shape[1]
        return self


class RFESelector(_BaseSelector):
    """Wrapper: recursive feature elimination with logistic regression (step = 1 feature).

    The full elimination ranking is computed once per training set; keeping the
    top-k ranked features is identical to ``RFE(n_features_to_select=k, step=1)``.
    """

    def __init__(self, k: int = 20, class_weight: str | None = "balanced", random_state: int = 42):
        self.k = k
        self.class_weight = class_weight
        self.random_state = random_state

    def _ranking(self, X, y):
        n = X.shape[1]
        remaining = list(range(n))
        rank = np.zeros(n)
        est = LogisticRegression(max_iter=3000, class_weight=self.class_weight)
        pos = n
        while remaining:
            if len(remaining) == 1:
                rank[remaining[0]] = pos
                break
            m = clone(est).fit(X[:, remaining], y)
            imp = np.abs(m.coef_).sum(axis=0)
            worst = int(np.argmin(imp))
            rank[remaining[worst]] = pos
            pos -= 1
            remaining.pop(worst)
        return -rank  # higher = better (eliminated later)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.scores_ = _cached(_key("rfe", X, y, self.class_weight), lambda: self._ranking(X, y))
        self.support_ = self._top_k(self.scores_, self.k)
        self.n_features_in_ = X.shape[1]
        return self


class L1Selector(_BaseSelector):
    """Embedded: features with a non-zero coefficient in an L1-penalised logistic regression
    (one-vs-rest for the multiclass task).

    At least ``min_features`` are always kept (largest |coefficient|) so a very
    strong penalty never produces an empty feature set.
    """

    def __init__(self, C: float = 0.1, class_weight: str | None = "balanced",
                 min_features: int = 3, random_state: int = 42):
        self.C = C
        self.class_weight = class_weight
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        def _fit():
            base = LogisticRegression(penalty="l1", C=self.C, solver="liblinear", max_iter=5000,
                                      class_weight=self.class_weight, random_state=self.random_state)
            if len(np.unique(y)) == 2:
                return np.abs(base.fit(X, y).coef_).max(axis=0)
            # multiclass: one-vs-rest L1 models; a feature is kept if any class uses it
            ovr = OneVsRestClassifier(base).fit(X, y)
            return np.max([np.abs(e.coef_).ravel() for e in ovr.estimators_], axis=0)

        self.scores_ = _cached(_key("l1", X, y, self.C, self.class_weight), _fit)
        mask = self.scores_ > 1e-8
        if mask.sum() < self.min_features:
            mask = self._top_k(self.scores_, self.min_features)
        self.support_ = mask
        self.n_features_in_ = X.shape[1]
        return self


class BorutaSelector(_BaseSelector):
    """All-relevant selection (Boruta; Kursa & Rudnicki 2010), compact implementation.

    In each of ``n_iter`` rounds every feature is copied and shuffled ("shadow
    features"); a random forest is fitted on real + shadow features and a real
    feature scores a *hit* when its impurity importance exceeds the best shadow
    importance. Features whose hit count is significantly above chance
    (one-sided binomial test, Bonferroni-corrected ``alpha``) are confirmed.
    If none is confirmed, the ``min_features`` features with most hits are kept.
    """

    def __init__(self, n_iter: int = 30, alpha: float = 0.05, n_estimators: int = 200,
                 class_weight: str | None = "balanced", min_features: int = 3,
                 random_state: int = 42):
        self.n_iter = n_iter
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.class_weight = class_weight
        self.min_features = min_features
        self.random_state = random_state

    def _hits(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[1]
        hits = np.zeros(n)
        for it in range(self.n_iter):
            shadow = np.column_stack([rng.permutation(X[:, j]) for j in range(n)])
            rf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=7, n_jobs=1,
                                        class_weight=self.class_weight,
                                        random_state=self.random_state + it)
            rf.fit(np.hstack([X, shadow]), y)
            imp = rf.feature_importances_
            hits += imp[:n] > imp[n:].max()
        return hits

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.hits_ = _cached(_key("boruta", X, y, self.n_iter, self.n_estimators,
                                  self.class_weight, self.random_state),
                             lambda: self._hits(X, y))
        n = X.shape[1]
        pvals = binom.sf(self.hits_ - 1, self.n_iter, 0.5)
        mask = pvals < self.alpha / n
        if mask.sum() < self.min_features:
            mask = self._top_k(self.hits_, self.min_features)
        self.scores_ = self.hits_
        self.support_ = mask
        self.n_features_in_ = n
        return self


def make_selector(method: str, cfg: dict[str, Any], class_weight: str | None, seed: int):
    """Return (selector, inner-CV parameter grid) for a feature-selection method key."""
    fsc = cfg["feature_selection"]
    if method == "none":
        return "passthrough", {}
    if method == "mi":
        return MISelector(random_state=seed), {"fs__k": fsc["k_grid"]}
    if method == "chi2":
        return Chi2Selector(), {"fs__k": fsc["k_grid"]}
    if method == "rfe":
        return RFESelector(class_weight=class_weight, random_state=seed), {"fs__k": fsc["k_grid"]}
    if method == "l1":
        return L1Selector(class_weight=class_weight, random_state=seed), {"fs__C": fsc["l1_C_grid"]}
    if method == "boruta":
        b = fsc["boruta"]
        return BorutaSelector(n_iter=b["n_iter"], alpha=b["alpha"], n_estimators=b["n_estimators"],
                              class_weight=class_weight, random_state=seed), {}
    raise KeyError(f"Unknown feature-selection method '{method}'")


FS_LABELS = {"none": "No FS", "mi": "Mutual information", "chi2": "Chi-square",
             "rfe": "RFE (LR)", "l1": "L1 logistic regression", "boruta": "Boruta (RF)"}


# ----------------------------------------------------------------------------
# Stability
# ----------------------------------------------------------------------------
def nogueira_stability(Z: np.ndarray) -> float:
    """Stability index of Nogueira, Sechidis & Brown (2018, JMLR).

    Z is an (M runs x d features) binary selection matrix. 1 = identical
    selections, ~0 = random selection.
    """
    Z = np.asarray(Z, dtype=float)
    M, d = Z.shape
    if M < 2:
        return float("nan")
    p = Z.mean(axis=0)
    s2 = M / (M - 1) * p * (1 - p)
    kbar = Z.sum(axis=1).mean()
    denom = (kbar / d) * (1 - kbar / d)
    if denom == 0:
        return 1.0
    return float(1 - s2.mean() / denom)


def selection_frequency(selections: list[list[str]], universe: list[str]) -> pd.Series:
    """Fraction of runs in which each feature in ``universe`` was selected."""
    idx = {f: i for i, f in enumerate(universe)}
    Z = np.zeros((len(selections), len(universe)))
    for r, sel in enumerate(selections):
        for f in sel:
            if f in idx:
                Z[r, idx[f]] = 1
    return pd.Series(Z.mean(axis=0), index=universe)


def selection_matrix(selections: list[list[str]], universe: list[str]) -> np.ndarray:
    idx = {f: i for i, f in enumerate(universe)}
    Z = np.zeros((len(selections), len(universe)))
    for r, sel in enumerate(selections):
        for f in sel:
            if f in idx:
                Z[r, idx[f]] = 1
    return Z


# ----------------------------------------------------------------------------
# Stage entry point: selection stability across outer folds
# ----------------------------------------------------------------------------
def reported_fs_method(hybrid: dict, summary: pd.DataFrame, cfg: dict, task: str, subset: str) -> str:
    """FS method whose selections are reported: the hybrid's, or the best FS method if the hybrid has none."""
    fs = hybrid[task][subset]["fs"]
    if fs != "none":
        return fs
    pm = cfg["cv"]["primary_metric"]
    s = summary[(summary.experiment == "fs") & (summary.task == task) & (summary.subset == subset)
                & (summary.metric == pm) & summary.model.isin(["voting", "stacking"])]
    return str(s.sort_values("mean", ascending=False).iloc[0]["fs"])


def run(ds, cfg: dict, paths) -> dict:
    """Selection frequency and Nogueira stability for every FS run (variable level)."""
    from . import catalog
    from .data import readable_feature, variable_of
    from .evaluation import summarise_run
    from .experiments import BASE_MODELS, load_run
    from .figures import selection_frequency_plot
    from .utils import get_logger, load_json, save_json

    log = get_logger()
    hybrid = load_json(paths.results / "hybrid_selection.json")
    rows, freq_out, out = [], {}, {}
    quick = []
    for task in ["binary", "severity6"]:
        for sub in cfg["run_subsets"]:
            for method in cfg["feature_selection"]["methods"]:
                rid = f"fs__{task}__{sub}__{cfg['key_feature_set']}__{method}__class_weight__default"
                run_ = load_run(rid, paths)
                quick.append(summarise_run(run_, cfg, with_ci=False, models=["voting", "stacking"]))
                universe = sorted({variable_of(f) for f in run_["meta"]["features"]})
                per_model = {}
                for m in BASE_MODELS:
                    sels = [sorted({variable_of(f) for f in fold["selected"][m]}) for fold in run_["meta"]["folds"]]
                    nfeat = [len(fold["selected"][m]) for fold in run_["meta"]["folds"]]
                    Z = selection_matrix(sels, universe)
                    per_model[m] = selection_frequency(sels, universe)
                    rows.append({"task": task, "subset": sub, "method": method, "model": m,
                                 "nogueira_stability": nogueira_stability(Z),
                                 "mean_selected_inputs": float(np.mean(nfeat)),
                                 "sd_selected_inputs": float(np.std(nfeat, ddof=1)),
                                 "mean_selected_variables": float(Z.sum(axis=1).mean())})
                freq = pd.concat(per_model, axis=1).mean(axis=1)
                freq.index = [readable_feature(f, ds) for f in freq.index]
                freq_out[(task, sub, method)] = freq
                freq.rename("selection_frequency").to_csv(
                    paths.results / f"fs_frequency__{task}__{sub}__{method}.csv", header=True)
    st = pd.DataFrame(rows)
    st.to_csv(paths.results / "fs_stability_full.csv", index=False)
    summ = pd.concat(quick)
    table = (st.groupby(["task", "subset", "method"])
             .agg(stability=("nogueira_stability", "mean"),
                  stability_min=("nogueira_stability", "min"),
                  stability_max=("nogueira_stability", "max"),
                  inputs=("mean_selected_inputs", "mean"),
                  variables=("mean_selected_variables", "mean")).reset_index())
    table["Task"] = table.task.map({"binary": "Binary", "severity6": "Severity (6)"})
    table["Subset"] = table.subset.map(lambda s: catalog.subset_label(cfg, s))
    table["Method"] = table.method.map(FS_LABELS)
    table["Nogueira stability (mean of base learners; range)"] = table.apply(
        lambda r: f"{r.stability:.3f} ({r.stability_min:.3f}–{r.stability_max:.3f})", axis=1)
    table["Selected model inputs (mean)"] = table.inputs.round(1)
    table["Selected variables (mean)"] = table.variables.round(1)
    catalog.save_table(table[["Task", "Subset", "Method", "Nogueira stability (mean of base learners; range)",
                              "Selected model inputs (mean)", "Selected variables (mean)"]],
                       "fs_stability", cfg, paths.tables,
                       note="Stability over the outer folds at the variable level (a categorical variable "
                            "counts as selected if any of its one-hot columns is selected). 1 = identical "
                            "selections in every fold; 0 = no better than random (Nogueira et al., 2018).")
    for sub in cfg["run_subsets"]:
        sel_rows, plot = [], {}
        for task in ["binary", "severity6"]:
            method = reported_fs_method(hybrid, summ, cfg, task, sub)
            f = freq_out[(task, sub, method)].sort_values(ascending=False)
            plot[f"{'Binary' if task == 'binary' else 'Severity'} — {FS_LABELS[method]}"] = f
            for feat, v in f.head(20).items():
                sel_rows.append({"Task": "Binary" if task == "binary" else "Severity (6)",
                                 "FS method": FS_LABELS[method], "Feature": feat,
                                 "Selection frequency": round(float(v), 3)})
            out.setdefault(sub, {})[task] = {"method": method,
                                             "top10": {k: float(v) for k, v in f.head(10).items()}}
        catalog.save_table(pd.DataFrame(sel_rows), f"selected_features_{sub}", cfg, paths.tables,
                           note=f"{catalog.subset_label(cfg, sub)}. Proportion of 25 outer training folds × 4 "
                                "base learners in which the feature was selected (top 20 per task).")
        name = catalog.figure("fs_stability", cfg)["name"]
        prim = sub == cfg["primary_subset"]
        selection_frequency_plot(plot, cfg, paths, name if prim else f"{name}_{sub}", appendix=not prim, top=12)
    save_json(out, paths.results / "fs_summary.json")
    log.info("FS stability: %s", {k: {t: v["method"] for t, v in d.items()} for k, d in out.items()})
    return {"reported_methods": {k: {t: v["method"] for t, v in d.items()} for k, d in out.items()}}
