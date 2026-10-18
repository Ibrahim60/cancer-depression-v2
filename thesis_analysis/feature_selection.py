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
