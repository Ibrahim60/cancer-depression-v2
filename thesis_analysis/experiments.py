"""Models x feature sets x feature-selection methods under repeated nested cross-validation.

Design (all fitted inside the outer training fold only)
-------------------------------------------------------
raw features (item scores + integer-coded categories)
  -> one-hot encoding of categorical variables   (fitted per fold)
  -> Min-Max scaling to [0, 1]                    (fitted per fold)
  -> [SMOTE]                                      (training data only, imbalance='smote')
  -> feature selection                            (fitted per fold)
  -> classifier

Outer loop : RepeatedStratifiedKFold (cv.outer_splits x cv.outer_repeats)
Inner loop : RandomizedSearchCV (cv.inner_splits folds, cv.n_iter candidates)
             tunes each base learner's hyper-parameters *and* the FS parameter.
Ensembles  : soft Voting (equal weights) and Stacking (logistic-regression meta
             learner, internal stratified CV) built from the tuned base pipelines
             of the same outer fold, so they never see the outer test fold.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .data import Dataset, task_spec
from .feature_selection import make_selector
from .utils import Paths, get_logger, save_json

MODEL_LABELS = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
                "svm": "SVM (RBF)", "voting": "Soft Voting", "stacking": "Stacking (LR meta)"}
BASE_MODELS = ["lr", "rf", "xgb", "svm"]
ENSEMBLES = ["voting", "stacking"]
ALL_MODELS = BASE_MODELS + ENSEMBLES


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
class XGBWrapper(ClassifierMixin, BaseEstimator):
    """XGBoost with optional 'balanced' sample weights and arbitrary integer labels."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.1,
                 subsample: float = 1.0, colsample_bytree: float = 1.0, balanced: bool = True,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.balanced = balanced
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        binary = len(self.classes_) == 2
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, random_state=self.random_state,
            objective="binary:logistic" if binary else "multi:softprob",
            eval_metric="logloss" if binary else "mlogloss",
            tree_method="hist", n_jobs=1, verbosity=0)
        sw = compute_sample_weight("balanced", yi) if self.balanced else None
        self.model_.fit(np.asarray(X, dtype=float), yi, sample_weight=sw)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


def make_model(name: str, imbalance: str, seed: int):
    """Base learner with the class-imbalance policy applied."""
    cw = "balanced" if imbalance == "class_weight" else None
    if name == "lr":
        return LogisticRegression(max_iter=5000, class_weight=cw)
    if name == "rf":
        return RandomForestClassifier(class_weight=cw, n_jobs=1, random_state=seed)
    if name == "xgb":
        return XGBWrapper(balanced=imbalance == "class_weight", random_state=seed)
    if name == "svm":
        return SVC(kernel="rbf", probability=True, class_weight=cw, random_state=seed)
    raise KeyError(f"Unknown model '{name}'")


@dataclass
class RunSpec:
    """One cell of the experimental design."""
    experiment: str
    task: str
    subset: str
    feature_set: str
    fs: str = "none"
    imbalance: str = "class_weight"
    variant: str = "default"           # ablation variant key
    drop: list[str] = field(default_factory=list)
    encoding: str = "onehot"           # 'onehot' | 'ordinal'
    scaling: bool = True
    repeats: int = 5

    @property
    def id(self) -> str:
        return "__".join([self.experiment, self.task, self.subset, self.feature_set,
                          self.fs, self.imbalance, self.variant])

    def hash(self, cfg: dict[str, Any]) -> str:
        relevant = {"spec": asdict(self), "cv": cfg["cv"], "grids": cfg["models"]["grids"],
                    "fs": cfg["feature_selection"], "seed": cfg["seed"],
                    "targets": cfg["targets"], "data": cfg["data"], "subsets": cfg["subsets"],
                    "imb": cfg["imbalance"]}
        return hashlib.sha1(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:12]


def build_pipeline(model: str, spec: RunSpec, cat_cols: list[str], num_cols: list[str],
                   cfg: dict[str, Any]) -> tuple[Pipeline, dict[str, list]]:
    """Full preprocessing + FS + classifier pipeline and its inner-CV search space."""
    seed = cfg["seed"]
    if spec.encoding == "onehot" and cat_cols:
        cat_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_tf = "passthrough"
    transformers = []
    if cat_cols:
        transformers.append(("cat", cat_tf, cat_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    prep = ColumnTransformer(transformers, verbose_feature_names_out=False)
    cw = "balanced" if spec.imbalance == "class_weight" else None
    selector, fs_grid = make_selector(spec.fs, cfg, cw, seed)
    steps = [("prep", prep), ("scale", MinMaxScaler() if spec.scaling else "passthrough")]
    if spec.imbalance == "smote":
        steps.append(("smote", SMOTE(k_neighbors=cfg["imbalance"]["smote_k_neighbors"],
                                     random_state=seed)))
    steps += [("fs", selector), ("clf", make_model(model, spec.imbalance, seed))]
    grid = {f"clf__{k}": v for k, v in cfg["models"]["grids"][model].items()}
    grid.update(fs_grid)
    return Pipeline(steps), grid


def selected_features(pipe: Pipeline) -> list[str]:
    """Names of the transformed features kept by the FS step of a fitted pipeline."""
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    fs = pipe.named_steps["fs"]
    if fs == "passthrough" or fs is None:
        return names
    return [n for n, keep in zip(names, fs.get_support()) if keep]


def _cap_grid(grid: dict[str, list], n_features: int) -> dict[str, list]:
    if "fs__k" in grid:
        ks = sorted({min(k, n_features) for k in grid["fs__k"]})
        grid = {**grid, "fs__k": ks}
    return grid


def _n_transformed(X: pd.DataFrame, cat_cols: list[str], num_cols: list[str], encoding: str) -> int:
    if encoding == "onehot":
        return int(sum(X[c].nunique() for c in cat_cols) + len(num_cols))
    return len(cat_cols) + len(num_cols)


# ----------------------------------------------------------------------------
# One outer fold
# ----------------------------------------------------------------------------
def run_outer_fold(X: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
                   spec: RunSpec, cat_cols: list[str], num_cols: list[str],
                   cfg: dict[str, Any], fold_seed: int) -> dict[str, Any]:
    """Tune base learners in the inner CV, build ensembles, predict the outer test fold."""
    warnings.filterwarnings("ignore")
    cvc = cfg["cv"]
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr = y[train_idx]
    inner = StratifiedKFold(cvc["inner_splits"], shuffle=True, random_state=fold_seed)
    n_feat = _n_transformed(X, cat_cols, num_cols, spec.encoding)
    out: dict[str, Any] = {"test_idx": test_idx, "proba": {}, "params": {}, "selected": {},
                           "inner_score": {}, "fit_seconds": {}}
    tuned = {}
    for m in BASE_MODELS:
        t0 = time.time()
        pipe, grid = build_pipeline(m, spec, cat_cols, num_cols, cfg)
        grid = _cap_grid(grid, n_feat)
        n_cand = int(np.prod([len(v) for v in grid.values()])) if grid else 1
        if n_cand > 1:
            search = RandomizedSearchCV(pipe, grid, n_iter=min(cvc["n_iter"], n_cand),
                                        scoring=cvc["inner_scoring"], cv=inner, n_jobs=1,
                                        random_state=fold_seed, refit=True, error_score="raise")
            search.fit(Xtr, ytr)
            best = search.best_estimator_
            out["params"][m] = {k: v for k, v in search.best_params_.items()}
            out["inner_score"][m] = float(search.best_score_)
        else:
            best = pipe.fit(Xtr, ytr)
            out["params"][m] = {}
        tuned[m] = best
        out["proba"][m] = best.predict_proba(Xte)
        out["selected"][m] = selected_features(best)
        out["fit_seconds"][m] = round(time.time() - t0, 2)

    base_unfitted = [(m, clone(tuned[m])) for m in BASE_MODELS]
    t0 = time.time()
    vote = VotingClassifier(base_unfitted, voting="soft").fit(Xtr, ytr)
    out["proba"]["voting"] = vote.predict_proba(Xte)
    out["fit_seconds"]["voting"] = round(time.time() - t0, 2)
    t0 = time.time()
    meta = LogisticRegression(max_iter=5000,
                              class_weight="balanced" if spec.imbalance == "class_weight" else None)
    stack = StackingClassifier(
        [(m, clone(tuned[m])) for m in BASE_MODELS], final_estimator=meta,
        cv=StratifiedKFold(cvc["stacking_cv"], shuffle=True, random_state=fold_seed),
        stack_method="predict_proba", n_jobs=1).fit(Xtr, ytr)
    out["proba"]["stacking"] = stack.predict_proba(Xte)
    out["fit_seconds"]["stacking"] = round(time.time() - t0, 2)
    return out


# ----------------------------------------------------------------------------
# A whole run (all outer folds of one RunSpec)
# ----------------------------------------------------------------------------
def design_matrix(ds: Dataset, spec: RunSpec, cfg: dict[str, Any]):
    """(X, y, cat_cols, num_cols, frame) for a run spec."""
    frame = ds.subset(spec.subset, cfg)
    cols = ds.feature_cols(spec.feature_set, cfg, drop=spec.drop)
    if not cols:
        raise ValueError(f"Run {spec.id} has no features left after dropping {spec.drop}")
    cat_cols = [c for c in cols if c.endswith("__code")]
    num_cols = [c for c in cols if c not in cat_cols]
    tsk = task_spec(spec.task, ds)
    y = frame[tsk["target"]].to_numpy()
    X = frame[cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(int)
    return X, y, cat_cols, num_cols, frame


def run_spec(ds: Dataset, spec: RunSpec, cfg: dict[str, Any], paths: Paths,
             force: bool = False) -> dict[str, Any]:
    """Run (or load cached) nested CV for one spec; saves predictions and metadata."""
    log = get_logger()
    pred_dir = paths.results / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    npz_path = pred_dir / f"{spec.id}.npz"
    meta_path = pred_dir / f"{spec.id}.json"
    h = spec.hash(cfg)
    if not force and npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("hash") == h:
            log.info("  cached  %s", spec.id)
            return load_run(spec.id, paths)

    X, y, cat_cols, num_cols, frame = design_matrix(ds, spec, cfg)
    cvc = cfg["cv"]
    rskf = RepeatedStratifiedKFold(n_splits=cvc["outer_splits"], n_repeats=spec.repeats,
                                   random_state=cfg["seed"])
    splits = list(rskf.split(X, y))
    t0 = time.time()
    results = Parallel(n_jobs=cfg["n_jobs"], backend="loky")(
        delayed(run_outer_fold)(X, y, tr, te, spec, cat_cols, num_cols, cfg, cfg["seed"] + i)
        for i, (tr, te) in enumerate(splits))
    n, K = len(y), len(np.unique(y))
    R, S = spec.repeats, cvc["outer_splits"]
    proba = {m: np.full((R, n, K), np.nan) for m in ALL_MODELS}
    fold_id = np.full((R, n), -1, dtype=int)
    folds_meta = []
    for i, res in enumerate(results):
        r, f = divmod(i, S)
        te = res["test_idx"]
        fold_id[r, te] = f
        for m in ALL_MODELS:
            proba[m][r, te, :] = res["proba"][m]
        folds_meta.append({"repeat": r, "fold": f, "params": res["params"],
                           "selected": res["selected"], "inner_score": res["inner_score"],
                           "fit_seconds": res["fit_seconds"]})
    np.savez_compressed(npz_path, y=y, fold_id=fold_id, row_id=frame["row_id"].to_numpy(),
                        **{f"proba_{m}": proba[m] for m in ALL_MODELS})
    meta = {"hash": h, "spec": asdict(spec), "id": spec.id, "n": int(n), "classes": int(K),
            "features": list(X.columns), "cat_cols": cat_cols, "num_cols": num_cols,
            "folds": folds_meta, "elapsed_s": round(time.time() - t0, 1)}
    save_json(meta, meta_path)
    log.info("  done    %s  (%.0f s)", spec.id, time.time() - t0)
    return load_run(spec.id, paths)


def load_run(run_id: str, paths: Paths) -> dict[str, Any]:
    """Load saved predictions + metadata of a run."""
    pred_dir = paths.results / "predictions"
    npz_path, meta_path = pred_dir / f"{run_id}.npz", pred_dir / f"{run_id}.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"No saved run '{run_id}'. Run:  python -m thesis_analysis.run_all --stage train")
    z = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"id": run_id, "y": z["y"], "fold_id": z["fold_id"], "row_id": z["row_id"],
            "proba": {m: z[f"proba_{m}"] for m in ALL_MODELS}, "meta": meta}


# ----------------------------------------------------------------------------
# Experimental design
# ----------------------------------------------------------------------------
MAIN_TASKS = ["binary", "severity6"]


def plan_main(cfg: dict[str, Any]) -> list[RunSpec]:
    """Experiment 1 (feature sets A-D, no FS) and experiment 2 (FS methods on the key set)."""
    R = cfg["cv"]["outer_repeats"]
    specs = []
    for task in MAIN_TASKS:
        for sub in cfg["run_subsets"]:
            for fset in cfg["feature_sets"]:
                specs.append(RunSpec("featuresets", task, sub, fset, repeats=R))
            for fs in cfg["feature_selection"]["methods"]:
                specs.append(RunSpec("fs", task, sub, cfg["key_feature_set"], fs=fs, repeats=R))
    return specs


def plan_followup(cfg: dict[str, Any], hybrid: dict[str, dict[str, str]]) -> list[RunSpec]:
    """Imbalance comparison, ablations and sensitivity analyses around the chosen hybrid.

    ``hybrid[task][subset] = {"fs": ..., "model": ...}``.
    """
    Ra = cfg["cv"]["ablation_repeats"]
    key = cfg["key_feature_set"]
    prim = cfg["primary_subset"]
    specs: list[RunSpec] = []
    for task in MAIN_TASKS:
        for sub in cfg["run_subsets"]:
            fs = hybrid[task][sub]["fs"]
            for imb in cfg["imbalance"]["strategies"]:
                if imb != cfg["imbalance"]["default"]:
                    specs.append(RunSpec("imbalance", task, sub, key, fs=fs, imbalance=imb, repeats=Ra))
        fs = hybrid[task][prim]["fs"]
        demo = [f"{c}__code" for c in cfg["feature_groups"]["demographic"]]
        clin = [f"{c}__code" for c in cfg["feature_groups"]["clinical"]]
        abl = {
            "no_demographic": dict(drop=demo),
            "no_clinical": dict(drop=clin),
            "no_medication": dict(drop=["on_medication__code"]),
            "no_mood_overlap_items": dict(drop=list(cfg["fcri_mood_overlap_items"])),
            "no_scaling": dict(scaling=False),
            "ordinal_encoding": dict(encoding="ordinal"),
        }
        for name, kw in abl.items():
            specs.append(RunSpec("ablation", task, prim, key, fs=fs, variant=name, repeats=Ra, **kw))
        specs.append(RunSpec("ablation", task, prim, "D", fs="none", variant="D_no_medication",
                             drop=["on_medication__code"], repeats=Ra))
    for sub in cfg["run_subsets"]:
        specs.append(RunSpec("sensitivity", "binary17", sub, key, fs=hybrid["binary"][sub]["fs"],
                             repeats=Ra))
        specs.append(RunSpec("sensitivity", "severity5", sub, key, fs=hybrid["severity6"][sub]["fs"],
                             repeats=Ra))
    return specs


def run_plan(ds: Dataset, specs: list[RunSpec], cfg: dict[str, Any], paths: Paths,
             force: bool = False) -> list[str]:
    """Run every spec in order; returns the run ids."""
    log = get_logger()
    ids = []
    for i, s in enumerate(specs, 1):
        log.info("[%d/%d] %s", i, len(specs), s.id)
        run_spec(ds, s, cfg, paths, force=force)
        ids.append(s.id)
    return ids


# ----------------------------------------------------------------------------
# Final models (fitted on the whole subset; used for SHAP, learning curves)
# ----------------------------------------------------------------------------
def modal_params(run: dict[str, Any], model: str) -> dict[str, Any]:
    """Most frequent tuned hyper-parameter value across outer folds."""
    rows = [f["params"].get(model, {}) for f in run["meta"]["folds"]]
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = [json.dumps(r[k]) for r in rows if k in r]
        out[k] = json.loads(pd.Series(vals).value_counts().index[0])
    return out


def build_final_model(ds: Dataset, spec: RunSpec, model: str, run: dict[str, Any],
                      cfg: dict[str, Any]):
    """Unfitted final pipeline using the modal nested-CV hyper-parameters of each base learner."""
    X, y, cat_cols, num_cols, frame = design_matrix(ds, spec, cfg)
    base = {}
    for m in BASE_MODELS:
        pipe, _ = build_pipeline(m, spec, cat_cols, num_cols, cfg)
        pipe.set_params(**modal_params(run, m))
        base[m] = pipe
    if model in BASE_MODELS:
        est = base[model]
    elif model == "voting":
        est = VotingClassifier(list(base.items()), voting="soft")
    else:
        meta = LogisticRegression(max_iter=5000,
                                  class_weight="balanced" if spec.imbalance == "class_weight" else None)
        est = StackingClassifier(list(base.items()), final_estimator=meta,
                                 cv=StratifiedKFold(cfg["cv"]["stacking_cv"], shuffle=True,
                                                    random_state=cfg["seed"]),
                                 stack_method="predict_proba", n_jobs=1)
    return est, X, y, frame
