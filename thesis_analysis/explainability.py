"""SHAP explanations of the hybrid models (thesis section 4.9).

The hybrid model of each task (FS method + ensemble chosen by nested CV) is
refitted on the whole subset with the modal nested-CV hyper-parameters and
explained with the model-agnostic permutation SHAP explainer on the *raw*
input variables (item scores and categorical variables before one-hot
encoding), so each categorical variable receives one SHAP value.

``shap_analyzer.py`` (used by the app) explains the deployed BDI+FCRI models on
their scaled feature matrix; it is not reused here because the thesis models
are full scikit-learn pipelines with in-fold encoding and feature selection.
Its BDI item titles are reproduced in ``data.BDI_TITLES``.

Essential figures go to thesis_outputs/figures; per-class and secondary-subset
plots go to thesis_outputs/figures/appendix.
"""
from __future__ import annotations

import textwrap
import time
from typing import Any

import numpy as np
import pandas as pd

from . import catalog
from .data import GROUP_LABELS, Dataset, readable_feature
from .experiments import MODEL_LABELS, RunSpec, build_final_model, load_run
from .feature_selection import FS_LABELS
from .utils import Paths, fig_size, get_logger, load_json, panel_title, save_fig, save_json


def _explain(est, X: pd.DataFrame, y: np.ndarray, binary: bool, cfg: dict[str, Any]):
    """Permutation SHAP values for a fitted pipeline on a stratified sample of participants."""
    import shap
    from sklearn.model_selection import train_test_split
    sc = cfg["shap"]
    seed = cfg["seed"]
    bg = X.sample(n=min(sc["background_size"], len(X)), random_state=seed)
    if len(X) > sc["explain_size"]:
        idx, _ = train_test_split(np.arange(len(X)), train_size=sc["explain_size"], stratify=y, random_state=seed)
    else:
        idx = np.arange(len(X))
    idx = np.sort(idx)
    Xs = X.iloc[idx]
    cols = list(X.columns)
    from joblib import Parallel, delayed
    M = X.shape[1]
    chunks = np.array_split(Xs.to_numpy(), max(1, min(cfg["n_jobs"], len(Xs))))
    parts = Parallel(n_jobs=cfg["n_jobs"])(
        delayed(_explain_chunk)(est, cols, bg.to_numpy(), c, binary, 2 * (2 * M + 1), seed + i)
        for i, c in enumerate(chunks))
    values = np.concatenate([p[0] for p in parts])
    base = np.concatenate([np.broadcast_to(p[1], (len(p[0]),) + np.shape(p[1])[1:]) for p in parts])
    return values, base, idx


def _explain_chunk(est, cols, bg, A, binary, max_evals, seed):
    """Permutation SHAP for one chunk of participants (runs in a worker process)."""
    import shap

    def f(Z):
        P = est.predict_proba(pd.DataFrame(Z, columns=cols))
        return P[:, 1] if binary else P

    ex = shap.PermutationExplainer(f, shap.maskers.Independent(bg, max_samples=len(bg)), seed=seed)(
        A, max_evals=max_evals, silent=True)
    return ex.values, np.asarray(ex.base_values)


def _short(label: str, n: int = 34) -> str:
    return label if len(label) <= n else label[: n - 1].rstrip() + "…"


def value_label(col: str, v: float, ds: Dataset, n: int = 22) -> str:
    """Category name for a coded categorical value; the number itself for item scores."""
    if col.endswith("__code"):
        cats = ds.df[col.replace("__code", "")].cat.categories
        try:
            return _short(str(cats[int(round(v))]), n)
        except IndexError:
            return f"{v:g}"
    return f"{v:g}"


NOMINAL = {"gender__code", "province__code", "cancer_type__code", "cancer_status__code",
           "treatment_type__code", "on_medication__code"}


def _group_of(col: str, ds: Dataset) -> str:
    for g, cols in ds.groups.items():
        if col in cols:
            return g
    return "other"


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_beeswarm(sv: np.ndarray, Xv: np.ndarray, names: list[str], cfg, paths, name, appendix, top=15,
                  cols: list[str] | None = None):
    import matplotlib.pyplot as plt
    from .utils import seq_cmap
    imp = np.abs(sv).mean(0)
    order = np.argsort(-imp)[:top][::-1]
    fig, ax = plt.subplots(figsize=fig_size(cfg, 1.0, 0.72))
    rng = np.random.default_rng(cfg["seed"])
    cmap = plt.get_cmap("coolwarm")
    for row, j in enumerate(order):
        x = sv[:, j]
        v = Xv[:, j].astype(float)
        vn = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.full_like(v, 0.5)
        # beeswarm-style vertical jitter proportional to local density
        hist, edges = np.histogram(x, bins=40)
        b = np.clip(np.digitize(x, edges) - 1, 0, len(hist) - 1)
        dens = hist[b] / hist.max()
        yj = row + rng.uniform(-0.38, 0.38, len(x)) * dens
        if cols is not None and cols[j] in NOMINAL:   # nominal: colour has no order -> grey
            ax.scatter(x, yj, color="#9a9993", s=7, lw=0, alpha=0.85, rasterized=True)
        else:
            ax.scatter(x, yj, c=vn, cmap=cmap, s=7, lw=0, alpha=0.85, vmin=0, vmax=1, rasterized=True)
    ax.axvline(0, color="#9a9993", lw=0.8)
    ax.set_yticks(range(len(order)), [names[j] for j in order])
    ax.set_xlabel("SHAP value (impact on predicted probability of depression)")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"])
    cb.set_label("Feature value (grey: nominal variable)")
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def plot_bar(panels: dict[str, pd.Series], cfg, paths, name, appendix, top=12):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 1, figsize=fig_size(cfg, 1.0, 0.42 * len(panels) + 0.1), squeeze=False)
    for ax, (title, s) in zip(axes[:, 0], panels.items()):
        s = s.sort_values(ascending=False).head(top)[::-1]
        ax.barh(range(len(s)), s.values, color=cfg["style"]["categorical"][0], height=0.7)
        ax.set_yticks(range(len(s)), s.index)
        panel_title(ax, title, cfg["style"]["font_size"])
    axes[-1, 0].set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def plot_dependence(sv, Xv, names, cfg, paths, name, appendix, k=5, cols=None, ds=None):
    """SHAP value against the feature value for the k most important features.

    Item scores are shown on the x-axis; categorical variables are drawn horizontally with one
    row per category so that category names stay readable.
    """
    import matplotlib.pyplot as plt
    imp = np.abs(sv).mean(0)
    order = np.argsort(-imp)[:k]
    nrow = int(np.ceil(k / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=fig_size(cfg, 1.0, 0.42 * nrow))
    rng = np.random.default_rng(cfg["seed"])
    col0 = cfg["style"]["categorical"][0]
    for ax, j in zip(axes.ravel(), order):
        x = Xv[:, j].astype(float)
        cat = cols is not None and cols[j].endswith("__code")
        if cat:
            levels = np.unique(x)
            pos = {v: i for i, v in enumerate(levels)}
            yy = np.array([pos[v] for v in x]) + rng.uniform(-0.2, 0.2, len(x))
            ax.scatter(sv[:, j], yy, s=8, alpha=0.5, lw=0, color=col0, rasterized=True)
            ax.axvline(0, color="#9a9993", lw=0.8)
            ax.set_yticks(range(len(levels)), [value_label(cols[j], v, ds, 14) for v in levels])
            ax.set_xlabel("SHAP value")
        else:
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(x)), sv[:, j], s=8, alpha=0.5, lw=0, color=col0,
                       rasterized=True)
            ax.axhline(0, color="#9a9993", lw=0.8)
            ax.set_xlabel("Item response (0–4)")
            ax.set_ylabel("SHAP value")
        ax.set_title(_short(names[j], 28), loc="left", fontsize=cfg["style"]["font_size"])
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def plot_waterfalls(cases: list[dict], cfg, paths, name, appendix, top=7, ds=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(cases), 1, figsize=fig_size(cfg, 1.0, 0.36 * len(cases) + 0.1), squeeze=False)
    pos, neg = cfg["style"]["categorical"][1], cfg["style"]["categorical"][0]
    for ax, c in zip(axes[:, 0], cases):
        sv, names, vals = c["shap"], c["names"], c["values"]
        order = np.argsort(-np.abs(sv))
        keep = order[:top]
        rest = sv[order[top:]].sum()
        labels = [f"{_short(names[j], 26)} = {value_label(c['cols'][j], vals[j], ds, 16)}" for j in keep]
        labels += [f"{len(sv) - top} other features"]
        contrib = list(sv[keep]) + [rest]
        start = c["base"]
        y = np.arange(len(contrib))[::-1]
        for yi, v in zip(y, contrib):
            ax.barh(yi, v, left=start, color=pos if v > 0 else neg, height=0.65)
            ax.text(start + v + (0.004 if v >= 0 else -0.004), yi, f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=cfg["style"]["font_size"])
            start += v
        ax.axvline(c["base"], color="#9a9993", lw=0.8, ls="--")
        ax.axvline(start, color=cfg["style"]["text"], lw=0.8)
        ax.set_yticks(y, labels)
        span = np.cumsum([c["base"]] + contrib)
        pad = 0.25 * (span.max() - span.min()) + 0.03
        lo, hi = span.min() - pad, span.max() + pad
        ax.set_xlim(max(-0.15, lo), min(1.15, hi))
        panel_title(ax, f"{c['title']}; base value {c['base']:.2f} → P(depressed) = {start:.2f}",
                    cfg["style"]["font_size"])
    axes[-1, 0].set_xlabel("Predicted probability of depression")
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def plot_groups(shares: pd.DataFrame, cfg, paths, name, appendix):
    import matplotlib.pyplot as plt
    groups = ["bdi", "fcri", "demographic", "clinical"]
    colors = dict(zip(groups, cfg["style"]["categorical"][:4]))
    fig, ax = plt.subplots(figsize=fig_size(cfg, 1.0, 0.36))
    y = np.arange(len(shares))[::-1]
    left = np.zeros(len(shares))
    for g in groups:
        v = shares[g].to_numpy() if g in shares else np.zeros(len(shares))
        ax.barh(y, v, left=left, color=colors[g], height=0.6, label=GROUP_LABELS[g], edgecolor="white", lw=1)
        for yi, l, vv in zip(y, left, v):
            if vv >= 0.07:
                ax.text(l + vv / 2, yi, f"{100 * vv:.0f}%", ha="center", va="center", color="white",
                        fontsize=cfg["style"]["font_size"])
        left += v
    ax.set_yticks(y, shares.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of total mean |SHAP|")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=4, handlelength=1)
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


# ----------------------------------------------------------------------------
# Stage entry point
# ----------------------------------------------------------------------------
def run(ds: Dataset, cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise ImportError("SHAP is not installed. Run:  python -m pip install shap") from exc
    log = get_logger()
    hybrid = load_json(paths.results / "hybrid_selection.json")
    shap_dir = paths.results / "shap"
    shap_dir.mkdir(exist_ok=True)
    prim = cfg["primary_subset"]
    out: dict[str, Any] = {}
    for sub in cfg["run_subsets"]:
        appendix = sub != prim
        sfx = "" if not appendix else f"_{sub}"
        res: dict[str, Any] = {}
        group_rows = {}
        for task in ["binary", "severity6"]:
            h = hybrid[task][sub]
            for fset, rid in ((cfg["key_feature_set"], h["run_id"]),
                              ("A", f"featuresets__{task}__{sub}__A__none__class_weight__default")):
                t0 = time.time()
                run_ = load_run(rid, paths)
                spec = RunSpec(**run_["meta"]["spec"])
                est, X, y, frame = build_final_model(ds, spec, h["model"], run_, cfg)
                est.fit(X, y)
                sv, base, idx = _explain(est, X, y, task == "binary", cfg)   # sv: (n, M) or (n, M, K)
                np.savez_compressed(shap_dir / f"shap__{task}__{sub}__{fset}.npz", values=sv, base=base,
                                    X=X.iloc[idx].to_numpy(), idx=idx, columns=np.array(X.columns))
                cols = list(X.columns)
                names = [readable_feature(c, ds) for c in cols]
                absmean = np.abs(sv).mean(0) if sv.ndim == 2 else np.abs(sv).mean(0).mean(-1)
                imp = pd.Series(absmean, index=cols)
                grp = imp.groupby([_group_of(c, ds) for c in cols]).sum()
                group_rows[f"{'Binary' if task == 'binary' else 'Severity'}, set {fset}"] = grp / grp.sum()
                res[(task, fset)] = {"imp": imp, "sv": sv, "X": X.iloc[idx].to_numpy(), "names": names,
                                     "base": base, "idx": idx, "est": est, "y": y, "cols": cols}
                log.info("SHAP %s/%s/set %s: %d participants in %.0f s", task, sub, fset, len(idx),
                         time.time() - t0)
        key = cfg["key_feature_set"]
        rb, rs = res[("binary", key)], res[("severity6", key)]
        hb, hsv = hybrid["binary"][sub], hybrid["severity6"][sub]
        lbl = lambda s: pd.Series(s.values, index=[_short(readable_feature(c, ds)) for c in s.index])
        plot_beeswarm(rb["sv"], rb["X"], [_short(n) for n in rb["names"]], cfg, paths,
                      catalog.figure("shap_beeswarm", cfg)["name"] + sfx, appendix, cols=rb["cols"])
        plot_bar({f"Binary: {MODEL_LABELS[hb['model']]} + {FS_LABELS[hb['fs']]}": lbl(rb["imp"]),
                  f"Severity (mean over 6 classes): {MODEL_LABELS[hsv['model']]} + {FS_LABELS[hsv['fs']]}":
                      lbl(rs["imp"])},
                 cfg, paths, catalog.figure("shap_bar", cfg)["name"] + sfx, appendix)
        plot_dependence(rb["sv"], rb["X"], rb["names"], cfg, paths,
                        catalog.figure("shap_dependence", cfg)["name"] + sfx, appendix, cfg["shap"]["top_dependence"],
                        cols=rb["cols"], ds=ds)
        # per-class severity (appendix)
        cls_panels = {}
        for k, cname in enumerate(ds.severity_labels):
            cls_panels[cname] = lbl(pd.Series(np.abs(rs["sv"][:, :, k]).mean(0), index=rs["cols"]))
        plot_bar(cls_panels, cfg, paths, f"{catalog.figure('shap_bar', cfg)['name']}_severity_per_class{sfx}",
                 True, top=8)
        # waterfalls: a correctly detected depressed, a correctly identified non-depressed, a misclassified case
        p1 = rb["est"].predict_proba(pd.DataFrame(rb["X"], columns=rb["cols"]))[:, 1]
        yt = rb["y"][rb["idx"]]
        pred = (p1 >= 0.5).astype(int)
        picks = []
        cand = np.where((yt == 1) & (pred == 1))[0]
        if len(cand):
            picks.append(("A: depressed, correctly detected", cand[np.argsort(p1[cand])[len(cand) // 2]]))
        cand = np.where((yt == 0) & (pred == 0))[0]
        if len(cand):
            picks.append(("B: not depressed, correctly identified", cand[np.argmin(p1[cand])]))
        cand = np.where(yt != pred)[0]
        if len(cand):
            picks.append(("C: misclassified", cand[0]))
        cases = [{"title": t, "shap": rb["sv"][i], "names": rb["names"], "values": rb["X"][i],
                  "base": float(np.ravel(rb["base"])[i]), "cols": rb["cols"]}
                 for t, i in picks[: cfg["shap"]["n_waterfall_patients"]]]
        plot_waterfalls(cases, cfg, paths, catalog.figure("shap_waterfall", cfg)["name"] + sfx, appendix, ds=ds)
        shares = pd.DataFrame(group_rows).T.fillna(0)
        order = [f"Binary, set A", f"Binary, set {key}", f"Severity, set A", f"Severity, set {key}"]
        shares = shares.reindex(order)
        shares.to_csv(paths.results / f"shap_group_shares__{sub}.csv")
        plot_groups(shares, cfg, paths, catalog.figure("shap_groups", cfg)["name"] + sfx, appendix)

        # ---- key factors table -------------------------------------------------
        biv = pd.read_csv(paths.tables / f"appendix_bivariate_items_{sub}.csv")
        biv = biv.set_index("key")
        fsum = load_json(paths.results / "fs_summary.json")[sub]
        rows = []
        sev_rank = rs["imp"].rank(ascending=False)
        imp_b = rb["imp"].sort_values(ascending=False)
        freq_b = pd.read_csv(paths.results / f"fs_frequency__binary__{sub}__{fsum['binary']['method']}.csv",
                             index_col=0).iloc[:, 0]
        for rank, (col, v) in enumerate(imp_b.head(15).items(), 1):
            lab = readable_feature(col, ds)
            b = biv.loc[col] if col in biv.index else None
            rows.append({"Feature": lab, "SHAP rank (binary)": rank, "Mean |SHAP| (binary)": round(float(v), 4),
                         "SHAP rank (severity)": int(sev_rank[col]),
                         "Bivariate test": "" if b is None else b["test"],
                         "Effect size": "" if b is None else f"{b['effect_name']} = {b['effect']:.3f}",
                         "p (Holm)": "" if b is None else ("<0.001" if b["p_holm"] < 0.001 else f"{b['p_holm']:.3f}"),
                         f"Selection frequency ({FS_LABELS[fsum['binary']['method']]})":
                             round(float(freq_b.get(lab, np.nan)), 2)})
        catalog.save_table(pd.DataFrame(rows), f"key_factors_{sub}", cfg, paths.tables,
                           note=f"{catalog.subset_label(cfg, sub)}; feature set {key}. Top 15 variables by mean "
                                "|SHAP| of the hybrid binary model. Bivariate tests against depression status "
                                "with Holm correction over all FCRI items, subscales and categorical variables.")
        out[sub] = {"group_shares": shares.round(4).to_dict(orient="index"),
                    "top_binary": {readable_feature(c, ds): float(v) for c, v in imp_b.head(10).items()},
                    "top_severity": {readable_feature(c, ds): float(v)
                                     for c, v in rs["imp"].sort_values(ascending=False).head(10).items()},
                    "waterfall_cases": [c["title"] for c in cases]}
    save_json(out, paths.results / "shap_summary.json")
    return {"subsets": list(out)}
