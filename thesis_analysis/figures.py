"""Result figures (Chapter 4). Every function saves PNG (300 dpi) + SVG via utils.save_fig."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from .experiments import ALL_MODELS, MODEL_LABELS
from .feature_selection import FS_LABELS
from .utils import Paths, fig_size, save_fig, seq_cmap

TASK_LABELS = {"binary": "Binary detection", "severity6": "Severity (6 classes)",
               "binary17": "Binary (BDI ≥ 17)", "severity5": "Severity (5 classes)"}
SHORT_MODEL = {"lr": "LR", "rf": "RF", "xgb": "XGB", "svm": "SVM", "voting": "Voting", "stacking": "Stacking"}


def _pm(v) -> str:
    v = np.asarray(v, float)
    return f"{v.mean():.3f}" if len(v) < 2 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def _colors(cfg):
    return cfg["style"]["categorical"]


def model_color(cfg, m: str) -> str:
    return _colors(cfg)[ALL_MODELS.index(m)]


# ----------------------------------------------------------------------------
def featureset_bars(summary: pd.DataFrame, cfg: dict[str, Any], paths: Paths, subset: str,
                    name: str, appendix: bool) -> None:
    """Grouped bars: feature set x model, one panel per task (macro F1 mean ± SD)."""
    import matplotlib.pyplot as plt
    pm = cfg["cv"]["primary_metric"]
    tasks = ["binary", "severity6"]
    fig, axes = plt.subplots(1, 2, figsize=fig_size(cfg, 1.0, 0.5), sharey=True)
    sets = list(cfg["feature_sets"])
    w = 0.8 / len(ALL_MODELS)
    for ax, task in zip(axes, tasks):
        s = summary[(summary.experiment == "featuresets") & (summary.task == task)
                    & (summary.subset == subset) & (summary.metric == pm)]
        for i, m in enumerate(ALL_MODELS):
            g = s[s.model == m].set_index("feature_set").reindex(sets)
            x = np.arange(len(sets)) + (i - (len(ALL_MODELS) - 1) / 2) * w
            ax.bar(x, g["mean"], w * 0.92, yerr=g["sd"], color=model_color(cfg, m),
                   label=SHORT_MODEL[m], error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": "#52514e"})
        ax.set_xticks(range(len(sets)), sets)
        ax.set_title(TASK_LABELS[task], loc="left", fontsize=cfg["style"]["font_size"])
        ax.set_xlabel("Feature set")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Macro F1 (mean ± SD)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07), columnspacing=1.0,
               handlelength=1.2)
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def model_fs_heatmap(summary: pd.DataFrame, cfg: dict[str, Any], paths: Paths, subset: str,
                     name: str, appendix: bool) -> None:
    """Heatmap of macro F1: models (rows) x FS methods (columns), one panel per task."""
    import matplotlib.pyplot as plt
    pm = cfg["cv"]["primary_metric"]
    key = cfg["key_feature_set"]
    methods = ["none"] + cfg["feature_selection"]["methods"]
    fig, axes = plt.subplots(1, 2, figsize=fig_size(cfg, 1.0, 0.48))
    for ax, task in zip(axes, ["binary", "severity6"]):
        s = summary[(summary.task == task) & (summary.subset == subset) & (summary.metric == pm)
                    & (summary.feature_set == key) & summary.experiment.isin(["fs", "featuresets"])]
        M = s.pivot_table(index="model", columns="fs", values="mean").reindex(index=ALL_MODELS, columns=methods)
        vmin, vmax = np.nanmin(M.values), np.nanmax(M.values)
        ax.imshow(M.values, cmap=seq_cmap(cfg), vmin=vmin - 0.15 * (vmax - vmin + 1e-9), vmax=vmax, aspect="auto")
        best = np.unravel_index(np.nanargmax(M.values), M.shape)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                frac = (v - vmin) / (vmax - vmin + 1e-9)
                ax.text(j, i, f"{v:.3f}".lstrip("0"), ha="center", va="center",
                        fontsize=cfg["style"]["font_size"],
                        fontweight="bold" if (i, j) == best else "normal",
                        color="white" if frac > 0.55 else cfg["style"]["text"])
        ax.set_xticks(range(len(methods)), ["None", "MI", "χ²", "RFE", "L1", "Boruta"], rotation=0)
        ax.set_yticks(range(len(ALL_MODELS)), [SHORT_MODEL[m] for m in ALL_MODELS])
        ax.set_title(TASK_LABELS[task], loc="left", fontsize=cfg["style"]["font_size"])
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def roc_pr_curves(run: dict[str, Any], cfg: dict[str, Any], paths: Paths, roc_name: str,
                  pr_name: str, appendix: bool) -> None:
    """ROC and PR curves of all models (pooled OOF predictions of the first repetition)."""
    import matplotlib.pyplot as plt
    y = run["y"]
    figs = {}
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=fig_size(cfg, 0.75, 0.85))
        for m in ALL_MODELS:
            P = run["proba"][m][:, :, 1]
            aucs = [roc_auc_score(y, P[r]) if kind == "roc" else average_precision_score(1 - y, 1 - P[r])
                    for r in range(P.shape[0])]
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y, P[0])
                ax.plot(fpr, tpr, color=model_color(cfg, m), lw=1.6,
                        label=f"{SHORT_MODEL[m]} (AUC {_pm(aucs)})")
            else:
                # minority class ("not depressed") as positive: the clinically hard class
                prec, rec, _ = precision_recall_curve(1 - y, 1 - P[0])
                ax.plot(rec, prec, color=model_color(cfg, m), lw=1.6,
                        label=f"{SHORT_MODEL[m]} (AP {_pm(aucs)})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color="#9a9993", lw=0.8)
            ax.set_xlabel("False-positive rate (1 − specificity)")
            ax.set_ylabel("True-positive rate (sensitivity)")
        else:
            base = (1 - y).mean()
            ax.axhline(base, ls="--", color="#9a9993", lw=0.8)
            ax.set_xlabel("Recall (not-depressed class)")
            ax.set_ylabel("Precision (not-depressed class)")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.02)
        ax.legend(loc="lower right" if kind == "roc" else "upper right",
                  fontsize=cfg["style"]["font_size"], handlelength=1.2)
        fig.tight_layout()
        save_fig(fig, paths, roc_name if kind == "roc" else pr_name, appendix=appendix)


def confusion_pair(y: np.ndarray, pred: np.ndarray, class_names: list[str], cfg: dict[str, Any],
                   paths: Paths, name: str, appendix: bool, short_names: list[str] | None = None) -> None:
    """Counts and row-normalised confusion matrices side by side."""
    import matplotlib.pyplot as plt
    K = len(class_names)
    C = confusion_matrix(y, pred, labels=list(range(K)))
    N = C / C.sum(axis=1, keepdims=True)
    labs = short_names or class_names
    aspect = 0.5 if K == 2 else 0.55
    fig, axes = plt.subplots(1, 2, figsize=fig_size(cfg, 1.0, aspect))
    for ax, M, fmt, title in ((axes[0], C, "{:d}", "Counts"), (axes[1], N, "{:.2f}", "Row-normalised (recall)")):
        ax.imshow(M, cmap=seq_cmap(cfg), vmin=0, vmax=M.max())
        for i in range(K):
            for j in range(K):
                v = M[i, j]
                ax.text(j, i, fmt.format(int(v) if fmt == "{:d}" else v), ha="center", va="center",
                        fontsize=cfg["style"]["font_size"],
                        color="white" if v > 0.55 * M.max() else cfg["style"]["text"])
        ax.set_xticks(range(K), labs, rotation=45 if K > 2 else 0, ha="right" if K > 2 else "center")
        ax.set_yticks(range(K), labs)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, loc="left", fontsize=cfg["style"]["font_size"])
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def cd_diagram(results: list[dict[str, Any]], cfg: dict[str, Any], paths: Paths, name: str,
               appendix: bool) -> None:
    """Critical-difference diagrams (Demsar 2006): average ranks, CD bar and non-significant cliques."""
    import matplotlib.pyplot as plt
    n = len(results)
    fs = cfg["style"]["font_size"]
    fig, axes = plt.subplots(n, 1, figsize=fig_size(cfg, 1.0, 0.36 * n), squeeze=False)
    for ax, res in zip(axes[:, 0], results):
        ranks = res["avg_ranks"].sort_values()
        k, cd = res["k"], res["cd"]
        ax.set_xlim(-1.2, k + 2.2)
        ax.set_ylim(-1.35, 1.0)
        ax.hlines(0, 1, k, color=cfg["style"]["text"], lw=1)
        for r in range(1, k + 1):
            ax.vlines(r, -0.05, 0.05, color=cfg["style"]["text"], lw=1)
            ax.text(r, 0.1, str(r), ha="center", va="bottom", fontsize=fs)
        ax.hlines(0.62, 1, 1 + cd, color=cfg["style"]["text"], lw=1.2)
        ax.vlines([1, 1 + cd], 0.57, 0.67, color=cfg["style"]["text"], lw=1.2)
        ax.text(1 + cd / 2, 0.7, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=fs)
        items = list(ranks.items())
        half = (len(items) + 1) // 2
        for i, (m, rk) in enumerate(items):
            left = i < half
            j = i if left else len(items) - 1 - i
            yy = -0.42 - 0.3 * j
            xe = 0.6 if left else k + 0.4
            ax.plot([rk, rk, xe], [0, yy, yy], color=model_color(cfg, m), lw=1.4)
            ax.text(xe + (-0.08 if left else 0.08), yy, f"{SHORT_MODEL[m]} ({rk:.2f})",
                    ha="right" if left else "left", va="center", fontsize=fs)
        vals = [v for _, v in items]
        cliques = []
        for i in range(len(vals)):
            j = i
            while j + 1 < len(vals) and vals[j + 1] - vals[i] < cd:
                j += 1
            if j > i and not any(a <= i and j <= b for a, b in cliques):
                cliques.append((i, j))
        for c_i, (a, b) in enumerate(cliques):
            yy = -0.14 - 0.09 * c_i
            ax.hlines(yy, vals[a] - 0.04, vals[b] + 0.04, color=cfg["style"]["text"], lw=3.5)
        ax.set_title(f"{res['title']} (Friedman p = {res['p']:.3g})", loc="left", fontsize=fs)
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def calibration_plot(y: np.ndarray, P: np.ndarray, label: str, cfg: dict[str, Any], paths: Paths,
                     name: str, appendix: bool, n_bins: int = 10) -> dict[str, float]:
    """Reliability diagram (quantile bins) and histogram of predicted probabilities."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    yy = np.tile(y, P.shape[0])
    pp = P.reshape(-1)
    frac, mean = calibration_curve(yy, pp, n_bins=n_bins, strategy="quantile")
    fig, (ax, axh) = plt.subplots(2, 1, figsize=fig_size(cfg, 0.7, 1.05), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], ls="--", color="#9a9993", lw=0.8, label="Perfect calibration")
    ax.plot(mean, frac, marker="o", ms=5, color=cfg["style"]["categorical"][0], label=label)
    ax.set_ylabel("Observed proportion depressed")
    ax.legend(loc="upper left", fontsize=cfg["style"]["font_size"])
    axh.hist(pp, bins=40, color=cfg["style"]["categorical"][0], alpha=0.8)
    axh.set_xlabel("Predicted probability of depression")
    axh.set_ylabel("Count")
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)
    return {"brier": float(brier_score_loss(yy, pp))}


def learning_curve_plot(curves: dict[str, dict[str, np.ndarray]], cfg: dict[str, Any], paths: Paths,
                        name: str, appendix: bool) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(curves), figsize=fig_size(cfg, 1.0, 0.45), squeeze=False)
    for ax, (title, c) in zip(axes[0], curves.items()):
        for key, col, lab in (("train", 1, "Training"), ("test", 0, "Cross-validation")):
            m, s = c[key].mean(axis=1), c[key].std(axis=1)
            ax.plot(c["sizes"], m, marker="o", ms=4, color=cfg["style"]["categorical"][col], label=lab)
            ax.fill_between(c["sizes"], m - s, m + s, color=cfg["style"]["categorical"][col], alpha=0.18, lw=0)
        ax.set_title(title, loc="left", fontsize=cfg["style"]["font_size"])
        ax.set_xlabel("Training participants")
        ax.grid(axis="y")
    axes[0, 0].set_ylabel("Macro F1")
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def error_plot(err: dict[str, pd.DataFrame], cfg: dict[str, Any], paths: Paths, name: str,
               appendix: bool) -> None:
    """Misclassification rate by distance (BDI points) to the nearest class boundary."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(err), figsize=fig_size(cfg, 1.0, 0.42), squeeze=False)
    for ax, (title, d) in zip(axes[0], err.items()):
        ax.bar(d["distance"].astype(str), d["error_rate"], color=cfg["style"]["categorical"][0], width=0.7)
        for i, (r, n) in enumerate(zip(d["error_rate"], d["n"])):
            ax.text(i, r, f"n={n}", ha="center", va="bottom", fontsize=cfg["style"]["font_size"])
        ax.set_title(title, loc="left", fontsize=cfg["style"]["font_size"])
        ax.set_ylim(0, min(1, d["error_rate"].max() * 1.25 + 0.02))
    axes[0, 0].set_ylabel("Misclassification rate")
    fig.supxlabel("Distance of BDI total from the nearest class cut-off (BDI points)",
                  fontsize=cfg["style"]["font_size"])
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def delta_bars(df: pd.DataFrame, cfg: dict[str, Any], paths: Paths, name: str, appendix: bool) -> None:
    """Horizontal bars of Δ macro F1 versus baseline (imbalance strategies and ablations)."""
    import matplotlib.pyplot as plt
    tasks = [t for t in ["binary", "severity6"] if t in set(df.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=fig_size(cfg, 1.0, 0.55), squeeze=False, sharey=True)
    for ax, task in zip(axes[0], tasks):
        d = df[df.task == task]
        y = np.arange(len(d))
        cols = [cfg["style"]["categorical"][0] if v >= 0 else cfg["style"]["categorical"][1] for v in d["delta"]]
        ax.barh(y, d["delta"], xerr=d["delta_sd"], color=cols, height=0.65,
                error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": "#52514e"})
        ax.axvline(0, color=cfg["style"]["text"], lw=0.8)
        ax.set_yticks(y, d["label"])
        ax.set_title(TASK_LABELS[task], loc="left", fontsize=cfg["style"]["font_size"])
        ax.invert_yaxis()
    fig.supxlabel("Δ macro F1 relative to the hybrid model (mean ± SD of paired fold differences)",
                  fontsize=cfg["style"]["font_size"])
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def selection_frequency_plot(freqs: dict[str, pd.Series], cfg: dict[str, Any], paths: Paths, name: str,
                             appendix: bool, top: int = 15) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(freqs), 1, figsize=fig_size(cfg, 1.0, 0.5 * len(freqs) + 0.1),
                             squeeze=False)
    for ax, (title, f) in zip(axes[:, 0], freqs.items()):
        f = f.sort_values(ascending=False).head(top)[::-1]
        ax.barh(range(len(f)), f.values, color=cfg["style"]["categorical"][0], height=0.7)
        ax.set_yticks(range(len(f)), f.index, fontsize=cfg["style"]["font_size"])
        ax.set_xlim(0, 1.0)
        ax.set_title(title, loc="left", fontsize=cfg["style"]["font_size"])
    axes[-1, 0].set_xlabel("Selection frequency across outer training folds")
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)
