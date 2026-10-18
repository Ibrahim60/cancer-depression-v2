"""Dataset description and data quality (thesis section 4.1).

Outputs
-------
* variables / feature-set tables (Chapter 3)
* data-quality table, participant characteristics (overall and by depression
  status, per subset), Cronbach's alpha
* class-distribution and score-distribution figures
* thesis_outputs/data/curated_dataset.csv (de-identified analysis dataset)
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from . import catalog
from .data import CAT_LABELS, FCRI_SUBSCALES, Dataset, subscale_cols
from .utils import Paths, fig_size, get_logger, save_fig, save_json


# ----------------------------------------------------------------------------
# Reliability
# ----------------------------------------------------------------------------
def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha for an (n x k) item matrix."""
    X = items.dropna().to_numpy(dtype=float)
    k = X.shape[1]
    if k < 2:
        return float("nan")
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var)) if total_var > 0 else float("nan")


def alpha_ci(items: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for Cronbach's alpha."""
    rng = np.random.default_rng(seed)
    X = items.dropna().reset_index(drop=True)
    vals = [cronbach_alpha(X.iloc[rng.integers(0, len(X), len(X))]) for _ in range(n_boot)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def fmt_mean_sd(x: pd.Series) -> str:
    return f"{x.mean():.1f} ± {x.std(ddof=1):.1f}"


def fmt_median_iqr(x: pd.Series) -> str:
    q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])
    return f"{q2:.1f} ({q1:.1f}–{q3:.1f})"


def characteristics_table(df: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """Participant characteristics: n (%) for categorical, mean ± SD and median (IQR) for scores."""
    groups = [("Overall", df), ("Depressed", df[df.y_binary == 1]),
              ("Not depressed", df[df.y_binary == 0])]
    head = {name: f"{name} (n = {len(g)})" for name, g in groups}
    rows = []
    for c in ds.cat_cols:
        rows.append({"Variable": CAT_LABELS[c], "Level": "", **{head[n]: "" for n, _ in groups}})
        for lev in df[c].cat.categories:
            r = {"Variable": "", "Level": str(lev)}
            for n, g in groups:
                k = int((g[c] == lev).sum())
                r[head[n]] = f"{k} ({100 * k / len(g):.1f})" if len(g) else "–"
            rows.append(r)
    for col in ["bdi_total", "fcri_total"] + subscale_cols():
        for stat, fn in (("mean ± SD", fmt_mean_sd), ("median (IQR)", fmt_median_iqr)):
            r = {"Variable": ds.labels[col], "Level": stat}
            for n, g in groups:
                r[head[n]] = fn(g[col]) if len(g) else "–"
            rows.append(r)
    for lab in ds.severity_labels:
        r = {"Variable": "BDI severity class" if lab == ds.severity_labels[0] else "", "Level": lab}
        for n, g in groups:
            k = int((g["severity"] == lab).sum())
            r[head[n]] = f"{k} ({100 * k / len(g):.1f})" if len(g) else "–"
        rows.append(r)
    return pd.DataFrame(rows)


def variables_table(ds: Dataset, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"Group": "BDI items", "Variables": "BDI 1–21 (sadness … loss of interest in sex)",
         "Scale": "0–3 per item; total 0–63", "Role": "Target source; features in set A only"},
        {"Group": "FCRI items", "Variables": "FCRI 1–42 (7 subscales)",
         "Scale": "0–4 per item (item 13 reverse-scored); total 0–168", "Role": "Features (sets A–C)"},
    ]
    for s, items in FCRI_SUBSCALES.items():
        rows.append({"Group": "  FCRI subscale", "Variables": f"{s} (items {items[0]}–{items[-1]})",
                     "Scale": f"0–{4 * len(items)}", "Role": "Statistical analysis"})
    for grp in ("demographic", "clinical"):
        for c in cfg["feature_groups"][grp]:
            levels = list(ds.df[c].cat.categories)
            rows.append({"Group": grp.capitalize(), "Variables": CAT_LABELS[c],
                         "Scale": f"Categorical ({len(levels)} levels)",
                         "Role": "Features (sets A, B, D); one-hot encoded"})
    rows.append({"Group": "Targets", "Variables": "Depression status (binary)",
                 "Scale": f"BDI total ≥ {cfg['targets']['binary_cutoff']}", "Role": "Outcome 1"})
    rows.append({"Group": "Targets", "Variables": "Depression severity (6 classes)",
                 "Scale": "; ".join(f"{a} {lo}–{hi}" for a, lo, hi in cfg["targets"]["severity_bounds"]),
                 "Role": "Outcome 2"})
    return pd.DataFrame(rows)


def feature_sets_table(ds: Dataset, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, fs in cfg["feature_sets"].items():
        cols = ds.feature_cols(k, cfg)
        n_onehot = sum(ds.df[c].nunique() for c in cols if c.endswith("__code"))
        n_num = sum(1 for c in cols if not c.endswith("__code"))
        rows.append({"Feature set": k, "Description": fs["label"].split(": ", 1)[1],
                     "Raw variables": len(cols), "Model inputs after one-hot encoding": n_num + n_onehot,
                     "Contains BDI (label source)": "Yes" if "bdi" in fs["groups"] else "No"})
    return pd.DataFrame(rows)


def data_quality_table(ds: Dataset, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Missingness, duplicates, outliers, consistency and provenance checks."""
    raw = ds.raw
    df = ds.df
    q: dict[str, Any] = {}
    q["raw_rows"] = len(raw)
    q["raw_columns"] = raw.shape[1]
    q["exact_duplicate_rows"] = int(raw.duplicated().sum())
    q["duplicate_answers_excl_name_time"] = int(raw.drop(columns=["Timestamp", "Name"]).duplicated().sum())
    bdi_raw = [f"{i}." for i in range(1, 22)]
    fcri_raw = [c for c in raw.columns if re.match(r"^\s*\d+\.\s+\S", c)]
    q["missing_bdi_cells"] = int(raw[bdi_raw].isna().sum().sum())
    q["missing_fcri_cells"] = int(raw[fcri_raw].isna().sum().sum())
    q["participants_with_any_missing_fcri"] = int(raw[fcri_raw].isna().any(axis=1).sum())
    q["missing_demographic_clinical_cells"] = int(raw.iloc[:, 2:10].isna().sum().sum())
    for col in ("bdi_total", "fcri_total"):
        x = df[col]
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        q[f"{col}_iqr_outliers"] = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
    q["bdi_straightlining"] = int((df[ds.bdi_cols].nunique(axis=1) == 1).sum())
    q["fcri_straightlining"] = int((df[ds.fcri_cols].nunique(axis=1) == 1).sum())
    female_only = {"Ovarian Cancer", "Cervical Cancer"}
    q["male_with_female_specific_cancer"] = int(((df.gender == "Male") & df.cancer_type.isin(female_only)).sum())
    q["female_with_prostate_cancer"] = int(((df.gender == "Female") & (df.cancer_type == "Prostate Cancer")).sum())
    q["under_18"] = int((df.age_group == "Below 18").sum())
    q["generator_matched_rows"] = int(df.generator_matched.sum())
    q["bdi_items_6_7_rescored"] = int(sum(
        raw[c].astype(str).str.strip().str.lower().str.rstrip(".").isin(
            [s.lower().rstrip(".") for s in cfg["data"]["bdi_zero_statements"]]).sum()
        for c in ("6.", "7.")))
    labels = {
        "raw_rows": "Rows in raw form export", "raw_columns": "Columns in raw form export",
        "exact_duplicate_rows": "Exact duplicate rows",
        "duplicate_answers_excl_name_time": "Duplicate answer patterns (ignoring name/timestamp)",
        "missing_bdi_cells": "Missing BDI item responses (cells)",
        "missing_fcri_cells": "Missing FCRI item responses (cells)",
        "participants_with_any_missing_fcri": "Participants with ≥1 missing FCRI item (imputed with own subscale mean)",
        "missing_demographic_clinical_cells": "Missing demographic/clinical responses (cells)",
        "bdi_total_iqr_outliers": "BDI total outliers (1.5 × IQR rule)",
        "fcri_total_iqr_outliers": "FCRI total outliers (1.5 × IQR rule)",
        "bdi_straightlining": "Identical answer on all 21 BDI items",
        "fcri_straightlining": "Identical answer on all 42 FCRI items",
        "male_with_female_specific_cancer": "Male participants reporting ovarian/cervical cancer",
        "female_with_prostate_cancer": "Female participants reporting prostate cancer",
        "under_18": "Participants aged under 18",
        "generator_matched_rows": "Rows matching mock-generator output in git history",
        "bdi_items_6_7_rescored": "BDI item 6/7 answers re-scored to 0 (parser correction)",
    }
    table = pd.DataFrame([{"Check": labels[k], "Result": v} for k, v in q.items()])
    return table, {"quality": q, "curation": list(ds.curation_log)}


def reliability_table(ds: Dataset, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    nb, seed = 1000, cfg["seed"]
    for sub in cfg["run_subsets"]:
        d = ds.subset(sub, cfg)
        scales = [("BDI (21 items)", ds.bdi_cols), ("FCRI total (42 items)", ds.fcri_cols)]
        scales += [(f"FCRI {s.lower()} ({len(it)} items)", [f"FCRI_{i}" for i in it])
                   for s, it in FCRI_SUBSCALES.items()]
        for name, cols in scales:
            a = cronbach_alpha(d[cols])
            lo, hi = alpha_ci(d[cols], nb, seed)
            rows.append({"Subset": catalog.subset_label(cfg, sub), "Scale": name, "n": len(d),
                         "Cronbach's alpha": round(a, 3), "95% CI": f"{lo:.3f}–{hi:.3f}"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_class_distribution(ds: Dataset, cfg: dict[str, Any], paths: Paths) -> None:
    import matplotlib.pyplot as plt
    st = cfg["style"]
    subs = cfg["run_subsets"]
    fig, axes = plt.subplots(len(subs), 2, figsize=fig_size(cfg, 1.0, 0.42 * len(subs)),
                             gridspec_kw={"width_ratios": [1, 2.6]}, squeeze=False)
    short = ["Normal", "Mild", "Border-" + chr(10) + "line", "Moder-" + chr(10) + "ate", "Severe", "Ex-" + chr(10) + "treme"]
    for r, sub in enumerate(subs):
        d = ds.subset(sub, cfg)
        ax = axes[r, 0]
        counts = d.y_binary.value_counts().reindex([0, 1]).fillna(0)
        bars = ax.bar(["Not\ndepressed", "Depressed"], counts.values, color=[st["categorical"][1], st["categorical"][0]],
                      width=0.6)
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v)}\n({100 * v / len(d):.1f}%)",
                    ha="center", va="bottom", fontsize=st["font_size"])
        ax.set_ylim(0, counts.max() * 1.3)
        ax.set_ylabel(catalog.subset_label(cfg, sub, short=True) + chr(10) + "Participants")
        ax.set_title("Binary", loc="left", fontsize=st["font_size"])
        ax = axes[r, 1]
        sc = d.severity.value_counts().reindex(ds.severity_labels).fillna(0)
        bars = ax.bar(short, sc.values, color=st["sequential"][3], width=0.65)
        for b, v in zip(bars, sc.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v)}", ha="center", va="bottom",
                    fontsize=st["font_size"])
        ax.set_ylim(0, sc.max() * 1.18)
        ax.set_title(f"Severity, 6 classes (n = {len(d)})", loc="left", fontsize=st["font_size"])
    fig.tight_layout()
    save_fig(fig, paths, catalog.figure("class_distribution", cfg)["name"])


def plot_score_distribution(ds: Dataset, cfg: dict[str, Any], paths: Paths) -> None:
    import matplotlib.pyplot as plt
    st = cfg["style"]
    fig, axes = plt.subplots(1, 2, figsize=fig_size(cfg, 1.0, 0.45))
    d = ds.df
    groups = [(d[~d.generator_matched], "Other rows", st["categorical"][0]),
              (d[d.generator_matched], "Generator-matched rows", st["categorical"][1])]
    for ax, col, bins in ((axes[0], "bdi_total", np.arange(0, 66, 3)),
                          (axes[1], "fcri_total", np.arange(0, 172, 8))):
        ax.hist([g[col] for g, _, _ in groups], bins=bins, stacked=True,
                color=[c for _, _, c in groups], label=[f"{l} (n = {len(g)})" for g, l, _ in groups],
                edgecolor="white", linewidth=0.5)
    for _, lo, _hi in cfg["targets"]["severity_bounds"][1:]:
        axes[0].axvline(lo - 0.5, color=st["muted"], ls="--", lw=0.8)
    axes[0].set_xlabel("BDI total (0–63)")
    axes[1].set_xlabel("FCRI total (0–168)")
    axes[0].set_ylabel("Participants")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save_fig(fig, paths, catalog.figure("score_distribution", cfg)["name"])


# ----------------------------------------------------------------------------
# Stage entry point
# ----------------------------------------------------------------------------
def run(ds: Dataset, cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    log = get_logger()
    # De-identified curated dataset (thesis objective 1)
    out = ds.df.drop(columns=[c for c in ds.df.columns if c.endswith("__code")])
    out.to_csv(paths.data_out / "curated_dataset.csv", index=False, encoding="utf-8-sig")
    catalog.save_table(variables_table(ds, cfg), "variables", cfg, paths.tables)
    catalog.save_table(feature_sets_table(ds, cfg), "feature_sets", cfg, paths.tables)
    qt, qinfo = data_quality_table(ds, cfg)
    catalog.save_table(qt, "data_quality", cfg, paths.tables)
    save_json(qinfo, paths.results / "data_quality.json")
    counts = {}
    for sub in cfg["run_subsets"]:
        d = ds.subset(sub, cfg)
        catalog.save_table(characteristics_table(d, ds), f"characteristics_{sub}", cfg, paths.tables,
                           note="Values are n (%) unless stated otherwise. Depressed: BDI total ≥ "
                                f"{cfg['targets']['binary_cutoff']}. Subset: {catalog.subset_label(cfg, sub)}.")
        counts[sub] = {"n": len(d), "binary": d.y_binary.value_counts().sort_index().to_dict(),
                       "severity6": d.severity.value_counts().reindex(ds.severity_labels).to_dict(),
                       "severity5": d.severity5.value_counts().reindex(d.severity5.cat.categories).to_dict(),
                       "binary17": d.y_binary17.value_counts().sort_index().to_dict(),
                       "bdi_total_mean": d.bdi_total.mean(), "bdi_total_sd": d.bdi_total.std(),
                       "fcri_total_mean": d.fcri_total.mean(), "fcri_total_sd": d.fcri_total.std()}
    save_json(counts, paths.results / "class_counts.json")
    rel = reliability_table(ds, cfg)
    catalog.save_table(rel, "reliability", cfg, paths.tables,
                       note="95% CI: percentile bootstrap (1,000 resamples).")
    plot_class_distribution(ds, cfg, paths)
    plot_score_distribution(ds, cfg, paths)
    log.info("EDA: %s", {k: v["n"] for k, v in counts.items()})
    return {"class_counts": counts, "quality": qinfo["quality"]}
