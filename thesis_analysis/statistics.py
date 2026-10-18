"""Classical statistical analysis (thesis section 4.2).

* Binary outcome: chi-square / Fisher's exact (Monte-Carlo chi-square for sparse
  r x c tables) with Cramér's V; Shapiro-Wilk then Welch t-test (Cohen's d) or
  Mann-Whitney U (rank-biserial r).
* Severity (6 classes): Kruskal-Wallis or one-way ANOVA (epsilon² / eta²) with
  Dunn post-hoc tests; chi-square for categorical variables.
* Spearman correlations of FCRI total and subscales with BDI total.
* Univariate and multivariable logistic regression (OR, 95% CI, p) and
  proportional-odds ordinal regression for severity.
* Multiple comparisons: Holm step-down correction within each table.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from . import catalog
from .data import CAT_LABELS, FCRI_SUBSCALES, Dataset, subscale_cols
from .utils import Paths, div_cmap, fig_size, get_logger, save_fig, save_json, seq_cmap

N_PERM = 10000


def _p(p: float) -> str:
    if p is None or np.isnan(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------
# Categorical association
# ----------------------------------------------------------------------------
def cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else float("nan")


def categorical_test(x: pd.Series, y: pd.Series, seed: int) -> dict[str, Any]:
    """Chi-square, Fisher's exact (2x2, sparse) or Monte-Carlo chi-square (r x c, sparse)."""
    tab = pd.crosstab(x, y)
    tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    T = tab.to_numpy()
    if T.shape[0] < 2 or T.shape[1] < 2:
        return {"test": "not testable", "statistic": np.nan, "p": np.nan, "effect": np.nan,
                "effect_name": "Cramér's V"}
    chi2, p, dof, exp = stats.chi2_contingency(T, correction=False)
    if (exp < 5).any():
        if T.shape == (2, 2):
            _, p = stats.fisher_exact(T)
            test = "Fisher's exact"
        else:
            p = _mc_chi2_p(x, y, chi2, seed)
            test = "Chi-square (Monte-Carlo)"
    else:
        test = "Chi-square"
    return {"test": test, "statistic": float(chi2), "df": int(dof), "p": float(p),
            "effect": cramers_v(T), "effect_name": "Cramér's V"}


def _chi2_stat(xi: np.ndarray, yi: np.ndarray, r: int, c: int) -> float:
    T = np.bincount(xi * c + yi, minlength=r * c).reshape(r, c).astype(float)
    E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum((T - E) ** 2 / E))


def _mc_chi2_p(x: pd.Series, y: pd.Series, observed: float, seed: int) -> float:
    """Monte-Carlo p-value of the chi-square statistic under random permutation of y."""
    xi = pd.factorize(x)[0]
    yi = pd.factorize(y)[0]
    r, c = xi.max() + 1, yi.max() + 1
    rng = np.random.default_rng(seed)
    ge = sum(_chi2_stat(xi, rng.permutation(yi), r, c) >= observed - 1e-9 for _ in range(N_PERM))
    return (ge + 1) / (N_PERM + 1)


# ----------------------------------------------------------------------------
# Continuous: two groups
# ----------------------------------------------------------------------------
def two_group_test(a: np.ndarray, b: np.ndarray, alpha: float) -> dict[str, Any]:
    """Welch t-test if both groups pass Shapiro-Wilk, otherwise Mann-Whitney U.

    ``a`` = depressed, ``b`` = not depressed. Positive effects mean higher values in ``a``.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    normal = all(len(g) >= 3 and stats.shapiro(g).pvalue > alpha for g in (a, b))
    if normal:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
        return {"test": "Welch t-test", "statistic": float(t), "p": float(p), "effect": float(d),
                "effect_name": "Cohen's d", "normal": True}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1
    return {"test": "Mann-Whitney U", "statistic": float(u), "p": float(p), "effect": float(r),
            "effect_name": "Rank-biserial r", "normal": False}


def bivariate_binary(d: pd.DataFrame, ds: Dataset, cfg: dict[str, Any],
                     include_items: bool = False) -> pd.DataFrame:
    """All bivariate tests against depression status, Holm-adjusted."""
    alpha = cfg["cv"]["alpha"]
    rows = []
    dep, non = d[d.y_binary == 1], d[d.y_binary == 0]
    cont = ["fcri_total"] + subscale_cols() + (ds.fcri_cols if include_items else [])
    for c in cont:
        r = two_group_test(dep[c], non[c], alpha)
        rows.append({"Variable": ds.labels[c], "key": c, "Type": "Continuous",
                     "Depressed": f"{dep[c].median():.1f} ({dep[c].quantile(.25):.1f}–{dep[c].quantile(.75):.1f})",
                     "Not depressed": f"{non[c].median():.1f} ({non[c].quantile(.25):.1f}–{non[c].quantile(.75):.1f})",
                     **r})
    for c in ([] if include_items == "items_only" else ds.cat_cols):
        r = categorical_test(d[c].astype(str), d.y_binary, cfg["seed"])
        rows.append({"Variable": CAT_LABELS[c], "key": f"{c}__code", "Type": "Categorical",
                     "Depressed": "see Table (characteristics)", "Not depressed": "", **r})
    df = pd.DataFrame(rows)
    ok = df.p.notna()
    df.loc[ok, "p_holm"] = multipletests(df.loc[ok, "p"], method="holm")[1]
    return df


def format_bivariate(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Variable": df.Variable,
        "Depressed, median (IQR)": df.Depressed.where(df.Type == "Continuous", ""),
        "Not depressed, median (IQR)": df["Not depressed"].where(df.Type == "Continuous", ""),
        "Test": df.test, "Statistic": df.statistic.round(2),
        "Effect size": df.effect_name + " = " + df.effect.round(3).astype(str),
        "p": df.p.map(_p), "p (Holm)": df.p_holm.map(_p),
    })
    return out


# ----------------------------------------------------------------------------
# Continuous: severity classes
# ----------------------------------------------------------------------------
def severity_tests(d: pd.DataFrame, ds: Dataset, cfg: dict[str, Any]) -> pd.DataFrame:
    alpha = cfg["cv"]["alpha"]
    rows = []
    sev = d.severity.astype(str)
    present = [s for s in ds.severity_labels if (sev == s).sum() >= 3]
    for c in ["fcri_total"] + subscale_cols():
        groups = [d.loc[sev == s, c].to_numpy(float) for s in present]
        normal = all(stats.shapiro(g).pvalue > alpha for g in groups)
        homo = stats.levene(*groups).pvalue > alpha
        n = sum(len(g) for g in groups)
        if normal and homo:
            f, p = stats.f_oneway(*groups)
            grand = np.concatenate(groups).mean()
            ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
            sst = sum(((g - grand) ** 2).sum() for g in groups)
            rows.append({"Variable": ds.labels[c], "Test": "One-way ANOVA", "Statistic": f,
                         "df": f"{len(groups) - 1}, {n - len(groups)}", "p": p,
                         "Effect size": f"η² = {ssb / sst:.3f}"})
        else:
            h, p = stats.kruskal(*groups)
            rows.append({"Variable": ds.labels[c], "Test": "Kruskal-Wallis H", "Statistic": h,
                         "df": str(len(groups) - 1), "p": p,
                         "Effect size": f"ε² = {h / (n - 1):.3f}"})
        medians = {s: d.loc[sev == s, c].median() for s in present}
        rows[-1]["Medians by class"] = "; ".join(f"{s.split()[0]} {v:.0f}" for s, v in medians.items())
    for c in ds.cat_cols:
        r = categorical_test(d[c].astype(str), sev, cfg["seed"])
        rows.append({"Variable": CAT_LABELS[c], "Test": r["test"], "Statistic": r["statistic"],
                     "df": str(r.get("df", "")), "p": r["p"],
                     "Effect size": f"Cramér's V = {r['effect']:.3f}", "Medians by class": ""})
    df = pd.DataFrame(rows)
    df["p (Holm)"] = multipletests(df.p, method="holm")[1]
    df["Statistic"] = df.Statistic.round(2)
    df["p"] = df.p.map(_p)
    df["p (Holm)"] = df["p (Holm)"].map(_p)
    return df


def dunn_posthoc(d: pd.DataFrame, ds: Dataset, col: str = "fcri_total") -> pd.DataFrame:
    import scikit_posthocs as sp
    sev = d.severity.astype(str)
    present = [s for s in ds.severity_labels if (sev == s).sum() >= 3]
    sub = d[sev.isin(present)]
    res = sp.posthoc_dunn(sub.assign(g=sub.severity.astype(str)), val_col=col, group_col="g",
                          p_adjust="holm")
    rows = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            rows.append({"Class A": a, "Class B": b,
                         "Median A": sub.loc[sub.severity == a, col].median(),
                         "Median B": sub.loc[sub.severity == b, col].median(),
                         "p (Holm)": _p(res.loc[a, b])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------------
def spearman_table(d: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    rows = []
    n = len(d)
    for c in ["fcri_total"] + subscale_cols():
        rho, p = stats.spearmanr(d[c], d.bdi_total)
        se = np.sqrt((1 + rho ** 2 / 2) / (n - 3))  # Bonett & Wright (2000)
        z = np.arctanh(rho)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        rows.append({"FCRI score": ds.labels[c], "rho": rho, "95% CI": f"{lo:.2f} to {hi:.2f}", "p": p, "n": n})
    df = pd.DataFrame(rows)
    df["p (Holm)"] = multipletests(df.p, method="holm")[1]
    df["rho"] = df.rho.round(3)
    df["p"] = df.p.map(_p)
    df["p (Holm)"] = df["p (Holm)"].map(_p)
    return df


# ----------------------------------------------------------------------------
# Regression
# ----------------------------------------------------------------------------
def _regression_frame(d: pd.DataFrame, ds: Dataset, min_level: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """Recode categorical levels with < min_level participants into 'Other (combined)'."""
    notes = []
    X = d.copy()
    for c in ds.cat_cols:
        s = X[c].astype(str)
        counts = s.value_counts()
        rare = counts[counts < min_level].index.tolist()
        if rare:
            s = s.where(~s.isin(rare), "Other (combined)")
            if (s == "Other (combined)").sum() < min_level:
                notes.append(f"{CAT_LABELS[c]}: {', '.join(rare)} "
                             f"(n = {int((s == 'Other (combined)').sum())}) excluded")
                keep = s != "Other (combined)"
                X, s = X[keep], s[keep]
            else:
                notes.append(f"{CAT_LABELS[c]}: {', '.join(rare)} combined")
        X[c] = s
    return X, notes


def _design(X: pd.DataFrame, ds: Dataset, cont: list[str], cats: list[str]) -> tuple[pd.DataFrame, dict]:
    parts, sds = [], {}
    for c in cont:
        sds[c] = X[c].std(ddof=1)
        parts.append(((X[c] - X[c].mean()) / sds[c]).rename(c))
    for c in cats:
        levels = [lv for lv in ds.df[c].cat.categories if lv in set(X[c])]
        if "Other (combined)" in set(X[c]):
            levels.append("Other (combined)")
        dm = pd.get_dummies(pd.Categorical(X[c], categories=levels), prefix=c, prefix_sep="=",
                            drop_first=True, dtype=float)
        dm.index = X.index
        parts.append(dm)
    return pd.concat(parts, axis=1), sds


def _fit_logit(y: pd.Series, D: pd.DataFrame):
    import statsmodels.api as sm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.Logit(y, sm.add_constant(D, has_constant="add")).fit(disp=0, maxiter=200, method="bfgs"
                                                                         if D.shape[1] > 40 else "newton")


def _or_rows(res, D: pd.DataFrame, ds: Dataset, sds: dict, model: str) -> list[dict]:
    rows = []
    ci = res.conf_int()
    for name in D.columns:
        if "=" in name:
            var, lev = name.split("=", 1)
            label = f"{CAT_LABELS[var]}: {lev}"
        else:
            label = f"{ds.labels[name]} (per SD = {sds[name]:.1f})"
        b = res.params[name]
        rows.append({"Model": model, "Predictor": label, "key": name, "OR": np.exp(b),
                     "CI low": np.exp(ci.loc[name, 0]), "CI high": np.exp(ci.loc[name, 1]),
                     "p": res.pvalues[name]})
    return rows


def logistic_regression(d: pd.DataFrame, ds: Dataset, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """Univariate and multivariable logistic regression of depression (BDI-defined)."""
    X, notes = _regression_frame(d, ds)
    y = X.y_binary.astype(int)
    cats = list(ds.cat_cols)
    cont_main = ["fcri_total"]
    rows: list[dict] = []
    info: dict[str, Any] = {"notes": notes, "n": len(X), "n_depressed": int(y.sum()),
                            "n_not_depressed": int((1 - y).sum())}
    for c in ["fcri_total"] + subscale_cols():
        D, sds = _design(X, ds, [c], [])
        try:
            rows += _or_rows(_fit_logit(y, D), D, ds, sds, "Univariate")
        except Exception as exc:  # separation etc.
            rows.append({"Model": "Univariate", "Predictor": ds.labels[c], "key": c, "OR": np.nan,
                         "CI low": np.nan, "CI high": np.nan, "p": np.nan, "error": str(exc)})
    for c in cats:
        D, sds = _design(X, ds, [], [c])
        try:
            rows += _or_rows(_fit_logit(y, D), D, ds, sds, "Univariate")
        except Exception as exc:
            rows.append({"Model": "Univariate", "Predictor": CAT_LABELS[c], "key": c, "OR": np.nan,
                         "CI low": np.nan, "CI high": np.nan, "p": np.nan, "error": str(exc)})
    D, sds = _design(X, ds, cont_main, cats)
    minority = int(min(y.sum(), (1 - y).sum()))
    info["multivariable_parameters"] = D.shape[1]
    info["events_per_variable"] = round(minority / D.shape[1], 2)
    try:
        res = _fit_logit(y, D)
        rows += _or_rows(res, D, ds, sds, "Multivariable")
        info["multivariable_converged"] = bool(res.mle_retvals.get("converged", True))
        info["pseudo_r2"] = float(res.prsquared)
        info["llr_p"] = float(res.llr_pvalue)
        se = res.bse.drop("const")
        info["max_se"] = float(se.max())
        info["possible_separation"] = bool(se.max() > 10)
    except Exception as exc:
        info["multivariable_error"] = str(exc)
    df = pd.DataFrame(rows)
    return df, info


def format_logistic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["OR (95% CI)"] = out.apply(lambda r: "not estimable" if np.isnan(r.OR) else
                                   f"{r.OR:.2f} ({r['CI low']:.2f}–{r['CI high']:.2f})", axis=1)
    out["p"] = out.p.map(_p)
    return out[["Model", "Predictor", "OR (95% CI)", "p"]]


def ordinal_regression(d: pd.DataFrame, ds: Dataset) -> tuple[pd.DataFrame, dict]:
    """Proportional-odds (ordinal logit) model: severity ~ FCRI total + covariates."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    X, notes = _regression_frame(d, ds)
    D, sds = _design(X, ds, ["fcri_total"], list(ds.cat_cols))
    y = X.severity.cat.codes
    info: dict[str, Any] = {"notes": notes, "n": len(X)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = OrderedModel(y, D, distr="logit").fit(method="bfgs", disp=0, maxiter=2000)
        except Exception as exc:
            info["error"] = str(exc)
            return pd.DataFrame(), info
    ci = res.conf_int()
    rows = []
    for name in D.columns:
        if "=" in name:
            var, lev = name.split("=", 1)
            label = f"{CAT_LABELS[var]}: {lev}"
        else:
            label = f"{ds.labels[name]} (per SD = {sds[name]:.1f})"
        rows.append({"Predictor": label, "key": name,
                     "OR (95% CI)": f"{np.exp(res.params[name]):.2f} ({np.exp(ci.loc[name, 0]):.2f}–{np.exp(ci.loc[name, 1]):.2f})",
                     "OR": float(np.exp(res.params[name])), "p": _p(res.pvalues[name])})
    info["converged"] = bool(res.mle_retvals.get("converged", True))
    info["pseudo_r2"] = float(res.prsquared)
    return pd.DataFrame(rows), info


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_fcri_by_severity(d: pd.DataFrame, ds: Dataset, cfg: dict[str, Any], paths: Paths,
                          name: str, appendix: bool) -> None:
    import matplotlib.pyplot as plt
    st = cfg["style"]
    cols = ["fcri_total"] + subscale_cols()
    fig, axes = plt.subplots(2, 4, figsize=fig_size(cfg, 1.0, 0.62))
    short = ["Nor", "Mild", "Bord", "Mod", "Sev", "Ext"]
    for ax, c in zip(axes.ravel(), cols):
        data = [d.loc[d.severity == s, c].to_numpy() for s in ds.severity_labels]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops={"color": st["text"], "lw": 1.2})
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(st["sequential"][2 + i])
            patch.set_edgecolor(st["muted"])
        ax.set_xticks(range(1, 7), short, rotation=90)
        title = ds.labels[c].replace("FCRI ", "")
        ax.set_title(title[0].upper() + title[1:], fontsize=st["font_size"])
    fig.supxlabel("BDI severity class (Normal → Extreme)", fontsize=st["font_size"])
    fig.supylabel("Score", fontsize=st["font_size"])
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def plot_correlation_heatmap(d: pd.DataFrame, ds: Dataset, cfg: dict[str, Any], paths: Paths,
                             name: str, appendix: bool) -> None:
    import matplotlib.pyplot as plt
    cols = ["bdi_total", "fcri_total"] + subscale_cols()
    C = d[cols].corr(method="spearman")
    labs = ["BDI total", "FCRI total"] + list(FCRI_SUBSCALES)
    fig, ax = plt.subplots(figsize=fig_size(cfg, 1.0, 0.8))
    lo = min(0.0, float(C.min().min()))
    im = ax.imshow(C.to_numpy(), cmap=div_cmap(cfg) if lo < 0 else seq_cmap(cfg),
                   vmin=-1 if lo < 0 else 0, vmax=1)
    ax.set_xticks(range(len(cols)), labs, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), labs)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = C.iloc[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", ".", 1), ha="center", va="center",
                    fontsize=cfg["style"]["font_size"],
                    color="white" if abs(v) > 0.55 else cfg["style"]["text"])
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Spearman ρ")
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


def plot_scatter(ds: Dataset, cfg: dict[str, Any], paths: Paths) -> None:
    import matplotlib.pyplot as plt
    st = cfg["style"]
    subs = cfg["run_subsets"]
    fig, axes = plt.subplots(1, len(subs), figsize=fig_size(cfg, 1.0, 0.45), sharey=True, squeeze=False)
    for ax, sub in zip(axes[0], subs):
        d = ds.subset(sub, cfg)
        ax.scatter(d.fcri_total, d.bdi_total, s=8, alpha=0.35, color=st["categorical"][0], lw=0)
        b = np.polyfit(d.fcri_total, d.bdi_total, 1)
        xs = np.linspace(d.fcri_total.min(), d.fcri_total.max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=st["categorical"][1], lw=2)
        rho = stats.spearmanr(d.fcri_total, d.bdi_total)[0]
        ax.set_title(f"{catalog.subset_label(cfg, sub, short=True)}" + chr(10) + f"(n = {len(d)}, ρ = {rho:.2f})",
                     fontsize=st["font_size"], loc="left")
        ax.set_xlabel("FCRI total")
        for _, lo, _hi in cfg["targets"]["severity_bounds"][1:]:
            ax.axhline(lo - 0.5, color="#d9d8d4", lw=0.7, zorder=0)
    axes[0, 0].set_ylabel("BDI total")
    fig.tight_layout()
    save_fig(fig, paths, catalog.figure("fcri_bdi_scatter", cfg)["name"])


def plot_forest(df: pd.DataFrame, cfg: dict[str, Any], paths: Paths, name: str, appendix: bool) -> None:
    import matplotlib.pyplot as plt
    st = cfg["style"]
    m = df[(df.Model == "Multivariable") & df.OR.notna()].iloc[::-1]
    if m.empty:
        return
    h = max(3.0, 0.22 * len(m) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_size(cfg)[0], h))
    y = np.arange(len(m))
    lo = m["CI low"].clip(lower=1e-3)
    hi = m["CI high"].clip(upper=1e3)
    ax.hlines(y, lo, hi, color=st["muted"], lw=1.2)
    sig = m.p < 0.05
    ax.scatter(m.OR, y, color=np.where(sig, st["categorical"][0], "#9a9993"), s=30, zorder=3)
    ax.axvline(1, color=st["text"], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y, m.Predictor)
    ax.set_xlabel("Adjusted odds ratio of depression (log scale)")
    fig.tight_layout()
    save_fig(fig, paths, name, appendix=appendix)


# ----------------------------------------------------------------------------
# Stage entry point
# ----------------------------------------------------------------------------
def run(ds: Dataset, cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    log = get_logger()
    out: dict[str, Any] = {}
    prim = cfg["primary_subset"]
    for sub in cfg["run_subsets"]:
        d = ds.subset(sub, cfg)
        appendix = sub != prim
        lab = catalog.subset_label(cfg, sub)
        bv = bivariate_binary(d, ds, cfg)
        catalog.save_table(format_bivariate(bv), f"bivariate_binary_{sub}", cfg, paths.tables,
                           note=f"{lab}. Holm correction across all tests in this table. "
                                "Continuous variables: Welch t-test when both groups passed Shapiro-Wilk "
                                "(α = 0.05), otherwise Mann-Whitney U.")
        items = pd.concat([bivariate_binary(d, ds, cfg, include_items="items_only"),
                           bv[bv.Type == "Categorical"]], ignore_index=True)
        items.to_csv(paths.tables / f"appendix_bivariate_items_{sub}.csv", index=False, encoding="utf-8-sig")
        sv = severity_tests(d, ds, cfg)
        catalog.save_table(sv, f"severity_tests_{sub}", cfg, paths.tables,
                           note=f"{lab}. Holm correction across all tests in this table.")
        dn = dunn_posthoc(d, ds)
        catalog.save_table(dn, f"dunn_{sub}", cfg, paths.tables, note=f"{lab}. FCRI total; Dunn's test, Holm-adjusted.")
        for c in subscale_cols():
            dunn_posthoc(d, ds, c).to_csv(paths.tables / f"appendix_dunn_{c}_{sub}.csv", index=False,
                                          encoding="utf-8-sig")
        sp = spearman_table(d, ds)
        catalog.save_table(sp, f"spearman_{sub}", cfg, paths.tables,
                           note=f"{lab}. 95% CI by Fisher z with the Bonett-Wright standard error.")
        lr, lr_info = logistic_regression(d, ds, cfg)
        lr.to_csv(paths.results / f"logistic_raw_{sub}.csv", index=False)
        note = (f"{lab}. Outcome: depressed (BDI ≥ {cfg['targets']['binary_cutoff']}). OR > 1 = higher odds "
                f"of depression. Continuous predictors standardised (per SD). Multivariable model: FCRI total + "
                f"all demographic and clinical variables; {lr_info.get('events_per_variable')} events per "
                "parameter (minority class).")
        if lr_info.get("notes"):
            note += " Rare levels: " + "; ".join(lr_info["notes"]) + "."
        if lr_info.get("possible_separation"):
            note += " Very large standard errors indicate (quasi-)separation; interpret with caution."
        catalog.save_table(format_logistic(lr), f"logistic_{sub}", cfg, paths.tables, note=note)
        od, od_info = ordinal_regression(d, ds)
        if not od.empty:
            catalog.save_table(od.drop(columns=["key", "OR"]), f"ordinal_{sub}", cfg, paths.tables,
                               note=f"{lab}. Proportional-odds logistic model of the six ordered severity "
                                    "classes; OR > 1 = higher odds of a more severe class.")
        fig_sev = catalog.figure("fcri_by_severity", cfg)["name"]
        fig_cor = catalog.figure("correlation_heatmap", cfg)["name"]
        fig_for = catalog.figure("logistic_forest", cfg)["name"]
        suffix = "" if not appendix else f"_{sub}"
        plot_fcri_by_severity(d, ds, cfg, paths, fig_sev + suffix, appendix)
        plot_correlation_heatmap(d, ds, cfg, paths, fig_cor + suffix, appendix)
        plot_forest(lr, cfg, paths, fig_for + suffix, appendix)
        rho_tot = stats.spearmanr(d.fcri_total, d.bdi_total)
        out[sub] = {"logistic": lr_info, "ordinal": od_info,
                    "spearman_fcri_total_bdi": {"rho": float(rho_tot[0]), "p": float(rho_tot[1])},
                    "n_significant_bivariate_holm": int((bv.p_holm < 0.05).sum()),
                    "n_bivariate_tests": int(len(bv))}
        log.info("Stats [%s]: rho(FCRI,BDI)=%.3f, EPV=%s", sub, rho_tot[0], lr_info.get("events_per_variable"))
    plot_scatter(ds, cfg, paths)
    save_json(out, paths.results / "statistics_summary.json")
    return out
