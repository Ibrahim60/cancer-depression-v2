"""Data curation: parsing, BDI scoring correction, provenance flags, targets, feature groups.

The questionnaire parsing re-uses ``ClinicalDataPreprocessor`` from
``depression_detection_pipeline.py`` so the thesis and the deployed app share
one scoring implementation. The only deviation is the BDI items 6/7
correction configured in ``config.yaml`` (``data.bdi_zero_statements``).
"""
from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .utils import REPO_ROOT, Paths, get_logger

BDI_TITLES = [
    "Sadness", "Pessimism", "Past failure", "Loss of pleasure", "Guilty feelings",
    "Punishment feelings", "Self-dislike", "Self-criticalness", "Suicidal thoughts", "Crying",
    "Agitation", "Loss of interest", "Indecisiveness", "Body image", "Work inhibition",
    "Sleep disturbance", "Fatigability", "Appetite change", "Weight loss",
    "Somatic preoccupation", "Loss of interest in sex",
]

FCRI_SUBSCALES: dict[str, list[int]] = {
    "Triggers": list(range(1, 9)),
    "Severity": list(range(9, 18)),
    "Psychological distress": list(range(18, 22)),
    "Functioning impairments": list(range(22, 28)),
    "Insight": list(range(28, 31)),
    "Reassurance": list(range(31, 34)),
    "Coping strategies": list(range(34, 43)),
}

CAT_LABELS = {
    "gender": "Gender", "age_group": "Age group", "province": "Province",
    "cancer_type": "Cancer type", "diagnosis_duration": "Time since diagnosis",
    "cancer_status": "Cancer status", "treatment_type": "Treatment type",
    "on_medication": "On psychotropic medication",
}

GROUP_LABELS = {"bdi": "BDI items", "fcri": "FCRI items",
                "demographic": "Demographic", "clinical": "Clinical"}


@dataclass
class Dataset:
    """Curated analysis dataset plus metadata."""
    df: pd.DataFrame                      # one row per participant (no names / timestamps)
    bdi_cols: list[str]
    fcri_cols: list[str]
    cat_cols: list[str]
    groups: dict[str, list[str]]          # group -> raw feature columns
    labels: dict[str, str]                # column -> readable label
    severity_labels: list[str]
    curation_log: list[dict[str, Any]] = field(default_factory=list)
    raw: pd.DataFrame | None = None       # raw CSV (kept for reliability analysis / profiling)

    def subset(self, name: str, cfg: dict[str, Any]) -> pd.DataFrame:
        """Rows belonging to an analysis subset defined in config.subsets."""
        spec = cfg["subsets"][name]
        d = self.df
        if spec.get("exclude_generator_matched"):
            d = d[~d["generator_matched"]]
        return d.reset_index(drop=True)

    def feature_cols(self, feature_set: str, cfg: dict[str, Any],
                     drop: list[str] | None = None) -> list[str]:
        """Raw feature columns for a feature set (A/B/C/D) minus any `drop` columns."""
        groups = cfg["feature_sets"][feature_set]["groups"]
        cols = [c for g in groups for c in self.groups[g]]
        return [c for c in cols if c not in set(drop or [])]


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------
def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str.rstrip(".")


def _row_keys(df: pd.DataFrame, bdi_cols: list[str]) -> pd.Series:
    return _norm(df["Name"]) + "|" + df[bdi_cols].apply(_norm).agg("|".join, axis=1)


def compute_provenance(raw: pd.DataFrame, cfg: dict[str, Any], paths: Paths) -> pd.DataFrame:
    """Flag rows whose Name + 21 BDI answers exactly match mock-generator files in git history.

    Falls back to the cached flag file when git history is unavailable.
    """
    log = get_logger()
    bdi_raw = [f"{i}." for i in range(1, 22)]
    keys = _row_keys(raw, bdi_raw)
    matched = pd.Series(False, index=raw.index)
    source = pd.Series("", index=raw.index, dtype=object)
    ok = True
    for src in cfg["provenance"]["generated_sources"]:
        spec = f"{src['commit']}:{src['path']}"
        try:
            txt = subprocess.check_output(["git", "show", spec], cwd=REPO_ROOT,
                                          stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            ok = False
            log.warning("Provenance: could not read %s from git history", spec)
            continue
        gen = pd.read_csv(io.BytesIO(txt))
        gen.columns = gen.columns.str.strip()
        gkeys = set(_row_keys(gen, [f"BDI_{i}" for i in range(1, 22)]))
        hit = keys.isin(gkeys) & ~matched
        matched |= hit
        source[hit] = spec
        log.info("Provenance: %d rows match %s", int(hit.sum()), spec)
    prov = pd.DataFrame({"row_id": raw.index, "timestamp": raw["Timestamp"].values,
                         "generator_matched": matched.values, "matched_source": source.values})
    if ok:
        paths.provenance_cache.parent.mkdir(parents=True, exist_ok=True)
        prov.to_csv(paths.provenance_cache, index=False)
        return prov
    if paths.provenance_cache.exists():
        cache = pd.read_csv(paths.provenance_cache)
        if len(cache) == len(raw) and (cache["timestamp"].astype(str).values ==
                                       raw["Timestamp"].astype(str).values).all():
            log.warning("Provenance: using cached flags from %s", paths.provenance_cache)
            return cache
    raise RuntimeError(
        "Could not determine row provenance: git history is unavailable and no valid cache "
        f"exists at {paths.provenance_cache}. Run from a full git clone of the repository.")


# ----------------------------------------------------------------------------
# Curation
# ----------------------------------------------------------------------------
def load_dataset(cfg: dict[str, Any], paths: Paths, keep_raw: bool = True) -> Dataset:
    """Load the CSV, score both questionnaires, flag provenance, curate and derive targets."""
    import contextlib
    log = get_logger()
    try:
        raw = pd.read_csv(paths.data)
    except Exception as exc:  # pragma: no cover - surfaced to the user
        raise RuntimeError(f"Could not read {paths.data}: {exc}") from exc
    if "Timestamp" not in raw.columns or "Name" not in raw.columns:
        raise ValueError(f"{paths.data} does not look like the form export "
                         "(missing 'Timestamp'/'Name' columns).")
    curation: list[dict[str, Any]] = [{"step": "Raw form export", "n": len(raw)}]

    with contextlib.redirect_stdout(io.StringIO()):
        from depression_detection_pipeline import ClinicalDataPreprocessor
    zero_statements = {s.lower().rstrip(".") for s in cfg["data"]["bdi_zero_statements"]}

    class CorrectedPreprocessor(ClinicalDataPreprocessor):
        """App preprocessor with the BDI items 6/7 negation fix."""

        def _parse_bdi(self, val):  # noqa: D401
            if not pd.isna(val) and str(val).strip().lower().rstrip(".") in zero_statements:
                return 0
            return super()._parse_bdi(val)

    prov = compute_provenance(raw, cfg, paths)
    work = raw.copy()
    work["row_id"] = np.arange(len(work))
    pre = CorrectedPreprocessor()
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        proc, _ = pre.fit_transform(work)
    for line in buf.getvalue().splitlines():
        if "Dropped" in line:
            curation.append({"step": line.strip(), "n": None})
    curation.append({"step": "After questionnaire quality filters", "n": len(proc)})

    # Sanity check: every BDI item must map its four statements onto 0..3 one-to-one.
    for i, c in enumerate(pre.bdi_columns, 1):
        m = {v: pre._parse_bdi(v) for v in raw[c].dropna().astype(str).str.strip()
             .str.rstrip(".").unique()}
        if sorted(set(m.values())) != [0, 1, 2, 3]:
            raise ValueError(f"BDI item {i} does not map onto scores 0-3: {m}")

    bdi_cols = [f"BDI_{i}" for i in range(1, 22)]
    fcri_cols = [f"FCRI_{i}" for i in range(1, 43)]
    rename = dict(zip(pre.bdi_columns, bdi_cols))
    fcri_text = {}
    for c in pre.fcri_columns:
        n = int(c.strip().split(".")[0])
        rename[c] = f"FCRI_{n}"
        fcri_text[f"FCRI_{n}"] = c.strip().split(". ", 1)[1].strip()
    cat_map = {k: v for k, v in pre.demographic_columns.items() if v is not None}
    cat_rename = {v: {"on_medication": "on_medication"}.get(k, k) for k, v in cat_map.items()}
    df = proc.rename(columns={**rename, **cat_rename})
    cat_cols = [c for c in cfg["feature_groups"]["demographic"] + cfg["feature_groups"]["clinical"]]
    missing = [c for c in cat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Configured feature columns not found in data: {missing}")
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()

    # Harmonise free-text cancer types, then apply exclusions.
    df["cancer_type"] = df["cancer_type"].replace(cfg["data"]["cancer_type_merge"])
    excl = df["cancer_type"].isin([s.strip() for s in cfg["data"]["exclude_cancer_type"]])
    if excl.any():
        curation.append({"step": f"Excluded cancer type {cfg['data']['exclude_cancer_type']}",
                         "n": int(len(df) - excl.sum())})
        df = df[~excl]
    if cfg["data"].get("exclude_under_18"):
        u18 = df["age_group"].eq("Below 18")
        df = df[~u18]
        curation.append({"step": "Excluded participants under 18", "n": len(df)})

    df = df.merge(prov[["row_id", "generator_matched"]], on="row_id", how="left")
    df[bdi_cols + fcri_cols] = df[bdi_cols + fcri_cols].astype(float)

    # Scores
    df["bdi_total"] = df[bdi_cols].sum(axis=1)
    df["fcri_total"] = df[fcri_cols].sum(axis=1)
    for sub, items in FCRI_SUBSCALES.items():
        df[f"fcri_{_slug(sub)}"] = df[[f"FCRI_{i}" for i in items]].sum(axis=1)

    # Targets
    tg = cfg["targets"]
    sev_labels = [b[0] for b in tg["severity_bounds"]]

    def sev(score: float) -> str:
        for lab, lo, hi in tg["severity_bounds"]:
            if lo <= score <= hi:
                return lab
        raise ValueError(f"BDI total {score} outside configured severity bands")

    df["severity"] = pd.Categorical(df["bdi_total"].map(sev), categories=sev_labels, ordered=True)
    df["y_severity6"] = df["severity"].cat.codes.astype(int)
    sev5 = df["severity"].astype(str).replace(tg["severity5_merge"])
    sev5_labels = list(dict.fromkeys(tg["severity5_merge"].get(s, s) for s in sev_labels))
    df["severity5"] = pd.Categorical(sev5, categories=sev5_labels, ordered=True)
    df["y_severity5"] = df["severity5"].cat.codes.astype(int)
    df["y_binary"] = (df["bdi_total"] >= tg["binary_cutoff"]).astype(int)
    df["y_binary17"] = (df["bdi_total"] >= tg["binary_cutoff_sensitivity"]).astype(int)
    df["depression_status"] = np.where(df["y_binary"] == 1, "Depressed", "Not depressed")

    # Integer codes for categorical features (fixed category order; one-hot happens in-fold)
    order = cfg["data"].get("category_order", {})
    for c in cat_cols:
        cats = order.get(c) or sorted(df[c].unique())
        unknown = set(df[c].unique()) - set(cats)
        if unknown:
            raise ValueError(f"Column {c} has values {unknown} missing from data.category_order")
        df[c] = pd.Categorical(df[c], categories=cats, ordered=c in order)
        df[f"{c}__code"] = df[c].cat.codes.astype(int)

    keep = (["row_id", "generator_matched"] + cat_cols + [f"{c}__code" for c in cat_cols]
            + bdi_cols + fcri_cols + ["bdi_total", "fcri_total"]
            + [f"fcri_{_slug(s)}" for s in FCRI_SUBSCALES]
            + ["severity", "severity5", "y_binary", "y_binary17", "y_severity6", "y_severity5",
               "depression_status"])
    df = df[keep].reset_index(drop=True)
    curation.append({"step": "Final curated dataset", "n": len(df)})

    groups = {
        "bdi": bdi_cols, "fcri": fcri_cols,
        "demographic": [f"{c}__code" for c in cfg["feature_groups"]["demographic"]],
        "clinical": [f"{c}__code" for c in cfg["feature_groups"]["clinical"]],
    }
    labels: dict[str, str] = {}
    for i, c in enumerate(bdi_cols):
        labels[c] = f"BDI {i + 1}: {BDI_TITLES[i]}"
    for c in fcri_cols:
        t = fcri_text.get(c, c)
        labels[c] = f"FCRI {c.split('_')[1]}: " + (t if len(t) <= 38 else t[:36].rstrip() + "…")
    for c in cat_cols:
        labels[f"{c}__code"] = CAT_LABELS.get(c, c)
        labels[c] = CAT_LABELS.get(c, c)
    labels["bdi_total"] = "BDI total"
    labels["fcri_total"] = "FCRI total"
    for s in FCRI_SUBSCALES:
        labels[f"fcri_{_slug(s)}"] = f"FCRI {s.lower()}"
    log.info("Curated dataset: %d participants (%d generator-matched)",
             len(df), int(df["generator_matched"].sum()))
    return Dataset(df=df, bdi_cols=bdi_cols, fcri_cols=fcri_cols, cat_cols=cat_cols,
                   groups=groups, labels=labels, severity_labels=sev_labels,
                   curation_log=curation, raw=raw if keep_raw else None)


def _slug(s: str) -> str:
    return s.lower().replace(" ", "_")


def subscale_cols() -> list[str]:
    """Column names of the seven FCRI subscale scores."""
    return [f"fcri_{_slug(s)}" for s in FCRI_SUBSCALES]


def task_spec(task: str, ds: Dataset) -> dict[str, Any]:
    """Target column and class names for a task key."""
    specs = {
        "binary": ("y_binary", ["Not depressed", "Depressed"]),
        "binary17": ("y_binary17", ["BDI < 17", "BDI >= 17"]),
        "severity6": ("y_severity6", ds.severity_labels),
        "severity5": ("y_severity5", list(ds.df["severity5"].cat.categories)),
    }
    if task not in specs:
        raise KeyError(f"Unknown task '{task}'. Choose from {list(specs)}")
    col, names = specs[task]
    return {"target": col, "class_names": names, "binary": task.startswith("binary")}


def variable_of(feature: str) -> str:
    """Raw variable behind a transformed feature name ('province__code_3' -> 'province__code')."""
    if "__code_" in feature:
        return feature.rsplit("_", 1)[0]
    return feature


def readable_feature(feature: str, ds: Dataset) -> str:
    """Human-readable label for raw or one-hot-encoded feature names."""
    if "__code_" in feature:
        var, code = feature.rsplit("_", 1)
        base = var.replace("__code", "")
        try:
            level = ds.df[base].cat.categories[int(float(code))]
        except (ValueError, IndexError, KeyError):
            level = code
        return f"{CAT_LABELS.get(base, base)}: {level}"
    return ds.labels.get(feature, feature)
