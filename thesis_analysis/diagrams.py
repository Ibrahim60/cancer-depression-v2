"""Methodology diagrams (Chapter 3) rendered with Graphviz to PNG (300 dpi) and SVG.

Numbers shown in the diagrams (cut-offs, CV settings, participant counts, the
selected hybrid configuration) are read from config.yaml and from results
files of earlier stages, never typed in by hand. The BDI cut-offs are checked
against ``TargetVariableEngineer.SEVERITY_BOUNDS`` in the training pipeline.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from . import catalog
from .utils import Paths, fig_size, get_logger, save_fig

FONT = "Arial"
FS = "11"
C = {  # light fills + borders derived from the figure palette
    "blue": ("#e6f0fd", "#2a78d6"), "orange": ("#fdeae1", "#c4471b"), "green": ("#e2f5ec", "#14875c"),
    "yellow": ("#fdf1d6", "#a87000"), "grey": ("#f1f1ef", "#6b6a66"), "violet": ("#ecebfb", "#6250d6"),
}
MAX_W_PT = 16 / 2.54 * 72


def _ensure_dot(cfg: dict[str, Any]) -> None:
    if shutil.which("dot"):
        return
    gb = cfg["paths"].get("graphviz_bin")
    if gb and Path(gb, "dot.exe").exists():
        os.environ["PATH"] = str(gb) + os.pathsep + os.environ["PATH"]
        return
    raise RuntimeError(
        "Graphviz 'dot' was not found. Install it with:\n"
        "  winget install --id Graphviz.Graphviz -e --source winget\n"
        "or set paths.graphviz_bin in thesis_analysis/config.yaml.")


def _graph(name: str, rankdir: str = "TB", **attrs):
    import graphviz
    g = graphviz.Digraph(name, format="svg")
    base = {"rankdir": rankdir, "bgcolor": "white", "fontname": FONT, "fontsize": FS, "nodesep": "0.25",
            "ranksep": "0.3", "pad": "0.15"}
    base.update(attrs)
    g.attr(**base)
    g.attr("node", shape="box", style="rounded,filled", fontname=FONT, fontsize=FS, margin="0.12,0.06",
           penwidth="1.2")
    g.attr("edge", color="#52514e", arrowsize="0.7", penwidth="1.1", fontname=FONT, fontsize=FS)
    return g


def _node(g, nid: str, label: str, color: str = "blue", **kw):
    fill, line = C[color]
    g.node(nid, label, fillcolor=fill, color=line, **kw)


def _render(g, paths: Paths, name: str) -> dict[str, Any]:
    """Render to SVG + 300-dpi PNG; report the effective font size at 16 cm width."""
    import re
    src = paths.diagrams_src / f"{name}.gv"
    src.write_text(g.source, encoding="utf-8")
    g.render(filename=name, directory=str(paths.figures), format="svg", cleanup=True)
    import subprocess
    subprocess.run(["dot", "-Tpng", "-Gdpi=300", str(src), "-o", str(paths.figures / f"{name}.png")],
                   check=True)
    svg = (paths.figures / f"{name}.svg").read_text(encoding="utf-8")
    m = re.search(r'width="([\d.]+)pt"', svg)
    w = float(m.group(1)) if m else 0.0
    eff = float(FS) * min(1.0, MAX_W_PT / w) if w else float(FS)
    if eff < 10:
        get_logger().warning("Diagram %s is %.0f pt wide; at 16 cm its text is %.1f pt", name, w, eff)
    return {"width_pt": w, "font_pt_at_16cm": round(eff, 1)}


def _html(title: str, lines: list[str]) -> str:
    from html import escape
    body = "".join(f"<BR/>{escape(ln)}" for ln in lines)
    return f"<<B>{escape(title)}</B>{body}>"


# ----------------------------------------------------------------------------
def methodology_flowchart(cfg, paths, info):
    g = _graph("methodology", "TB", nodesep="0.6", ranksep="0.28")
    left = [
        ("d", "1. Data collection", ["BDI-I (21), FCRI (42),", "demographic & clinical"], "grey"),
        ("c", "2. Data curation", ["scoring, quality filters,", "provenance, targets"], "blue"),
        ("p", "3. Preprocessing", ["encoding, scaling,", "imbalance (in-fold)"], "blue"),
        ("s", "4. Statistical analysis", ["bivariate tests,", "correlation, regression"], "yellow"),
        ("f", "5. Feature selection", ["MI, χ², RFE, L1, Boruta", "(in every training fold)"], "green"),
    ]
    right = [
        ("t", "6. Model training", ["LR, RF, XGBoost, SVM,", "voting, stacking"], "green"),
        ("e", "7. Evaluation", ["nested CV, bootstrap CI,", "significance tests"], "orange"),
        ("x", "8. Explainability", ["SHAP: global, group,", "patient level"], "violet"),
        ("w", "9. Web decision support", ["Streamlit application"], "grey"),
    ]
    for nid, t, lines, col in left + right:
        _node(g, nid, _html(t, lines), col, width="2.45")
    for (a, *_), (b, *_) in zip(left, left[1:]):
        g.edge(a, b)
    for a, b in [("f", "t"), ("s", "e"), ("p", "x"), ("c", "w")]:
        with g.subgraph() as r:
            r.attr(rank="same")
            r.node(a)
            r.node(b)
    g.edge("f", "t")
    for b, a in [("t", "e"), ("e", "x"), ("x", "w")]:
        g.edge(a, b, dir="back")
    return _render(g, paths, catalog.figure("methodology_flowchart", cfg)["name"])


def preprocessing_pipeline(cfg, paths, info):
    g = _graph("preprocessing", "TB", nodesep="0.25", ranksep="0.3", newrank="true", splines="line")
    with g.subgraph(name="cluster_data") as d:
        d.attr(label="Applied once, per participant", style="dashed,rounded",
               color="#2a78d6", fontcolor="#2a78d6", fontname=FONT, fontsize=FS)
        _node(d, "raw", _html("Raw form export", ["text responses"]), "grey", width="2.0")
        _node(d, "parse", _html("Response scoring", ["BDI → 0–3 (items 6/7 fix)", "FCRI → 0–4", "FCRI item 13 reversed"]),
              width="2.0")
        _node(d, "q", _html("Quality filters", ["all-zero FCRI responders;", "> 50% missing items"]), width="2.0")
        _node(d, "imp", _html("Within-person imputation", ["missing item → own", "subscale mean"]), width="2.0")
        _node(d, "cur", _html("Curation", ["cancer-type harmonisation,", "exclusions, totals, targets"]),
              width="2.0")
        for a, b in [("raw", "parse"), ("parse", "q"), ("q", "imp"), ("imp", "cur")]:
            d.edge(a, b)
    with g.subgraph(name="cluster_fold") as f:
        f.attr(label="Fitted on each training fold", style="dashed,rounded", color="#c4471b",
               fontcolor="#c4471b", fontname=FONT, fontsize=FS)
        _node(f, "oh", _html("One-hot encoding", ["categorical variables"]), "orange", width="2.0")
        _node(f, "mm", _html("Min-Max scaling", ["to [0, 1]"]), "orange", width="2.0")
        _node(f, "sm", _html("SMOTE (optional)", ["training data only"]), "orange", width="2.0")
        _node(f, "fs", _html("Feature selection", ["MI / χ² / RFE / L1 / Boruta"]), "orange", width="2.0")
        _node(f, "clf", _html("Classifier", ["base learner or ensemble"]), "green", width="2.0")
        for a, b in [("oh", "mm"), ("mm", "sm"), ("sm", "fs"), ("fs", "clf")]:
            f.edge(a, b)
    for a, b in [("raw", "oh"), ("parse", "mm"), ("q", "sm"), ("imp", "fs"), ("cur", "clf")]:
        with g.subgraph() as r:
            r.attr(rank="same")
            r.node(a)
            r.node(b)
    g.edge("cur", "oh", constraint="false", tailport="e", headport="w")
    return _render(g, paths, catalog.figure("preprocessing", cfg)["name"])


def hybrid_framework(cfg, paths, info):
    from .feature_selection import FS_LABELS
    from .experiments import MODEL_LABELS
    hyb = info.get("hybrid", {})
    prim = cfg["primary_subset"]
    g = _graph("hybrid", "TB", nodesep="0.18", ranksep="0.32")
    _node(g, "in", _html(f"Input: feature set {cfg['key_feature_set']}",
                         ["42 FCRI items + demographic + clinical variables"]), "grey")
    _node(g, "pre", _html("Preprocessing (in-fold)", ["one-hot encoding, Min-Max scaling, class weights"]))
    _node(g, "fs", _html("Feature selection (in-fold)", ["MI or χ² filter, RFE (LR), L1-LR or Boruta"]), "green")
    with g.subgraph(name="cluster_base") as b:
        b.attr(label="Base learners (tuned in the inner CV loop)", style="dashed,rounded", color="#6b6a66",
               fontname=FONT, fontsize=FS, labelloc="b")
        for m, lab in (("lr", "Logistic\nregression"), ("rf", "Random\nforest"), ("xgb", "XGBoost"),
                       ("svm", "SVM\n(RBF)")):
            _node(b, m, lab, "blue", width="1.05")
    _node(g, "vote", _html("Soft voting", ["mean of probabilities"]), "orange")
    _node(g, "stack", _html("Stacking", ["LR meta-learner"]), "orange")
    _node(g, "o1", _html("Output 1: binary", ["depressed / not depressed"]), "violet")
    _node(g, "o2", _html("Output 2: severity", ["six BDI classes"]), "violet")
    g.edge("in", "pre")
    g.edge("pre", "fs")
    for m in ("lr", "rf", "xgb", "svm"):
        g.edge("fs", m)
        g.edge(m, "vote")
        g.edge(m, "stack")
    for o in ("o1", "o2"):
        g.edge("vote", o)
        g.edge("stack", o)
    lines = []
    for task, lab in (("binary", "binary"), ("severity6", "severity")):
        if task in hyb and prim in hyb[task]:
            h = hyb[task][prim]
            lines.append(f"{lab}: {FS_LABELS[h['fs']]} + {MODEL_LABELS[h['model']]}")
    if lines:
        _node(g, "sel", _html("Selected hybrid (nested CV, macro F1)", lines), "yellow")
        g.edge("o1", "sel", style="invis")
        g.edge("o2", "sel", style="invis")
    return _render(g, paths, catalog.figure("hybrid_framework", cfg)["name"])


def nested_cv(cfg, paths, info):
    cv = cfg["cv"]
    g = _graph("nestedcv", "TB")
    _node(g, "data", _html("Analysis dataset", ["stratified by class"]), "grey")
    _node(g, "outer", _html(f"Outer loop: {cv['outer_splits']}-fold stratified CV × {cv['outer_repeats']} repeats",
                            [f"{cv['outer_splits'] * cv['outer_repeats']} train/test splits"]), "orange")
    with g.subgraph(name="cluster_o") as o:
        o.attr(label="For each outer split", style="dashed,rounded", color="#c4471b", fontcolor="#c4471b",
               fontname=FONT, fontsize=FS)
        _node(o, "otr", _html("Outer training fold", ["(≈80% of participants)"]), "blue")
        _node(o, "ote", _html("Outer test fold", ["(≈20%, never used for tuning)"]), "yellow")
        with o.subgraph(name="cluster_i") as i:
            i.attr(label="Inner loop", style="dashed,rounded", color="#2a78d6", fontcolor="#2a78d6",
                   fontname=FONT, fontsize=FS)
            _node(i, "inner", _html(f"{cv['inner_splits']}-fold stratified CV",
                                    [f"RandomizedSearchCV, {cv['n_iter']} candidates",
                                     f"score: {cv['inner_scoring'].replace('_', ' ')}"]), "blue")
            _node(i, "best", _html("Best hyper-parameters", ["incl. FS parameter"]), "blue")
            i.edge("inner", "best")
        _node(o, "refit", _html("Refit pipeline on outer training fold", ["build voting & stacking"]), "green")
        _node(o, "pred", _html("Predict outer test fold", ["store probabilities"]), "green")
        o.edge("otr", "inner"); o.edge("best", "refit"); o.edge("refit", "pred"); o.edge("ote", "pred")
    _node(g, "agg", _html("Aggregate", ["mean ± SD over folds, bootstrap 95% CI,",
                                        "paired significance tests"]), "violet")
    g.edge("data", "outer"); g.edge("outer", "otr"); g.edge("outer", "ote"); g.edge("pred", "agg")
    return _render(g, paths, catalog.figure("nested_cv", cfg)["name"])


def web_architecture(cfg, paths, info):
    g = _graph("web", "TB", nodesep="0.3", ranksep="0.3")
    _node(g, "user", _html("User", ["patient or clinician"]), "grey")
    _node(g, "a", _html("Assessment page", ["Quick or Full mode;", "demographics → BDI → FCRI;",
                                            "BDI item 9 safety check"]))
    _node(g, "fv", _html("build_feature_vector()", ["label encoders; unanswered items",
                                                     "→ training means"]), "green")
    _node(g, "sc", _html("StandardScaler", ["binary_scaler.joblib"]), "green")
    with g.subgraph(name="cluster_m") as m:
        m.attr(label="Saved models (models/)", style="dashed,rounded", color="#14875c", fontcolor="#14875c",
               fontname=FONT, fontsize=FS)
        _node(m, "mb", _html("Binary ensemble", ["soft voting, tuned threshold"]), "yellow")
        _node(m, "mm", _html("Severity ensemble", ["soft voting, 6 classes"]), "yellow")
    _node(g, "pred", _html("predict() + final_verdict()", ["probabilities, severity, advice"]), "orange")
    _node(g, "r", _html("Results page", ["BDI & FCRI reference scores,", "model verdict, CSV/PDF export"]))
    _node(g, "h", _html("History page", ["session history, CSV export"]))
    _node(g, "shap", _html("SHAP explanation", ["shap_analyzer.py;", "command-line predictor only"]),
          "violet", style="rounded,filled,dashed")
    for u, v in [("user", "a"), ("a", "fv"), ("fv", "sc"), ("sc", "mb"), ("sc", "mm"), ("mb", "pred"),
                 ("mm", "pred"), ("pred", "r"), ("r", "h")]:
        g.edge(u, v)
    g.edge("pred", "shap", style="dashed")
    with g.subgraph() as r:
        r.attr(rank="same")
        r.node("r")
        r.node("shap")
    return _render(g, paths, catalog.figure("web_architecture", cfg)["name"])


def curation_flow(cfg, paths, info):
    q = info.get("quality")
    counts = info.get("class_counts")
    if not q or not counts:
        get_logger().warning("Curation-flow diagram skipped: run the 'eda' stage first.")
        return {}
    g = _graph("curation", "TB", nodesep="0.2")
    cur = q["curation"]
    n_raw = int(cur[0]["n"])
    final_n = int([s["n"] for s in cur if s["step"].startswith("Final")][0])
    lines = []
    prev = n_raw
    for s in cur[1:]:
        if s["n"] is None:
            lines.append(s["step"].replace("Dropped ", "").replace(" rows with >50% missing questionnaire items", " with > 50% items missing"))
        elif s["step"].startswith("Excluded cancer type"):
            types = s["step"].split("[", 1)[1].rstrip("]").replace("'", "")
            lines.append(f"cancer type '{types}' (n = {int(prev - s['n'])})")
        elif s["step"].startswith("Excluded"):
            lines.append(f"{s['step']} (n = {int(prev - s['n'])})")
        if s["n"] is not None:
            prev = s["n"]
    _node(g, "raw", _html(f"Raw form export (n = {n_raw})", []), "grey")
    _node(g, "ex", _html(f"Excluded (n = {n_raw - final_n})", lines), "orange")
    gm = q["quality"]["generator_matched_rows"]
    _node(g, "fin", _html(f"Curated dataset (n = {final_n})",
                          [f"provenance flag: {gm} rows match", "mock-generator output"]))
    g.edge("raw", "fin")
    g.edge("raw", "ex", style="dashed")
    for sname in cfg["run_subsets"]:
        c = counts[sname]
        b = {str(k): v for k, v in c["binary"].items()}
        _node(g, f"s_{sname}", _html(catalog.subset_label(cfg, sname, short=True),
                                     [f"n = {c['n']}", f"depressed {b['1']}", f"not depressed {b['0']}"]), "green")
        g.edge("fin", f"s_{sname}")
    return _render(g, paths, catalog.figure("curation_flow", cfg)["name"])


def bdi_mapping(cfg, paths, info):
    """BDI total -> six severity classes and binary target (equal-width boxes; not to scale)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from depression_detection_pipeline import TargetVariableEngineer
    bounds = [tuple(b) for b in cfg["targets"]["severity_bounds"]]
    code = [tuple(b) for b in TargetVariableEngineer.SEVERITY_BOUNDS]
    if bounds != code:
        raise ValueError(f"Severity cut-offs in config.yaml {bounds} differ from the training code {code}")
    st = cfg["style"]
    fs = st["font_size"]
    cut = cfg["targets"]["binary_cutoff"]
    fig, ax = plt.subplots(figsize=fig_size(cfg, 1.0, 0.36))
    shades = st["sequential"][1:7]
    w, gap = 1.0, 0.06
    names = {"Mild mood disturbance": "Mild mood\ndisturbance",
             "Borderline clinical depression": "Borderline\nclinical",
             "Moderate depression": "Moderate", "Severe depression": "Severe",
             "Extreme depression": "Extreme"}
    for i, ((lab, lo, hi), col) in enumerate(zip(bounds, shades)):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 0.9, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=col, ec="white"))
        dark = i >= 3
        ax.text(x + w / 2, 1.62, names.get(lab, lab), ha="center", va="center", fontsize=fs,
                color="white" if dark else st["text"], linespacing=1.0)
        ax.text(x + w / 2, 1.18, f"{lo}–{hi}", ha="center", va="center", fontsize=fs, fontweight="bold",
                color="white" if dark else st["text"])
    n_not = sum(1 for _, lo, hi in bounds if hi < cut)
    xs = [(0, n_not * (w + gap) - gap, st["categorical"][1], f"Not depressed\n(BDI < {cut})"),
          (n_not * (w + gap), len(bounds) * (w + gap) - gap, st["categorical"][0], f"Depressed (BDI ≥ {cut})")]
    for x0, x1, col, txt in xs:
        ax.add_patch(FancyBboxPatch((x0, 0.2), x1 - x0, 0.55, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=col, ec="white"))
        ax.text((x0 + x1) / 2, 0.475, txt, ha="center", va="center", color="white", fontsize=fs)
    ax.text(-0.08, 1.45, "Severity\ntarget", ha="right", va="center", fontsize=fs)
    ax.text(-0.08, 0.475, "Binary\ntarget", ha="right", va="center", fontsize=fs)
    ax.set_xlim(-1.0, len(bounds) * (w + gap))
    ax.set_ylim(0.1, 2.0)
    ax.axis("off")
    ax.set_title("BDI total score (sum of 21 items, 0–63); boxes are not drawn to scale", fontsize=fs,
                 loc="center")
    fig.tight_layout()
    save_fig(fig, paths, catalog.figure("bdi_mapping", cfg)["name"])
    return {"bounds_match_code": True}


def run(cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    from .utils import apply_style
    _ensure_dot(cfg)
    apply_style(cfg)
    info: dict[str, Any] = {}
    for fname, key in (("hybrid_selection.json", "hybrid"), ("class_counts.json", "class_counts")):
        p = paths.results / fname
        if p.exists():
            import json
            info[key] = json.loads(p.read_text(encoding="utf-8"))
    qp = paths.results / "data_quality.json"
    if qp.exists():
        import json
        info["quality"] = json.loads(qp.read_text(encoding="utf-8"))
    out = {}
    for fn in (methodology_flowchart, curation_flow, bdi_mapping, preprocessing_pipeline, hybrid_framework,
               nested_cv, web_architecture):
        out[fn.__name__] = fn(cfg, paths, info)
    return out
