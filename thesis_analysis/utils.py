"""Shared helpers: configuration, logging, reproducibility, figure style and I/O."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"

try:  # Windows consoles default to cp1252; keep logging robust.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing, malformed or points at missing files."""


@dataclass
class Paths:
    """All input/output locations used by the analysis."""
    root: Path
    data: Path
    out: Path
    figures: Path
    appendix: Path
    tables: Path
    logs: Path
    results: Path
    data_out: Path
    diagrams_src: Path
    provenance_cache: Path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate config.yaml."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    required = ["seed", "paths", "data", "targets", "feature_groups", "feature_sets",
                "cv", "models", "feature_selection", "style", "subsets"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ConfigError(f"{path} is missing required sections: {missing}")
    data_path = REPO_ROOT / cfg["paths"]["data"]
    if not data_path.exists():
        raise ConfigError(
            f"Dataset not found at {data_path}. Set paths.data in {path} "
            "to the location of cancer_psychology.csv (relative to the repo root).")
    for s in cfg["run_subsets"]:
        if s not in cfg["subsets"]:
            raise ConfigError(f"run_subsets lists '{s}', which is not defined under subsets:")
    cfg["_config_path"] = str(path)
    return cfg


def get_paths(cfg: dict[str, Any]) -> Paths:
    """Resolve and create every output directory."""
    out = REPO_ROOT / cfg["paths"]["output_dir"]
    p = Paths(
        root=REPO_ROOT,
        data=REPO_ROOT / cfg["paths"]["data"],
        out=out,
        figures=out / "figures",
        appendix=out / "figures" / "appendix",
        tables=out / "tables",
        logs=out / "logs",
        results=out / "results",
        data_out=out / "data",
        diagrams_src=out / "diagrams_src",
        provenance_cache=REPO_ROOT / cfg["paths"]["provenance_cache"],
    )
    for d in (p.out, p.figures, p.appendix, p.tables, p.logs, p.results, p.data_out, p.diagrams_src):
        d.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Fix every random number generator we use."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def git_commit() -> str:
    """Current git commit (with a '-dirty' suffix when the tree has uncommitted changes)."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def library_versions() -> dict[str, str]:
    """Versions of every library that affects results (reported in the thesis)."""
    import importlib
    names = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
             "xgboost": "xgboost", "imbalanced-learn": "imblearn", "statsmodels": "statsmodels",
             "scikit-posthocs": "scikit_posthocs", "shap": "shap", "matplotlib": "matplotlib",
             "python-docx": "docx", "graphviz (python)": "graphviz", "PyYAML": "yaml"}
    out = {"python": platform.python_version()}
    for label, mod in names.items():
        try:
            m = importlib.import_module(mod)
            out[label] = getattr(m, "__version__", "installed")
        except ImportError:
            out[label] = "not installed"
    return out


_LOGGER: logging.Logger | None = None


def get_logger(paths: Paths | None = None, run_name: str = "run") -> logging.Logger:
    """Logger writing to the console and thesis_outputs/logs/<run>_<timestamp>.log."""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger("thesis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if paths is not None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(paths.logs / f"{run_name}_{stamp}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _LOGGER = logger
    return logger


def write_run_record(paths: Paths, cfg: dict[str, Any], stage: str, extra: dict[str, Any]) -> Path:
    """Write a JSON record of a run: timestamp, config, git commit, versions and metrics."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "git_commit": git_commit(),
        "config_path": cfg.get("_config_path"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "library_versions": library_versions(),
        **extra,
    }
    path = paths.logs / f"run_{stage}_{stamp}.json"
    path.write_text(json.dumps(rec, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj: Any, path: Path) -> None:
    """Write JSON with numpy-aware serialisation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path) -> Any:
    """Read a JSON results file with a helpful error when the producing stage has not run."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the stage that creates it first "
                                "(see `python -m thesis_analysis.run_all --help`).")
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path, stage: str) -> Path:
    """Fail with a clear message when an input produced by an earlier stage is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}.\nRun:  python -m thesis_analysis.run_all --stage {stage}")
    return path


# ----------------------------------------------------------------------------
# Figure style
# ----------------------------------------------------------------------------
CM = 1 / 2.54


def apply_style(cfg: dict[str, Any]) -> None:
    """Consistent academic matplotlib style: white background, fonts >= 10 pt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = cfg["style"]
    fs = st["font_size"]
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": st["dpi"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": st["font_family"],
        "font.size": fs,
        "axes.titlesize": fs + 1,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "figure.titlesize": fs + 1,
        "axes.edgecolor": st["muted"],
        "axes.labelcolor": st["text"],
        "text.color": st["text"],
        "xtick.color": st["muted"],
        "ytick.color": st["muted"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
        "axes.prop_cycle": matplotlib.cycler(color=st["categorical"]),
        "svg.fonttype": "none",
        "legend.frameon": False,
    })


def fig_size(cfg: dict[str, Any], width_frac: float = 1.0, aspect: float = 0.62) -> tuple[float, float]:
    """Figure size in inches; width capped at style.max_width_cm."""
    w = cfg["style"]["max_width_cm"] * CM * width_frac
    return (w, w * aspect)


_MAX_WIDTH_IN = 16 * CM


_PAD_IN = 0.05


def panel_title(ax, text: str, fontsize: float) -> None:
    """Panel heading anchored at the figure's left edge (not the axes), so long headings
    above axes with long tick labels do not widen the figure."""
    ax.annotate(text, xy=(0.01, 1.0), xycoords=("figure fraction", "axes fraction"),
                xytext=(0, 6), textcoords="offset points", ha="left", va="bottom", fontsize=fontsize)


def _fit_width(fig, max_in: float) -> None:
    """Shrink the canvas until the saved image (tight box + padding) fits max_in.

    Fonts keep their size; stops early rather than squeezing any axes below 1.5 inches.
    """
    target = max_in - 2 * _PAD_IN
    for _ in range(12):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        if bb.width <= target:
            return
        if any(ax.get_position().width * fig.get_size_inches()[0] < 1.5 for ax in fig.axes if ax.get_visible()
               and ax.axison):
            return
        w, h = fig.get_size_inches()
        fig.set_size_inches(max(w - (bb.width - target) * 1.2 - 0.01, 2.0), h)
        try:
            fig.tight_layout()
        except Exception:  # complex layouts
            pass


def save_fig(fig, paths: Paths, name: str, appendix: bool = False) -> list[Path]:
    """Save a figure as 300-dpi PNG and SVG, re-flowed so it is never wider than 16 cm."""
    import matplotlib.pyplot as plt
    _fit_width(fig, _MAX_WIDTH_IN)
    folder = paths.appendix if appendix else paths.figures
    out = []
    for ext in ("png", "svg"):
        p = folder / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=_PAD_IN)
        out.append(p)
    plt.close(fig)
    return out


def seq_cmap(cfg: dict[str, Any]):
    """Single-hue sequential colormap (light to dark blue)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("thesis_seq", cfg["style"]["sequential"])


def div_cmap(cfg: dict[str, Any]):
    """Two-hue diverging colormap with a neutral grey midpoint."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("thesis_div", cfg["style"]["diverging"])
