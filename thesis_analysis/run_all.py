"""Command-line entry point for the thesis analysis.

Usage (PowerShell, from the repository root, venv activated):

    python -m thesis_analysis.run_all --stage all
    python -m thesis_analysis.run_all --stage eda
    python -m thesis_analysis.run_all --stage stats
    python -m thesis_analysis.run_all --stage train      # nested CV (several hours)
    python -m thesis_analysis.run_all --stage fs         # feature-selection stability
    python -m thesis_analysis.run_all --stage eval       # metrics, CIs, significance, curves
    python -m thesis_analysis.run_all --stage shap
    python -m thesis_analysis.run_all --stage diagrams
    python -m thesis_analysis.run_all --stage report     # Word tables + RESULTS_SUMMARY.md + THESIS_GUIDE.docx

Stages after ``train`` reuse the saved predictions in thesis_outputs/results/.
Use ``--force`` to recompute cached cross-validation runs.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings

from .utils import ConfigError, apply_style, get_logger, get_paths, load_config, set_seed, write_run_record

STAGES = ["eda", "stats", "train", "fs", "eval", "shap", "diagrams", "report"]


def stage_eda(ctx):
    from . import data_profile
    return data_profile.run(ctx["ds"], ctx["cfg"], ctx["paths"])


def stage_stats(ctx):
    from . import statistics
    return statistics.run(ctx["ds"], ctx["cfg"], ctx["paths"])


def stage_train(ctx):
    from .evaluation import choose_hybrid, summarise_run
    from .experiments import load_run, plan_followup, plan_main, run_plan
    from .utils import save_json
    import pandas as pd
    ds, cfg, paths, log = ctx["ds"], ctx["cfg"], ctx["paths"], ctx["log"]
    main = plan_main(cfg)
    log.info("Main design: %d runs x %d outer folds", len(main),
             cfg["cv"]["outer_splits"] * cfg["cv"]["outer_repeats"])
    ids = run_plan(ds, main, cfg, paths, force=ctx["force"])
    quick = pd.concat([summarise_run(load_run(i, paths), cfg, with_ci=False) for i in ids])
    hybrid = choose_hybrid(quick, cfg)
    save_json(hybrid, paths.results / "hybrid_selection.json")
    log.info("Hybrid selection: %s", hybrid)
    follow = plan_followup(cfg, {t: {s: v for s, v in d.items()} for t, d in hybrid.items()})
    ids2 = run_plan(ds, follow, cfg, paths, force=ctx["force"])
    save_json({"main": ids, "followup": ids2}, paths.results / "run_index.json")
    return {"n_runs": len(ids) + len(ids2), "hybrid": hybrid}


def stage_fs(ctx):
    from . import feature_selection
    return feature_selection.run(ctx["ds"], ctx["cfg"], ctx["paths"])


def stage_eval(ctx):
    from . import evaluation
    return evaluation.run(ctx["ds"], ctx["cfg"], ctx["paths"])


def stage_shap(ctx):
    from . import explainability
    return explainability.run(ctx["ds"], ctx["cfg"], ctx["paths"])


def stage_diagrams(ctx):
    from . import diagrams
    return diagrams.run(ctx["cfg"], ctx["paths"])


def stage_report(ctx):
    from . import report
    return report.run(ctx["ds"], ctx["cfg"], ctx["paths"])


FUNCS = {"eda": stage_eda, "stats": stage_stats, "train": stage_train, "fs": stage_fs,
         "eval": stage_eval, "shap": stage_shap, "diagrams": stage_diagrams, "report": stage_report}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Thesis analysis pipeline")
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES,
                    help="which stage to run (default: all)")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--force", action="store_true", help="recompute cached CV runs")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    paths = get_paths(cfg)
    log = get_logger(paths, run_name=f"thesis_{args.stage}")
    set_seed(cfg["seed"])
    apply_style(cfg)

    from .data import load_dataset
    ds = None
    stages = STAGES if args.stage == "all" else [args.stage]
    if any(s != "diagrams" for s in stages):
        ds = load_dataset(cfg, paths)
    ctx = {"cfg": cfg, "paths": paths, "log": log, "ds": ds, "force": args.force}
    summary = {}
    for st in stages:
        log.info("=" * 70)
        log.info("STAGE: %s", st)
        t0 = time.time()
        try:
            res = FUNCS[st](ctx) or {}
        except Exception as exc:  # report clearly, keep the traceback in the log
            log.error("Stage '%s' failed: %s", st, exc)
            log.error(traceback.format_exc())
            write_run_record(paths, cfg, st, {"status": "failed", "error": str(exc)})
            return 1
        summary[st] = {"seconds": round(time.time() - t0, 1)}
        rec = write_run_record(paths, cfg, st, {"status": "ok", "result": res,
                                                "seconds": summary[st]["seconds"]})
        log.info("Stage '%s' finished in %.0f s (record: %s)", st, time.time() - t0, rec.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
