"""Command line entry point; every operation derives from the supplied config."""

import argparse
from pathlib import Path
import sys

from .analysis import analyze, export_analysis
from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tesseract geometry study and experimental CAD (mm, degrees)")
    parser.add_argument("command", choices=["analyze", "render", "cad", "all"])
    parser.add_argument("--config", type=Path, help="TOML settings; built-in defaults when omitted")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        print(f"Analyzing {config.model.plane} / {config.model.projection}, config {config.digest()}…", flush=True)
        study = analyze(config)
        export_analysis(study, args.output)
        if args.command in ("render", "all"):
            from .render import render
            render(study, args.output)
        if args.command in ("cad", "all"):
            from .cad import export_cad
            export_cad(study, args.output / "cad")
        for name in ("prototype_candidate", "target_candidate"):
            s = study.scans[name]["summary"]
            print(f"{name}: {s['blocked_samples']}/{s['samples']} blocked; minimum clearance {s['min_clearance_mm']} mm")
        print(f"Outputs: {args.output.resolve()}")
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"tesseract: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
