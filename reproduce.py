'''
Re-creates every benchmark table and figure:

    python reproduce.py                          # everything (MNIST takes several hours)
    python reproduce.py --datasets iris wine     # a subset
    python reproduce.py --no-broad-grid          # skip the reference-grid comparison
    python reproduce.py --output results/final   # where to write (default: results/)

Steps:
  1. measurement noise (benchmark_suite.noise) on the tabular workloads; the suggested runtime
     and precision tolerances are written to <output>/config.json
  2. the benchmark (benchmark_suite.run) with that config, for every workload
  3. the analysis (benchmark_suite.analyze): tables (CSV + LaTeX) and figures in <output>/summary/

Each run records the configuration, the tool's derived quantities, the search time, and the
software / hardware environment and git commit (run_info.json).
'''
import argparse
from pathlib import Path

from benchmark_suite import analyze, noise, run
from benchmark_suite.datasets import DATASETS

TABULAR = [name for name in DATASETS if name != "mnist"]


def main():
    parser = argparse.ArgumentParser(description="Reproduce all benchmark results.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-broad-grid", action="store_true")
    parser.add_argument("--noise-repeats", type=int, default=5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "config.json"

    noise_sets = [d for d in args.datasets if d in TABULAR] or TABULAR
    noise.main(["--datasets", *noise_sets, "--repeats", str(args.noise_repeats),
                "--output", str(args.output / "noise"), "--write-config", str(config)])

    run_args = ["--datasets", *args.datasets, "--output", str(args.output), "--config", str(config)]
    if args.no_broad_grid:
        run_args.append("--no-broad-grid")
    run.main(run_args)

    analyze.main(["--results", str(args.output)])


if __name__ == "__main__":
    main()
