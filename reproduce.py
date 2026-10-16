'''
Re-creates every benchmark table and figure:

    python reproduce.py                      # all workloads (MNIST takes several hours)
    python reproduce.py --datasets iris wine # a subset
    python reproduce.py --no-fuller          # skip the near-exhaustive comparison

Results go to results/<dataset>/<timestamp>/ and the summary tables (CSV + LaTeX) and figures to
results/summary/. Each run records the configuration, precision estimates, search time and the
software/hardware environment (run_info.json).
'''
import sys

from benchmark_suite import analyze, run


if __name__ == "__main__":
    run.main(sys.argv[1:])
    analyze.main(["--results", "results"] if "--output" not in sys.argv else
                 ["--results", sys.argv[sys.argv.index("--output") + 1]])
