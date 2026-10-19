import argparse
import logging
from pathlib import Path

'''
Command line interface:

    ckks-param-tool run --dataset diabetes [--profile default] [--config cfg.json] [--output results]

runs the tool on one of the built-in workloads (client/server split, model trained on the
server split) and saves the results.
'''


def main(argv=None):
    from ckks_param_tool import Experiment, DataDescriptor, client_server_split
    from ckks_param_tool.config import load_experiment_config
    from ckks_param_tool.config.experiment import SEARCH_PROFILES
    from ckks_param_tool.data.registry import DATASETS

    parser = argparse.ArgumentParser(prog="ckks-param-tool", description="CKKS parameter recommendation tool")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the tool on a built-in workload")
    run.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    run.add_argument("--profile", choices=sorted(SEARCH_PROFILES), default=None,
                     help="search profile (overrides the config file)")
    run.add_argument("--config", type=Path, default=None, help="experiment config JSON")
    run.add_argument("--output", type=Path, default=Path("results"))
    run.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(name)s: %(message)s")

    config = load_experiment_config(args.config).to_dict()
    if args.profile is not None:
        config["search"] = {**config["search"], "profile": args.profile}

    loader, trainer, _ = DATASETS[args.dataset]
    X, y = loader()
    X_client, X_server, _, y_server = client_server_split(X, y)
    model = trainer(X_server, y_server)

    # the client computes its descriptor locally; only the descriptor enters the tool
    experiment = Experiment(DataDescriptor.from_data(X_client), model, config=config)
    experiment.run(verbose=not args.quiet)
    run_dir = experiment.save(args.output / args.dataset)
    print(f"results written to {run_dir}")


if __name__ == "__main__":
    main()
