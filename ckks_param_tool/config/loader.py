import json
from pathlib import Path

from .experiment import (
    ExperimentConfig,
    DataConfig,
    EvaluationConfig,
    PlottingConfig,
    SearchPolicyConfig,
)
from .defaults import DEFAULT_EXPERIMENT_CONFIG

'''
This file contains the function that turns a user config (None, a JSON path, a dict, or an
ExperimentConfig) into an ExperimentConfig object. Missing sections and fields keep their
default values.
'''

_SECTIONS = {
    "data": DataConfig,
    "evaluation": EvaluationConfig,
    "plotting": PlottingConfig,
    "search": SearchPolicyConfig,
}


def load_experiment_config(config=None) -> ExperimentConfig:

    if config is None:
        return DEFAULT_EXPERIMENT_CONFIG

    if isinstance(config, ExperimentConfig):
        return config

    if isinstance(config, (str, Path)):
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            config = json.load(f)

    if isinstance(config, dict):
        unknown = set(config) - set(_SECTIONS)
        if unknown:
            raise KeyError(f"Unknown config sections: {sorted(unknown)}")

        return ExperimentConfig(**{
            name: cls(**config.get(name, {}))
            for name, cls in _SECTIONS.items()
        })

    raise TypeError("config must be None, a path, a dict or an ExperimentConfig.")
