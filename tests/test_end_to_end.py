import json

import numpy as np
import pytest

pytest.importorskip("tenseal")

from ckks_param_tool import Experiment, client_server_split, load_diabetes_data, train_linear_regression_model


def test_fast_profile_end_to_end(tmp_path):
    X, y = load_diabetes_data()
    X_client, X_server, y_client, y_server = client_server_split(X, y)
    model = train_linear_regression_model(X_server, y_server)

    config = {"search": {"profile": "fast"}, "data": {"num_samples": 3}, "evaluation": {"num_samples": 3}}
    exp = Experiment(X_client, model, config=config)
    result = exp.run()

    assert result.param_search_results
    assert all(r.status == "success" for _, r in result.param_search_results)
    assert {sec for sec, _, _ in result.recommended_params} == {128, 192, 256}
    for _, _, inference in result.recommended_params:
        assert np.isfinite(inference.error)

    run_dir = exp.save(tmp_path)
    for name in ("config.json", "param_search.csv", "infeasible.csv", "recommended_params.json", "run_info.json"):
        assert (run_dir / name).exists()
    assert json.loads((run_dir / "config.json").read_text())["search"]["profile"] == "fast"


def test_no_feasible_candidates_does_not_crash(tmp_path):
    from ckks_param_tool import load_breast_cancer_data
    from ckks_param_tool.models import LinearModel

    X, _ = load_breast_cancer_data()
    model = LinearModel(np.ones(X.shape[1]), 0.0)

    # a single 2^10 scale has no NTT-friendly primes for N >= 8192
    config = {"search": {"profile": "fast", "base_scale_buffer": 0, "budget_scales": False},
              "data": {"target_precision_bits": 1}}
    exp = Experiment(X, model, config=config)
    result = exp.run()

    assert result.param_search_results == []
    assert result.infeasible
    assert result.recommended_params == []
    exp.save(tmp_path)
