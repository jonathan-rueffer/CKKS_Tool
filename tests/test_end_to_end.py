import json

import numpy as np
import pytest

pytest.importorskip("tenseal")

from ckks_param_tool import Experiment, DataDescriptor, client_server_split, load_diabetes_data, train_linear_regression_model


def test_fast_profile_end_to_end(tmp_path):
    X, y = load_diabetes_data()
    X_client, X_server, y_client, y_server = client_server_split(X, y)
    model = train_linear_regression_model(X_server, y_server)

    config = {"search": {"profile": "fast"}, "data": {"num_samples": 3}, "evaluation": {"num_samples": 3}}
    exp = Experiment(DataDescriptor.from_data(X_client), model, config=config)
    result = exp.run()

    assert result.param_search_results
    assert all(r.status == "success" for _, r in result.param_search_results)
    assert {sec for sec, _, _ in result.recommended_params} == {128, 192, 256}
    for _, _, tv in result.recommended_params:
        assert np.isfinite(tv.error)

    # the cost of evaluating each candidate remotely is measured
    for _, r in result.param_search_results:
        assert r.eval_key_bytes > 0 and r.input_ciphertext_bytes > 0 and r.encrypted_queries >= 3

    run_dir = exp.save(tmp_path)
    for name in ("config.json", "descriptor.json", "param_search.csv", "infeasible.csv",
                 "recommended_params.json", "run_info.json"):
        assert (run_dir / name).exists()
    assert json.loads((run_dir / "config.json").read_text())["search"]["profile"] == "fast"

    # the metadata needed to interpret a run without the source tree
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["environment"]["seal"] and info["environment"]["packages"]["tenseal"]
    assert info["environment"]["git"]["commit"] is not None
    assert info["search_policy"]["poly_degrees"] == [8192, 16384] # the resolved "fast" profile
    assert info["search_policy"]["step_down_bits"] == 5
    assert "2^output_headroom_bits" in info["safety_checks"]["catastrophic_deviation"]
    assert info["output_margin_bits"] == 3 and "output_int_bits" in info
    for entry in json.loads((run_dir / "recommended_params.json").read_text()):
        assert {"coefficient_headroom_bits", "coefficient_headroom_below_margin",
                "output_headroom_bits", "max_coefficient"} <= set(entry)


def test_no_feasible_candidates_does_not_crash(tmp_path):
    from ckks_param_tool import load_breast_cancer_data
    from ckks_param_tool.models import LinearModel

    X, _ = load_breast_cancer_data()
    model = LinearModel(np.ones(X.shape[1]), 0.0)

    # a single 2^10 scale has no NTT-friendly primes for N >= 8192
    config = {"search": {"profile": "fast", "budget_scales": False, "num_scale_options": 1,
                         "min_scale_bits": 10}}
    exp = Experiment(DataDescriptor.from_data(X), model, config=config)
    result = exp.run()

    assert result.param_search_results == []
    assert result.infeasible
    assert result.recommended_params == []
    exp.save(tmp_path)


def test_descriptor_only_run(tmp_path):
    from ckks_param_tool import load_breast_cancer_data
    from ckks_param_tool.models import train_logistic_linear_regression_model

    X, y = load_breast_cancer_data()
    X_client, X_server, _, y_server = client_server_split(X, y)
    model = train_logistic_linear_regression_model(X_server, y_server)

    config = {"search": {"profile": "fast"}, "data": {"num_samples": 3, "magnitude_samples": 200}}
    exp = Experiment(DataDescriptor.from_data(X_client), model, config=config)
    result = exp.run()

    assert result.estimated_required_bits is not None # classifier: margin-based estimate
    assert result.recommended_params
    assert {label for _, label, _, _ in result.front_points} == {"fastest", "most_precise"}

    run_dir = exp.save(tmp_path)
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["estimated_required_bits"] == result.estimated_required_bits
    assert (run_dir / "descriptor.json").exists()


def test_target_reports_fastest_meeting_it(tmp_path):
    X, y = load_diabetes_data()
    X_client, X_server, _, y_server = client_server_split(X, y)
    model = train_linear_regression_model(X_server, y_server)

    config = {"search": {"profile": "fast"}, "data": {"num_samples": 3, "magnitude_samples": 200,
              "target_precision_bits": 15}, "evaluation": {"num_samples": 3}}
    result = Experiment(DataDescriptor.from_data(X_client), model, config=config).run()
    matches = [tv for _, label, _, tv in result.front_points if label == "fastest_meeting_target"]
    assert matches and all(tv.precision_bits >= 15 for tv in matches)


def test_raw_data_is_rejected():
    X, y = load_diabetes_data()
    model = train_linear_regression_model(X, y)
    with pytest.raises(TypeError):
        Experiment(X, model)


def test_low_coefficient_headroom_is_flagged_not_rejected():
    # reported points keep their place; the flag only marks less spare room than the margin
    from types import SimpleNamespace
    from ckks_param_tool.config import load_experiment_config
    from ckks_param_tool.evaluation import ParameterEvaluationResult

    fake = SimpleNamespace(config=load_experiment_config({}))
    result = lambda h: ParameterEvaluationResult(error=1e-6, runtime=1.0, security_bits=128, status="success",
                                                 coefficient_headroom_bits=h)
    assert Experiment._low_headroom(fake, result(0.5))
    assert not Experiment._low_headroom(fake, result(3.0))
    assert not Experiment._low_headroom(fake, result(float("nan"))) # not measured
