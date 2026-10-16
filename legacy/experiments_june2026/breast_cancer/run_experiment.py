from ckks_param_tool.experiment import Experiment
from ckks_param_tool.data import client_server_split
from ckks_param_tool.data import load_breast_cancer_data
from ckks_param_tool.models import train_logistic_linear_regression_model
import os
from pathlib import Path

if __name__ == "__main__":

    X, y = load_breast_cancer_data()
    X_client, X_server, y_client, y_server = client_server_split(X, y)

    model = train_logistic_linear_regression_model(X_server, y_server)

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    experiment = Experiment(X_client, model, config_path)

    experiment.run(verbose=True)

    base_dir = Path(__file__).parent          
    results_dir = base_dir / "results"

    experiment.save(results_dir)

