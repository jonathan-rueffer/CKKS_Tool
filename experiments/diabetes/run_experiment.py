from ckks_param_tool import Experiment
from ckks_param_tool import client_server_split
from ckks_param_tool import load_diabetes_data
from ckks_param_tool import train_linear_regression_model

import os
from pathlib import Path

if __name__ == "__main__":

    X, y = load_diabetes_data()
    X_client, X_server, y_client, y_server = client_server_split(X, y)

    model = train_linear_regression_model(X_server, y_server)
    
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    experiment = Experiment(X_client, model)

    experiment.run(verbose=True)

    base_dir = Path(__file__).parent          
    results_dir = base_dir / "results"

    experiment.save(results_dir)

