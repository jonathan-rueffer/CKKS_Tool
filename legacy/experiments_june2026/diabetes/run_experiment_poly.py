from ckks_param_tool.experiment import Experiment
from ckks_param_tool.data import client_server_split
from ckks_param_tool.data import load_diabetes_data
from ckks_param_tool.models import PolynomialModel
from ckks_param_tool.models import train_linear_regression_model
from ckks_param_tool.models import WrappedModel
import os
from pathlib import Path

import numpy as np

class ToyPolynomialModel:
    def __init__(self, weights, alpha=0.01, bias=0.0):
        self.w = weights
        

    def __call__(self, x):
        z = np.dot(x, self.w)
        return z + z**5 

def encrypted_poly_forward(x_enc, weights, alpha=0.00, bias=0.0):
    
    z = x_enc.dot(weights)     # ciphertext
    z2 = z * z * z * z * z     # ciphertext multiplication (depth +1)
    return z + z2

if __name__ == "__main__":
    X, y = load_diabetes_data()
    X_client, X_server, y_client, y_server = client_server_split(X, y)

    # ----- create toy polynomial weights -----
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 1, X.shape[1])

    # ----- plaintext model -----
    poly_model = ToyPolynomialModel(weights, alpha=0.00, bias=0.0)

    # ----- encrypted forward -----
    def enc_forward(x_enc):
        return encrypted_poly_forward(x_enc, weights, alpha=0.00, bias=0.0)

    # ----- wrapped model -----
    model = WrappedModel(
        model=poly_model,
        depth=4,
        encrypted_forward_fn=enc_forward
    )

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    experiment = Experiment(X_client, model)

    experiment.run(verbose=True)

    base_dir = Path(__file__).parent          
    results_dir = base_dir / "thesis_results_poly"

    experiment.save(results_dir)