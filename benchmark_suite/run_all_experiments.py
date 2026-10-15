from ckks_param_tool import client_server_split

from ckks_param_tool.data import (
    load_diabetes_data,
    load_breast_cancer_data,
    load_wine_data,
    load_iris_data,
    load_california_housing_data,
    load_concrete_data,
    load_ionosphere_data,
    load_heart_disease_data,
    load_mnist_data
)

from ckks_param_tool.models import (
    train_linear_regression_model,
    train_logistic_linear_regression_model,
    train_mnist_model
)

from benchmark_suite.run_full_vs_tool import run_full_vs_tool
from benchmark_suite.run_test_vectors_vs_random_entries import run_test_vectors_vs_random_entries


def run_dataset(
    name,
    loader,
    trainer
):
    X, y = loader()

    X_client, X_server, y_client, y_server = client_server_split(
        X,
        y
    )

    model = trainer(
        X_server,
        y_server
    )

    run_test_vectors_vs_random_entries(
        X_client,
        model,
        dataset_name=name
    )


if __name__ == "__main__":

    datasets = [

        (
            "diabetes",
            load_diabetes_data,
            train_linear_regression_model
        ),

        # (
        #     "breast_cancer",
        #     load_breast_cancer_data,
        #     train_logistic_linear_regression_model
        # ),

        # (
        #     "wine",
        #     load_wine_data,
        #     train_logistic_linear_regression_model
        # ),

        # (
        #     "iris",
        #     load_iris_data,
        #     train_logistic_linear_regression_model
        # ),

        # (
        #     "california_housing",
        #     load_california_housing_data,
        #     train_linear_regression_model
        # ),

        # (
        #     "concrete",
        #     load_concrete_data,
        #     train_linear_regression_model
        # ),

        # (
        #     "ionosphere",
        #     load_ionosphere_data,
        #     train_logistic_linear_regression_model
        # ),

        # (
        #     "heart_disease",
        #     load_heart_disease_data,
        #     train_logistic_linear_regression_model
        # ),

        # (
        #     "MNIST",
        #     load_mnist_data,
        #     train_mnist_model

        # )
    ]

    for name, loader, trainer in datasets:

        print("\n" + "=" * 60)
        print(f"Running {name}")
        print("=" * 60)

        run_dataset(
            name,
            loader,
            trainer
        )