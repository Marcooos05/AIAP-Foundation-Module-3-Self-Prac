import numpy as np
import pandas as pd
import pytest

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining


@pytest.fixture
def config():
    return {
        "target_column": "final_test",
        "val_test_size": 0.3,
        "validation_size": 0.5,
        "random_state": 42,
        "cv": 2,
        "scoring": "r2",
        "numeric_features": ["number_of_siblings", "attendance_rate", "hours_per_week"],
        "nominal_features": ["learning_style", "CCA"],
        "passthrough_features": ["direct_admission", "tuition"],
        "param_grids": {
            "ridge_tuned": {"regressor__alpha": [1.0]},
            "lasso_tuned": {"regressor__alpha": [1.0]},
            "random_forest_tuned": {"regressor__n_estimators": [10], "regressor__max_depth": [3]},
            "gradient_boosting_tuned": {
                "regressor__n_estimators": [10],
                "regressor__learning_rate": [0.1],
                "regressor__max_depth": [2],
            },
        },
    }


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "index": range(n),
        "student_id": [f"s{i}" for i in range(n)],
        "number_of_siblings": rng.integers(0, 4, n),
        "direct_admission": rng.choice(["Yes", "No"], n),
        "CCA": rng.choice(["Sports", "Arts", "Clubs", "None"], n),
        "learning_style": rng.choice(["Visual", "Auditory"], n),
        "tuition": rng.choice(["Yes", "No"], n),
        "final_test": rng.uniform(40, 100, n),
        "n_male": rng.integers(5, 15, n),
        "n_female": rng.integers(5, 15, n),
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.integers(14, 17, n),
        "hours_per_week": rng.integers(2, 20, n),
        "attendance_rate": rng.uniform(70, 100, n),
        "sleep_time": ["22:00"] * n,
        "wake_time": ["06:00"] * n,
        "mode_of_transport": rng.choice(["walk", "public transport"], n),
        "bag_color": rng.choice(["red", "blue"], n),
    })


def test_end_to_end_pipeline_runs(config, raw_df):
    data_preparation = DataPreparation(config)
    df_cleaned = data_preparation.clean_data(raw_df.copy())

    model_training = ModelTraining(config, data_preparation.create_preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(df_cleaned)

    baseline_models, baseline_metrics = model_training.train_and_evaluate_baseline_models(
        X_train, y_train, X_val, y_val
    )
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    assert set(all_models) == {
        "dummy_median", "linear_regression", "ridge", "lasso",
        "ridge_tuned", "lasso_tuned", "random_forest_tuned", "gradient_boosting_tuned",
    }
    for metrics in all_metrics.values():
        assert {"MAE", "MSE", "RMSE", "R2 Score"}.issubset(metrics)

    best_model_name = max(all_metrics, key=lambda name: all_metrics[name]["R2 Score"])
    best_model = all_models[best_model_name]

    refit_model = model_training.refit_on_train_val(best_model, X_train, X_val, y_train, y_val)
    final_metrics = model_training.evaluate_final_model(refit_model, X_test, y_test, best_model_name)
    assert {"MAE", "MSE", "RMSE", "R2 Score"}.issubset(final_metrics)

    feature_importance = model_training.explain_model(refit_model, best_model_name)
    assert isinstance(feature_importance, dict)
