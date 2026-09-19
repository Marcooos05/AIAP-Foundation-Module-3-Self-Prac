import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings

from src.config_validation import validate_config
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path('./src/config.yaml')
MODELS_DIR = Path('models')
REPORTS_DIR = Path('reports')


def _save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Loads and validates the pipeline YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at '{config_path}'.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    validate_config(config)
    return config


def _load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Loads the raw dataset referenced by the config and checks required columns exist."""
    file_path = Path(config['file_path'])
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at '{file_path}'.")

    logger.info("Loading data from CSV file.")
    df = pd.read_csv(file_path)
    logger.info("Data loading completed.")

    required_columns = {
        config['target_column'],
        *config['numeric_features'],
        *config['nominal_features'],
        *config['passthrough_features'],
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Data file is missing configured column(s): {sorted(missing_columns)}.")

    return df


def _select_best_model(
    all_models: Dict[str, Pipeline],
    all_metrics: Dict[str, Dict[str, Any]],
) -> Tuple[str, Pipeline]:
    """Selects the model with the highest validation R2 Score."""
    best_model_name = max(all_metrics, key=lambda name: all_metrics[name]['R2 Score'])
    best_model = all_models[best_model_name]
    logger.info(
        "Best model: %s with R2 Score: %.4f",
        best_model_name, all_metrics[best_model_name]['R2 Score'],
    )
    return best_model_name, best_model


def _persist_artifacts(
    refit_model: Pipeline,
    final_metrics: Dict[str, float],
    best_model_name: str,
    best_model_metrics: Dict[str, Any],
    feature_importance: Dict[str, float],
) -> None:
    """Saves the final model and its accompanying reports to disk."""
    joblib.dump(refit_model, MODELS_DIR / 'best_pipeline.joblib')
    _save_json(final_metrics, REPORTS_DIR / 'final_metrics.json')
    _save_json(
        {
            'model_name': best_model_name,
            'hyperparameters': best_model_metrics.get('best_params', {}),
        },
        REPORTS_DIR / 'selected_hyperparameters.json',
    )
    _save_json(feature_importance, REPORTS_DIR / 'feature_importance.json')
    logger.info("Model saved to %s, reports saved to %s", MODELS_DIR.resolve(), REPORTS_DIR.resolve())


@ignore_warnings(category=Warning)
def main():
    """
    Runs the end-to-end training pipeline:

    1. Load and validate the YAML config.
    2. Load the raw dataset and verify configured columns are present.
    3. Generate a data validation report, then clean the data.
    4. Split into train/validation/test sets.
    5. Train and evaluate baseline models, then hyperparameter-tuned models.
    6. Select the best model by validation R2 Score.
    7. Refit the best model on train+validation data and evaluate on the test set.
    8. Compute a feature-importance explanation for the final model.
    9. Persist the final model and all reports to disk.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    config = _load_config(CONFIG_PATH)
    df = _load_data(config)

    # Data preparation
    data_preparation = DataPreparation(config)

    start = time.perf_counter()
    validation_report = data_preparation.generate_validation_report(df)
    _save_json(validation_report, REPORTS_DIR / 'data_validation_report.json')
    logger.info("Data validation completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    df_cleaned = data_preparation.clean_data(df)
    logger.info("Data cleaning completed in %.2fs.", time.perf_counter() - start)

    # Model training and evaluation
    model_training = ModelTraining(config, data_preparation.create_preprocessor)

    start = time.perf_counter()
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(df_cleaned)
    logger.info("Data splitting completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    baseline_models, baseline_metrics = (
        model_training.train_and_evaluate_baseline_models(
            X_train, y_train, X_val, y_val
        )
    )
    logger.info("Baseline model training completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )
    logger.info("Tuned model training completed in %.2fs.", time.perf_counter() - start)

    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    best_model_name, best_model = _select_best_model(all_models, all_metrics)

    start = time.perf_counter()
    refit_model = model_training.refit_on_train_val(best_model, X_train, X_val, y_train, y_val)
    final_metrics = model_training.evaluate_final_model(refit_model, X_test, y_test, best_model_name)
    logger.info("Final refit and evaluation completed in %.2fs.", time.perf_counter() - start)

    feature_importance = model_training.explain_model(refit_model, best_model_name)

    _persist_artifacts(
        refit_model, final_metrics, best_model_name, all_metrics[best_model_name], feature_importance,
    )


if __name__ == "__main__":
    main()
