import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.utils._testing import ignore_warnings

from src.config_validation import validate_config
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path('models')


def _save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


@ignore_warnings(category=Warning)
def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # Load and validate configuration
    with open('./src/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    validate_config(config)

    # Load data
    start = time.perf_counter()
    logger.info("Loading data from CSV file.")
    df = pd.read_csv(config['file_path'])
    logger.info("Data loading completed in %.2fs.", time.perf_counter() - start)

    # Data preparation
    data_preparation = DataPreparation(config)

    start = time.perf_counter()
    validation_report = data_preparation.generate_validation_report(df)
    _save_json(validation_report, MODELS_DIR / 'data_validation_report.json')
    logger.info("Data validation completed in %.2fs.", time.perf_counter() - start)

    start = time.perf_counter()
    df_cleaned = data_preparation.clean_data(df)
    logger.info("Data cleaning completed in %.2fs.", time.perf_counter() - start)

    # Model training and evaluation
    model_training = ModelTraining(config, data_preparation.preprocessor)

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

    best_model_name = max(all_metrics, key=lambda x: all_metrics[x]['R2 Score'])
    best_model = all_models[best_model_name]
    logger.info("Best model: %s with R2 Score: %.4f", best_model_name, all_metrics[best_model_name]['R2 Score'])

    start = time.perf_counter()
    refit_model = model_training.refit_on_train_val(best_model, X_train, X_val, y_train, y_val)
    final_metrics = model_training.evaluate_final_model(refit_model, X_test, y_test, best_model_name)
    logger.info("Final refit and evaluation completed in %.2fs.", time.perf_counter() - start)

    feature_importance = model_training.explain_model(refit_model, best_model_name)

    # Persist artifacts
    joblib.dump(refit_model, MODELS_DIR / 'best_pipeline.joblib')
    _save_json(final_metrics, MODELS_DIR / 'final_metrics.json')
    _save_json(
        {
            'model_name': best_model_name,
            'hyperparameters': all_metrics[best_model_name].get('best_params', {}),
        },
        MODELS_DIR / 'selected_hyperparameters.json',
    )
    _save_json(feature_importance, MODELS_DIR / 'feature_importance.json')
    logger.info("Artifacts saved to %s", MODELS_DIR.resolve())


if __name__ == "__main__":
    main()
