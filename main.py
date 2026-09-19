import logging

import pandas as pd
import yaml
from sklearn.utils._testing import ignore_warnings

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(level=logging.INFO)

@ignore_warnings(category=Warning)
def main():
    # Load configuration
    with open('./src/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load data
    logging.info("Loading data from CSV file.")
    df = pd.read_csv(config['file_path'])
    logging.info("Data loading completed.")

    # Data preparation
    data_preparation = DataPreparation(config)
    df_cleaned = data_preparation.clean_data(df)

    # Model training and evaluation
    model_training = ModelTraining(config, data_preparation.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(df_cleaned)
    baseline_models, baseline_metrics = (
        model_training.train_and_evaluate_baseline_models(
            X_train, y_train, X_val, y_val
        )
    )

    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    best_model_name = max(all_metrics, key=lambda x: all_metrics[x]['R2 Score'])
    best_model = all_models[best_model_name]
    logging.info(f"Best model: {best_model_name} with R2 Score: {all_metrics[best_model_name]['R2 Score']:.4f}")

    final_metrics = model_training.evaluate_final_model(best_model, X_test, y_test, best_model_name)

if __name__ == "__main__":
    main()
