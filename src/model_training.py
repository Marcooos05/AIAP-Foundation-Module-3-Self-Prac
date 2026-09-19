import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

class ModelTraining:
    """
    Class for model training and evaluation.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for model training.
        preprocessor (ColumnTransformer): Preprocessor for data transformation of numerical, nominal, and passthrough features.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initializes the ModelTraining class with the given configuration and preprocessor.

        Args:
            config (Dict[str, Any]): Configuration dictionary for model training.
            preprocessor (ColumnTransformer): Preprocessor for data transformation of numerical, nominal, and passthrough features.
        """
        self.config = config
        self.preprocessor = preprocessor

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Splits the input DataFrame into training, validation, and testing sets.

        Args:
            df (pd.DataFrame): Input DataFrame containing cleaned data.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: Training, validation, and testing sets for features and target variable.
        """
        logging.info("Starting data splitting process.")
        X = df.drop(columns=[self.config['target_column']])
        y = df[self.config['target_column']]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=self.config['val_test_size'], random_state=self.config['random_state'])
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=self.config['validation_size'], random_state=self.config['random_state'])

        logging.info("Data splitting process completed.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Trains and evaluates baseline models (Linear Regression, Ridge Regression, Lasso Regression) on the training and validation sets.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: Trained baseline models and their evaluation metrics on the validation set.
        """

        logging.info("Starting baseline model training and evaluation.")
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso()
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name}.")
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', model)])
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline

            logging.info(f"Evaluating {model_name} on validation set.")

            metrics[model_name] = self._evaluate_model(pipeline, X_val, y_val, model_name)

        return pipelines, metrics

    def train_and_evaluate_tuned_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Trains and evaluates tuned models (Ridge Regression, Lasso Regression) on the training and validation sets.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: Trained tuned models and their evaluation metrics on the validation set.
        """

        logging.info("Starting tuned model training and evaluation.")
        tuned_models = {}
        tuned_metrics = {}
        param_grid = self.config['param_grid']
        cv = self.config['cv']
        scoring = self.config['scoring']

        models = {
            'ridge_tuned': Ridge(),
            'lasso_tuned': Lasso()
        }

        for model_name, model in models.items():
            logging.info(f"Training {model_name} with hyperparameter tuning.")

            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('regressor', model)])

            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            tuned_models[model_name] = grid_search.best_estimator_

            tuned_metrics[model_name] = self._evaluate_model(tuned_models[model_name], X_val, y_val, model_name + " (Tuned)")

        logging.info("Tuned model training and evaluation completed.")
        return tuned_models, tuned_metrics

    def evaluate_final_model(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluates the final selected model on the testing set.

        Args:
            model (Pipeline): The final selected model pipeline.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target variable.
            model_name (str): Name of the model being evaluated.
        Returns:
            Dict[str, float]: Evaluation metrics for the final model on the testing set.
        """
        logging.info("Evaluating final model on testing set.")
        return self._evaluate_model(model, X_test, y_test, model_name)

    def _evaluate_model(self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str) -> Dict[str, float]:
        """
        Evaluates a given model on the validation set and calculates evaluation metrics.

        Args:
            model (Pipeline): The model pipeline to be evaluated.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
            model_name (str): Name of the model being evaluated.
        Returns:
            Dict[str, float]: Evaluation metrics for the model on the validation set.
        """
        y_pred = model.predict(X_val)
        metrics = {
            'MAE': mean_absolute_error(y_val, y_pred),
            'MSE': mean_squared_error(y_val, y_pred),
            'RMSE': root_mean_squared_error(y_val, y_pred),
            'R2 Score': r2_score(y_val, y_pred)
        }
        logging.info(f"Evaluation metrics for {model_name}: {metrics}")
        return metrics
