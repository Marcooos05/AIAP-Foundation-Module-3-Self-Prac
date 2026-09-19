import logging
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

logger = logging.getLogger(__name__)

class ModelTraining:
    """
    Class for model training and evaluation.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for model training.
        preprocessor_factory (Callable[[], ColumnTransformer]): Builds a fresh,
            unfitted preprocessor. Called once per model pipeline so every
            pipeline owns its own independent preprocessor instance.
    """

    def __init__(self, config: Dict[str, Any], preprocessor_factory: Callable[[], ColumnTransformer]):
        """
        Initializes the ModelTraining class with the given configuration and preprocessor factory.

        Args:
            config (Dict[str, Any]): Configuration dictionary for model training.
            preprocessor_factory (Callable[[], ColumnTransformer]): Callable that
                returns a new, unfitted ColumnTransformer each time it's called.
        """
        self.config = config
        self.preprocessor_factory = preprocessor_factory

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Splits the input DataFrame into training, validation, and testing sets.

        Args:
            df (pd.DataFrame): Input DataFrame containing cleaned data.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: Training, validation, and testing sets for features and target variable.
        """
        logger.info("Starting data splitting process.")
        X = df.drop(columns=[self.config['target_column']])
        y = df[self.config['target_column']]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=self.config['val_test_size'], random_state=self.config['random_state'])
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=self.config['validation_size'], random_state=self.config['random_state'])

        logger.info("Data splitting process completed.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Trains and evaluates baseline models (a median dummy regressor,
        Linear Regression, Ridge Regression, Lasso Regression) on the training
        and validation sets.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: Trained baseline models and their evaluation metrics on the validation set.
        """

        logger.info("Starting baseline model training and evaluation.")
        models = {
            'dummy_median': DummyRegressor(strategy='median'),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            logger.info("Training %s.", model_name)
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor_factory()), ('regressor', model)])
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline

            logger.info("Evaluating %s on validation set.", model_name)

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
        Trains and evaluates tuned models (Ridge, Lasso, Random Forest, and
        Gradient Boosting regressors) on the training and validation sets,
        each with its own hyperparameter grid.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: Trained tuned models and their evaluation metrics on the validation set.
        """

        logger.info("Starting tuned model training and evaluation.")
        tuned_models = {}
        tuned_metrics = {}
        param_grids = self.config['param_grids']
        cv = self.config['cv']
        scoring = self.config['scoring']

        models = {
            'ridge_tuned': Ridge(),
            'lasso_tuned': Lasso(),
            'random_forest_tuned': RandomForestRegressor(random_state=self.config['random_state']),
            'gradient_boosting_tuned': GradientBoostingRegressor(random_state=self.config['random_state']),
        }

        for model_name, model in models.items():
            logger.info("Training %s with hyperparameter tuning.", model_name)

            if model_name not in param_grids:
                raise ValueError(f"Config 'param_grids' is missing an entry for model '{model_name}'.")

            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor_factory()), ('regressor', model)])
            param_grid = param_grids[model_name]

            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            tuned_models[model_name] = grid_search.best_estimator_

            tuned_metrics[model_name] = self._evaluate_model(tuned_models[model_name], X_val, y_val, model_name + " (Tuned)")
            tuned_metrics[model_name]['best_params'] = grid_search.best_params_

        logger.info("Tuned model training and evaluation completed.")
        return tuned_models, tuned_metrics

    def refit_on_train_val(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> Pipeline:
        """
        Refits a clone of the selected model on the combined training and
        validation data, so the final test evaluation reflects a model trained
        on as much data as possible (standard best practice once model/
        hyperparameter selection is finished).

        Args:
            model (Pipeline): The selected model pipeline (fitted on X_train only).
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            y_train (pd.Series): Training target variable.
            y_val (pd.Series): Validation target variable.
        Returns:
            Pipeline: A new pipeline, refit on the combined train+val data.
        """
        logger.info("Refitting selected model on combined train+val data.")
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        refit_model = clone(model)
        refit_model.fit(X_train_val, y_train_val)
        return refit_model

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
        logger.info("Evaluating final model on testing set.")
        return self._evaluate_model(model, X_test, y_test, model_name)

    def explain_model(self, model: Pipeline, model_name: str) -> Dict[str, float]:
        """
        Produces a simple feature-importance explanation for the final model:
        coefficients for linear models, impurity-based importances for tree
        models. Connects the EDA's feature analysis to what the trained model
        actually learned.

        Args:
            model (Pipeline): The final fitted model pipeline.
            model_name (str): Name of the model being explained.
        Returns:
            Dict[str, float]: Feature name -> importance/coefficient, sorted by
                descending absolute value. Empty if the model exposes neither
                `coef_` nor `feature_importances_`.
        """
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['regressor']
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(regressor, 'coef_'):
            values = np.ravel(regressor.coef_)
        elif hasattr(regressor, 'feature_importances_'):
            values = regressor.feature_importances_
        else:
            logger.info("Model %s exposes no coef_/feature_importances_; skipping explainability.", model_name)
            return {}

        importance = dict(zip(feature_names, (float(v) for v in values)))
        importance = dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))
        logger.info("Feature importance for %s: %s", model_name, importance)
        return importance

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
        logger.info("Evaluation metrics for %s: %s", model_name, metrics)
        if metrics['R2 Score'] < 0:
            logger.warning(
                "%s has a negative R2 Score (%.4f), i.e. it performs worse than "
                "predicting the mean.", model_name, metrics['R2 Score'],
            )
        return metrics
