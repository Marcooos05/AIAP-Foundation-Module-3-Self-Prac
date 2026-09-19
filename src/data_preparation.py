#import the libraries
import logging
from typing import Any, Dict

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataPreparation:
    """
    Class for data preparation and feature engineering.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for data preparation.
        preprocessor (ColumnTransformer): Preprocessor for data transformation of numerical, nominal, and passthrough features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for data preparation.
        """
        self.config = config

        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing preprocessing steps for student score data.

        Args:
            df (pd.DataFrame): Input DataFrame containing raw data.
        """

        logging.info("Starting data cleaning process.")
        df.dropna(subset=['final_test'], inplace=True)

        df.fillna({'CCA': 'None', 'attendance_rate': df['attendance_rate'].mean()}, inplace=True)

        df.replace({'CCA': {'ARTS': 'Arts', 'SPORTS': 'Sports', 'NONE': 'None', 'CLUBS': 'Clubs'}}, inplace=True)
        df.replace({'tuition': {'Y': 'Yes', 'N': 'No'}}, inplace=True)

        binary_map = {'Yes': 1, 'No': 0}
        df.replace({'direct_admission': binary_map, 'tuition': binary_map}, inplace=True)

        df.drop(columns=['index', 'student_id', 'gender', 'bag_color', 'mode_of_transport',
                          'age', 'n_male', 'n_female', 'sleep_time', 'wake_time'], inplace=True)
        logging.info("Data cleaning process completed.")
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor for data transformation of numerical, nominal, and passthrough features.
        Returns:
            ColumnTransformer: A preprocessor for data transformation.
        """
        numerical_features = self.config['numeric_features']
        nominal_features = self.config['nominal_features']
        passthrough_features = self.config['passthrough_features']

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        nominal_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', nominal_transformer, nominal_features),
                ('pass', 'passthrough', passthrough_features),
            ],
            remainder="passthrough",
            n_jobs=-1
        )

        return preprocessor
