#import the libraries
import logging
from typing import Any, Dict

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

class DataPreparation:
    """
    Class for data preparation and feature engineering.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for data preparation.
    """

    REQUIRED_RAW_COLUMNS = ('final_test', 'CCA', 'attendance_rate', 'tuition', 'direct_admission')

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for data preparation.
        """
        self.config = config

    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes data quality statistics on the raw DataFrame before cleaning,
        so every run is auditable.

        Args:
            df (pd.DataFrame): Raw input DataFrame.
        Returns:
            Dict[str, Any]: Row/column counts, missing-value counts, duplicate
                count, and invalid-age count.
        """
        report = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "missing_values_per_column": {
                col: int(count) for col, count in df.isna().sum().items() if count > 0
            },
            "n_duplicate_rows": int(df.duplicated().sum()),
        }
        if "age" in df.columns:
            report["n_invalid_age"] = int(((df["age"] < 10) | (df["age"] > 20)).sum())

        logger.info("Data validation report: %s", report)
        return report

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing preprocessing steps for student score data.

        Args:
            df (pd.DataFrame): Input DataFrame containing raw data.
        Raises:
            ValueError: If the DataFrame is empty or is missing a required raw column.
        """
        if df.empty:
            raise ValueError("Cannot clean an empty DataFrame.")

        missing_columns = [col for col in self.REQUIRED_RAW_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Data is missing required raw column(s): {missing_columns}.")

        logger.info("Starting data cleaning process. Rows before cleaning: %d", len(df))

        n_duplicates = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        logger.info("Removed %d duplicate row(s).", n_duplicates)

        n_missing_target = df['final_test'].isna().sum()
        df.dropna(subset=['final_test'], inplace=True)
        logger.info("Dropped %d row(s) missing target 'final_test'.", n_missing_target)

        n_missing_cca = df['CCA'].isna().sum()
        attendance_mean = df['attendance_rate'].mean()
        n_missing_attendance = df['attendance_rate'].isna().sum()
        df.fillna({'CCA': 'None', 'attendance_rate': attendance_mean}, inplace=True)
        logger.info(
            "Filled %d missing 'CCA' value(s) with 'None' and %d missing "
            "'attendance_rate' value(s) with mean %.4f.",
            n_missing_cca, n_missing_attendance, attendance_mean,
        )
        if len(df) and n_missing_attendance / len(df) > 0.1:
            logger.warning(
                "More than 10%% of 'attendance_rate' values were missing and mean-imputed "
                "(%d/%d rows); this may bias the feature.",
                n_missing_attendance, len(df),
            )

        df.replace({'CCA': {'ARTS': 'Arts', 'SPORTS': 'Sports', 'NONE': 'None', 'CLUBS': 'Clubs'}}, inplace=True)
        df.replace({'tuition': {'Y': 'Yes', 'N': 'No'}}, inplace=True)

        binary_map = {'Yes': 1, 'No': 0}
        df.replace({'direct_admission': binary_map, 'tuition': binary_map}, inplace=True)

        # `age`, `gender`, `bag_color`, `mode_of_transport`, `n_male`/`n_female`,
        # `sleep_time`/`wake_time` are all dropped per the EDA (eda.ipynb, section 9
        # "Feature decisions summary"): each showed no meaningful relationship with
        # `final_test` (or, for age, still ~0 correlation even after correcting the
        # invalid entries), or was redundant/collinear with a stronger feature.
        df.drop(columns=['index', 'student_id', 'gender', 'bag_color', 'mode_of_transport',
                          'age', 'n_male', 'n_female', 'sleep_time', 'wake_time'], inplace=True)
        logger.info("Data cleaning process completed. Rows after cleaning: %d", len(df))
        return df

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Builds a new preprocessor for data transformation of numerical, nominal, and
        passthrough features. Called once per model pipeline so each pipeline owns
        an independent, unfitted preprocessor instance.
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
