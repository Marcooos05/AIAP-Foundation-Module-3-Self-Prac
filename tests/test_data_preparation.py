import numpy as np
import pandas as pd
import pytest

from src.data_preparation import DataPreparation


@pytest.fixture
def config():
    return {
        "numeric_features": ["number_of_siblings", "attendance_rate", "hours_per_week"],
        "nominal_features": ["learning_style", "CCA"],
        "passthrough_features": ["direct_admission", "tuition"],
    }


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "index": [0, 1, 2, 3, 4],
        "student_id": ["a", "b", "c", "d", "e"],
        "number_of_siblings": [1, 2, 0, 1, 3],
        "direct_admission": ["Yes", "No", "Yes", "No", "Yes"],
        "CCA": ["SPORTS", np.nan, "ARTS", "CLUBS", "NONE"],
        "learning_style": ["Visual", "Auditory", "Visual", "Visual", "Auditory"],
        "tuition": ["Y", "N", "Yes", "No", "Y"],
        "final_test": [70.0, 65.0, np.nan, 80.0, 90.0],
        "n_male": [10, 12, 11, 9, 10],
        "n_female": [10, 8, 9, 11, 10],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "age": [15, 16, 15, 15, -5],
        "hours_per_week": [10, 12, 8, 15, 20],
        "attendance_rate": [90.0, np.nan, 85.0, 95.0, 100.0],
        "sleep_time": ["22:00"] * 5,
        "wake_time": ["06:00"] * 5,
        "mode_of_transport": ["walk"] * 5,
        "bag_color": ["red"] * 5,
    })


class TestGenerateValidationReport:
    def test_reports_row_and_column_counts(self, config, raw_df):
        report = DataPreparation(config).generate_validation_report(raw_df)
        assert report["n_rows"] == 5
        assert report["n_columns"] == raw_df.shape[1]

    def test_reports_missing_values_per_column(self, config, raw_df):
        report = DataPreparation(config).generate_validation_report(raw_df)
        assert report["missing_values_per_column"] == {"CCA": 1, "final_test": 1, "attendance_rate": 1}

    def test_reports_invalid_age_count(self, config, raw_df):
        report = DataPreparation(config).generate_validation_report(raw_df)
        assert report["n_invalid_age"] == 1


class TestCleanData:
    def test_drops_rows_missing_target(self, config, raw_df):
        cleaned = DataPreparation(config).clean_data(raw_df.copy())
        assert cleaned["direct_admission"].notna().all()
        assert len(cleaned) == 4

    def test_fills_missing_cca_with_none_category(self, config, raw_df):
        cleaned = DataPreparation(config).clean_data(raw_df.copy())
        assert not cleaned["CCA"].isna().any()

    def test_fills_missing_attendance_with_mean(self, config, raw_df):
        cleaned = DataPreparation(config).clean_data(raw_df.copy())
        assert not cleaned["attendance_rate"].isna().any()

    def test_normalizes_cca_casing(self, config, raw_df):
        cleaned = DataPreparation(config).clean_data(raw_df.copy())
        assert set(cleaned["CCA"]).issubset({"Sports", "Arts", "Clubs", "None"})

    def test_maps_binary_columns_to_integers(self, config, raw_df):
        cleaned = DataPreparation(config).clean_data(raw_df.copy())
        assert set(cleaned["direct_admission"].unique()).issubset({0, 1})
        assert set(cleaned["tuition"].unique()).issubset({0, 1})

    def test_drops_excluded_columns(self, config, raw_df):
        cleaned = DataPreparation(config).clean_data(raw_df.copy())
        for col in ("index", "student_id", "gender", "bag_color", "mode_of_transport",
                    "age", "n_male", "n_female", "sleep_time", "wake_time"):
            assert col not in cleaned.columns

    def test_removes_duplicate_rows(self, config, raw_df):
        df_with_dupe = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
        cleaned = DataPreparation(config).clean_data(df_with_dupe)
        assert len(cleaned) == 4

    def test_empty_dataframe_raises(self, config, raw_df):
        with pytest.raises(ValueError, match="empty"):
            DataPreparation(config).clean_data(raw_df.iloc[0:0])

    def test_missing_required_column_raises(self, config, raw_df):
        with pytest.raises(ValueError, match="missing required raw column"):
            DataPreparation(config).clean_data(raw_df.drop(columns=["final_test"]))


class TestCreatePreprocessor:
    def test_returns_independent_instances(self, config):
        data_prep = DataPreparation(config)
        first = data_prep.create_preprocessor()
        second = data_prep.create_preprocessor()
        assert first is not second
