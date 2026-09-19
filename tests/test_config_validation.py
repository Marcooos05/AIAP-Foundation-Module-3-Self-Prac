import pytest

from src.config_validation import validate_config


def _valid_config():
    return {
        "file_path": "./data/regression_bonus_practice_data.csv",
        "target_column": "final_test",
        "val_test_size": 0.2,
        "validation_size": 0.5,
        "random_state": 42,
        "param_grids": {"ridge_tuned": {"regressor__alpha": [1.0]}},
        "cv": 5,
        "scoring": "r2",
        "numeric_features": ["hours_per_week"],
        "nominal_features": ["CCA"],
        "passthrough_features": ["tuition"],
    }


def test_valid_config_passes():
    validate_config(_valid_config())


def test_missing_required_key_raises():
    config = _valid_config()
    del config["cv"]
    with pytest.raises(ValueError, match="missing required key"):
        validate_config(config)


def test_wrong_type_raises():
    config = _valid_config()
    config["random_state"] = "42"
    with pytest.raises(ValueError, match="must be of type int"):
        validate_config(config)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_split_size_out_of_range_raises(value):
    config = _valid_config()
    config["val_test_size"] = value
    with pytest.raises(ValueError, match="val_test_size"):
        validate_config(config)


def test_cv_below_minimum_raises():
    config = _valid_config()
    config["cv"] = 1
    with pytest.raises(ValueError, match="cv"):
        validate_config(config)


def test_feature_list_with_non_string_raises():
    config = _valid_config()
    config["numeric_features"] = ["hours_per_week", 123]
    with pytest.raises(ValueError, match="numeric_features"):
        validate_config(config)


def test_malformed_param_grid_raises():
    config = _valid_config()
    config["param_grids"] = {"ridge_tuned": ["not", "a", "dict"]}
    with pytest.raises(ValueError, match="param_grids"):
        validate_config(config)
