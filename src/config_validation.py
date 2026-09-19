"""Validation for the pipeline's YAML configuration."""

from typing import Any, Dict

REQUIRED_KEYS = {
    "file_path": str,
    "target_column": str,
    "val_test_size": float,
    "validation_size": float,
    "random_state": int,
    "param_grids": dict,
    "cv": int,
    "scoring": str,
    "numeric_features": list,
    "nominal_features": list,
    "passthrough_features": list,
}


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates the pipeline configuration dictionary.

    Args:
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
    Raises:
        ValueError: If a required key is missing, has the wrong type, or an
            out-of-range value.
    """
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config is missing required key(s): {missing}")

    for key, expected_type in REQUIRED_KEYS.items():
        value = config[key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"Config key '{key}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )

    for key in ("val_test_size", "validation_size"):
        value = config[key]
        if not 0.0 < value < 1.0:
            raise ValueError(f"Config key '{key}' must be in (0, 1), got {value}.")

    if config["cv"] < 2:
        raise ValueError(f"Config key 'cv' must be >= 2, got {config['cv']}.")

    for key in ("numeric_features", "nominal_features", "passthrough_features"):
        if not all(isinstance(item, str) for item in config[key]):
            raise ValueError(f"Config key '{key}' must be a list of strings.")

    for model_name, grid in config["param_grids"].items():
        if not isinstance(grid, dict):
            raise ValueError(
                f"Config key 'param_grids.{model_name}' must be a dict of hyperparameter lists."
            )
