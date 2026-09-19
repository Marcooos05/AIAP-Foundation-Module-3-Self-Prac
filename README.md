# Student Score Regression Pipeline

A configurable, end-to-end machine learning pipeline that predicts a student's
O-level mathematics examination score (`final_test`) from demographic,
academic, and lifestyle attributes. It loads raw student records, validates
and cleans them, trains a set of baseline and hyperparameter-tuned regression
models, and automatically selects and reports the best performer on a
held-out test set.

The pipeline is driven entirely by configuration (`src/config.yaml`), so
features, model hyperparameters, and data splits can be changed without
editing code.

---

## Prerequisites and Installation

**Prerequisites**

- Python 3.11 (see `.python-version`)
- The dependencies in `requirements.txt`: numpy, pandas, scikit-learn, scipy, PyYAML
- For development/testing: `requirements-dev.txt` additionally installs pytest, matplotlib, and seaborn (the latter two for `eda.ipynb` only)

**Installation**

```bash
cd AIAP_Mod3_Self-Prac
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Runtime only
pip install -r requirements.txt

# Runtime + dev/test tooling
pip install -r requirements-dev.txt
```

---

## Instructions for Executing the Pipeline

Run the full pipeline from the `AIAP_Mod3_Self-Prac/` directory:

```bash
python main.py
```

`main.py` loads and validates `src/config.yaml`, reads the dataset, checks
that every configured column is present, then runs validation reporting →
cleaning → splitting → baseline training → tuning → model selection → refit
→ final test evaluation → explainability. Progress, timings, and evaluation
metrics are logged to the console, ending with the selected best model and
its test-set performance.

**Modifying parameters** — edit `src/config.yaml` (no code changes needed):

| Key | Purpose |
| --- | --- |
| `file_path` | Path to the input CSV |
| `target_column` | Column to predict (`final_test`) |
| `val_test_size`, `validation_size` | Train / validation / test split ratios |
| `random_state` | Seed for reproducible splits |
| `param_grids`, `cv`, `scoring` | Per-model hyperparameter grids, CV folds, and scoring for tuning |
| `numeric_features`, `nominal_features`, `passthrough_features` | Feature roles in preprocessing |

**Outputs:**

| Path | Contents |
| --- | --- |
| `models/best_pipeline.joblib` | The final fitted model (preprocessor + regressor), refit on train+validation data |
| `reports/data_validation_report.json` | Pre-cleaning data quality stats (row/column counts, missing values, duplicates, invalid ages) |
| `reports/final_metrics.json` | Test-set MAE / MSE / RMSE / R² for the final model |
| `reports/selected_hyperparameters.json` | Chosen model's name and tuned hyperparameters |
| `reports/feature_importance.json` | Feature importances (tree models) or coefficients (linear models) |

---

## Running Tests

```bash
pytest
```

Covers the deterministic data-cleaning and config-validation helpers
(`tests/test_data_preparation.py`, `tests/test_config_validation.py`), plus a
miniature end-to-end run of the full pipeline on synthetic data
(`tests/test_pipeline_integration.py`).

---

## Pipeline Flow

```
config.yaml + CSV
       │
       ▼
┌───────────────────┐
│ Load & validate   │  yaml.safe_load, validate_config, column-presence check
│ config + data     │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Validation report │  DataPreparation.generate_validation_report
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Clean data        │  DataPreparation.clean_data
│                    │  (dedup, drop missing target, impute, normalise, drop columns)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Split data        │  train / validation / test
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Preprocess         │  independent ColumnTransformer per model Pipeline
│ (scale / encode)   │  (fit on train, applied per fold)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Train baselines   │  Dummy (median), Linear, Ridge, Lasso
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Tune models        │  GridSearchCV: Ridge, Lasso, Random Forest, Gradient Boosting
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Select best        │  highest validation R²
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Refit + evaluate   │  refit on train+val, evaluate on held-out test set
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Explain & persist  │  feature importance/coefficients, save model + reports
└───────────────────┘
```

**Step summary**

1. **Load & validate** (`main.py`) — read `config.yaml`, validate its schema, load the CSV, and check every configured feature/target column exists.
2. **Validation report** (`data_preparation.py`) — compute row/column counts, missing-value counts, duplicate count, and invalid-age count on the raw data, before any cleaning.
3. **Clean** (`data_preparation.py`) — drop duplicates; drop rows missing the target `final_test`; fill missing `CCA` with `'None'` and missing `attendance_rate` with the column mean; normalise inconsistent `CCA`/`tuition` labels; map `direct_admission`/`tuition` to binary 0/1; drop columns with no meaningful relationship to the target or that are redundant (`index`, `student_id`, `gender`, `bag_color`, `mode_of_transport`, `age`, `n_male`, `n_female`, `sleep_time`, `wake_time`).
4. **Split** (`model_training.py`) — partition into train, validation, and test sets via two `train_test_split` calls.
5. **Preprocess** — each model `Pipeline` gets its own `ColumnTransformer` instance (scales numeric features, one-hot encodes nominal features, passes through binary features), fit only on that pipeline's training data.
6. **Train baselines** — Dummy (median), Linear, Ridge, and Lasso regression, scored on the validation set.
7. **Tune** — `GridSearchCV` over each model's grid in `param_grids` (Ridge, Lasso, Random Forest, Gradient Boosting), scored on R² with `cv`-fold cross-validation.
8. **Select** — pick the model with the highest validation R².
9. **Refit & evaluate** — refit the selected model on combined train+validation data, then report MAE / MSE / RMSE / R² on the untouched test set.
10. **Explain & persist** — extract feature importances/coefficients, then save the model and all reports to disk.

---

## Key Findings from EDA

Detailed analysis lives in `eda.ipynb`; the highlights that shaped the pipeline:

- **15,900 student records**, with `final_test` (O-level math score) as the target.
- **No duplicate rows** in the raw data.
- **Missing values** — `CCA` (3,829), `final_test` (495), `attendance_rate` (778). Rows missing the target are dropped (a synthetic score would bias the model); missing `CCA` is treated as its own `'None'` category; missing `attendance_rate` is mean-imputed.
- **Invalid `age` values** — minimum observed age was -5, with 451 entries outside the plausible 14–16 range, most likely sign/typo errors. `age` was ultimately dropped rather than corrected, since it showed ~0 correlation with `final_test` even after cleaning.
- **Inconsistent category labels** — `CCA` contained mixed casing (`SPORTS`, `Sports`, `ARTS`, etc.) and `tuition` contained both `Yes/No` and `Y/N` → both normalised to a single consistent form.
- **Weak/redundant features dropped** — `gender`, `bag_color`, `mode_of_transport`, and `age` showed poor relationships with `final_test`; `n_male`/`n_female` were highly correlated with each other and weakly related to the target; `sleep_hours_daily` (engineered from `sleep_time`/`wake_time`) was dropped in favour of the more strongly correlated `attendance_rate` due to collinearity.
- **Strongest correlates of `final_test`** — `number_of_siblings` (negative), `attendance_rate`, `tuition`, and `direct_admission` (all positive).

---

## Feature Handling

Raw fields are cleaned in `clean_data`, then transformed by a
`ColumnTransformer` before modelling (roles configured in `config.yaml`).

| Feature | Type | Cleaning | Transform |
| --- | --- | --- | --- |
| `number_of_siblings` | Numeric | Used as-is | `StandardScaler` |
| `attendance_rate` | Numeric | Missing values filled with column mean | `StandardScaler` |
| `hours_per_week` | Numeric | Used as-is | `StandardScaler` |
| `learning_style` | Nominal | Used as-is | `OneHotEncoder` |
| `CCA` | Nominal | Missing → `'None'`; casing normalised (`SPORTS`→`Sports`, etc.) | `OneHotEncoder` |
| `direct_admission` | Passthrough | Mapped `Yes`/`No` → `1`/`0` | Passthrough |
| `tuition` | Passthrough | Normalised `Y`/`N` → `Yes`/`No`, then mapped → `1`/`0` | Passthrough |

**Dropped columns:** `index`, `student_id` (identifiers); `gender`,
`bag_color`, `mode_of_transport`, `age`, `n_male`, `n_female`, `sleep_time`,
`wake_time` (weak relationship with the target or redundant/collinear with a
stronger feature — see EDA findings above).

---

## Model Choices

| Model | Why included |
| --- | --- |
| **Dummy (median)** | Naive baseline — predicts the median training score regardless of input; anything worse than this signals a broken pipeline. |
| **Linear Regression** | Simplest real baseline; establishes a reference R² with no regularisation. |
| **Ridge (L2)** | Penalises large coefficients to curb overfitting given the one-hot encoded feature space. |
| **Lasso (L1)** | Adds sparsity, effectively performing feature selection by driving weak coefficients to zero. |
| **Random Forest** | Captures non-linear relationships and feature interactions the linear models can't. |
| **Gradient Boosting** | Sequentially corrects prior errors; typically the strongest performer among tree ensembles on tabular data of this size. |

Ridge, Lasso, Random Forest, and Gradient Boosting are additionally **tuned**
with `GridSearchCV` over the grids defined per-model in `config.yaml`
(`param_grids`), using `cv`-fold cross-validation scored on R².

---

## Model Evaluation

Each model is scored on the validation set; the best is refit on
train+validation and re-evaluated on the untouched test set. Metrics
computed in `_evaluate_model`:

| Metric | Meaning | Why it matters here |
| --- | --- | --- |
| **MAE** | Mean Absolute Error | Average score-point error, robust to outliers — intuitive for exam scores. |
| **MSE** | Mean Squared Error | Penalises large errors more heavily. |
| **RMSE** | Root Mean Squared Error | Same units as the score; interpretable error magnitude. |
| **R²** | Coefficient of determination | Share of score variance explained; used as the **selection criterion** and tuning score. |

**Selection:** `main.py` picks the model with the highest validation **R²**,
refits it on train+validation data, then reports its full metric set on the
test set so the headline performance reflects unseen data.

---

## Considerations for Deployment

- **Reproducible inference** — the fitted `Pipeline` bundles preprocessing and the regressor, so the same transformations applied in training run at inference. The chosen pipeline is persisted via `joblib.dump` (`models/best_pipeline.joblib`) rather than retraining on each call.
- **Schema & input validation** — `main.py` verifies configured columns exist in the input data before training; `OneHotEncoder(handle_unknown='ignore')` gives graceful handling of unseen `CCA`/`learning_style` categories at inference, but upstream validation of incoming records is still advisable.
- **Data and concept drift** — student behaviour and academic standards shift over cohorts; plan periodic retraining and monitor live error (MAE/RMSE) against the reported test-set baseline.
- **Independent preprocessors** — each model pipeline builds and fits its own `ColumnTransformer` instance rather than sharing one, avoiding any risk of state leaking between models during experimentation.
- **Model ceiling** — if accuracy is insufficient for production, additional feature engineering (e.g. revisiting the dropped `sleep_hours_daily`/`age` features) or ensembling is a natural next step and slots into the same pipeline structure.

---

## Project Structure

```
AIAP_Mod3_Self-Prac/
├── main.py                    # Pipeline entry point
├── eda.ipynb                  # Exploratory data analysis
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # + dev/test dependencies (pytest, matplotlib, seaborn)
├── pytest.ini                 # Test discovery config
├── .python-version            # Pinned Python version
├── data/
│   └── regression_bonus_practice_data.csv
├── models/
│   └── best_pipeline.joblib   # Final fitted model (generated by `python main.py`)
├── reports/                   # Validation report, metrics, hyperparameters, feature importance
├── tests/
│   ├── test_data_preparation.py
│   ├── test_config_validation.py
│   └── test_pipeline_integration.py
└── src/
    ├── config.yaml             # Features, model, and split configuration
    ├── config_validation.py    # Config schema/range validation
    ├── data_preparation.py     # Validation reporting, cleaning, preprocessor factory
    └── model_training.py       # Split, train, tune, evaluate, explain
```

## Future Improvements

If this project continues to grow, natural next steps include a
`Dockerfile` or `pyproject.toml` packaging for more robust environment
reproduction, and CI to run `pytest` automatically on each change.
