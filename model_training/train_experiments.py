import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


EXPERIMENTS = (
    {"fit_intercept": True, "positive": False},
    {"fit_intercept": False, "positive": False},
    {"fit_intercept": True, "positive": True},
)
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_PATH = ARTIFACT_DIR / "best_model.joblib"
FEATURE_SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
VALIDATION_REPORT_PATH = ARTIFACT_DIR / "validation_report.json"
VALIDATION_PREDICTIONS_PATH = ARTIFACT_DIR / "validation_predictions.csv"


def run_experiments(
    X_train: np.ndarray,
    X_validation: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    experiments: Sequence[Mapping[str, bool]] = EXPERIMENTS,
) -> list[dict[str, object]]:
    results = []

    for experiment_number, parameters in enumerate(experiments, start=1):
        model = LinearRegression(**parameters)
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        validation_mse = mean_squared_error(y_validation, predictions)
        validation_r_squared = r2_score(y_validation, predictions)
        metrics = {
            "model_name": f"Experiment {experiment_number}",
            "validation_mean_squared_error": float(validation_mse),
            "validation_r_squared": float(validation_r_squared),
        }
        result = {
            "experiment": experiment_number,
            "parameters": dict(parameters),
            "intercept": float(model.intercept_),
            "coefficients": model.coef_.tolist(),
            **metrics,
        }
        results.append(result)

        print(f"\n--- Experiment {experiment_number} ---")
        print(f"parameters: {json.dumps(result['parameters'], sort_keys=True)}")
        print(f"intercept: {result['intercept']:.6f}")
        print(f"coefficients: {json.dumps(result['coefficients'])}")
        print(f"validation_mean_squared_error: {validation_mse:.4f}")
        print(f"validation_r_squared: {validation_r_squared:.4f}")

    return results


def train_best_model(
    X: pd.DataFrame,
    y: np.ndarray,
    experiment_results: Sequence[Mapping[str, object]],
    artifact_path: Path = ARTIFACT_PATH,
) -> dict[str, object]:
    best_experiment = min(
        experiment_results,
        key=lambda result: float(result["validation_mean_squared_error"]),
    )
    parameters = best_experiment["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("Experiment parameters must be a mapping.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression(**parameters)
    model.fit(X_scaled, y)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "parameters": dict(parameters),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    print(
        "Selected parameters: "
        f"{json.dumps(artifact['parameters'], sort_keys=True)} "
        f"(validation MSE: {best_experiment['validation_mean_squared_error']:.4f})"
    )
    print(f"Saved model artifact to {artifact_path}")
    return artifact


def write_feature_schema(
    raw_features: pd.DataFrame,
    features: pd.DataFrame,
    artifact_path: Path = FEATURE_SCHEMA_PATH,
) -> dict[str, object]:
    categorical_columns = list(
        raw_features.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    )
    schema = {
        "target_column": "tip",
        "source_columns": list(raw_features.columns),
        "feature_columns": list(features.columns),
        "categorical_columns": categorical_columns,
        "category_values": {
            column: sorted(raw_features[column].dropna().astype(str).unique().tolist())
            for column in categorical_columns
        },
        "categorical_encoding": "pandas.get_dummies(drop_first=True)",
        "scaling": "StandardScaler fitted on training data",
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return schema


def write_validation_artifacts(
    experiment_results: Sequence[Mapping[str, object]],
    validation_targets: np.ndarray,
    validation_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    test_targets: np.ndarray,
    test_predictions: np.ndarray,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, object]:
    best_experiment = min(
        experiment_results,
        key=lambda result: float(result["validation_mean_squared_error"]),
    )
    validation_mse = float(best_experiment["validation_mean_squared_error"])
    baseline_mse = float(mean_squared_error(validation_targets, baseline_predictions))
    passed = bool(np.isfinite(validation_mse) and validation_mse < baseline_mse)
    report = {
        "validation": {
            "mean_squared_error": validation_mse,
            "r_squared": float(best_experiment["validation_r_squared"]),
            "baseline_mean_squared_error": baseline_mse,
            "passed": passed,
            "pass_criteria": "selected model validation MSE is lower than mean baseline MSE",
        },
        "test": {
            "mean_squared_error": float(
                mean_squared_error(test_targets, test_predictions)
            ),
            "r_squared": float(r2_score(test_targets, test_predictions)),
        },
        "selected_parameters": dict(best_experiment["parameters"]),
        "experiments": [dict(result) for result in experiment_results],
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / VALIDATION_REPORT_PATH.name
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "actual": validation_targets,
            "predicted": validation_predictions,
            "baseline_prediction": baseline_predictions,
        }
    ).to_csv(artifact_dir / VALIDATION_PREDICTIONS_PATH.name, index=False)

    print(f"Validation passed: {passed}")
    print(f"Saved validation report to {report_path}")
    return report


def main() -> None:
    np.random.seed(42)

    tips = sns.load_dataset("tips")
    df = pd.get_dummies(tips, drop_first=True)
    X = df.drop(columns=["tip"])
    y = df["tip"].values

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation, y_train_validation, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    metadata = {
        "train_shape": X_train_scaled.shape,
        "validation_shape": X_validation_scaled.shape,
        "test_shape": X_test.shape,
        "num_features": X_train_scaled.shape[1],
    }
    print(f"Pipeline Data Diagnostics: {metadata}")

    X_train_sm = sm.add_constant(X_train_scaled)
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    print(model_sm.summary())

    results = run_experiments(
        X_train_scaled, X_validation_scaled, y_train, y_validation
    )
    best_experiment = min(
        results, key=lambda result: float(result["validation_mean_squared_error"])
    )
    validation_model = LinearRegression(**best_experiment["parameters"])
    validation_model.fit(X_train_scaled, y_train)
    validation_predictions = validation_model.predict(X_validation_scaled)

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_validation)

    artifact = train_best_model(X_train_validation, y_train_validation, results)
    test_predictions = artifact["model"].predict(
        artifact["scaler"].transform(X_test)
    )

    write_feature_schema(tips.drop(columns=["tip"]), X)
    write_validation_artifacts(
        results,
        y_validation,
        validation_predictions,
        baseline_predictions,
        y_test,
        test_predictions,
    )

print(f"End of training experiments.")

if __name__ == "__main__":
    main()
