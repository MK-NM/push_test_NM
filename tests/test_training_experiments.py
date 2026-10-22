import joblib
import json
import numpy as np
import pandas as pd
import pytest

from model_training.train_experiments import (
    run_experiments,
    train_best_model,
    write_feature_schema,
    write_validation_artifacts,
)
from model_training.validate_artifacts import validate_artifacts


def test_run_experiments_tracks_parameters_and_metrics(capsys):
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 3.0, 5.0, 7.0])
    X_test = np.array([[4.0], [5.0]])
    y_test = np.array([9.0, 11.0])

    results = run_experiments(X_train, X_test, y_train, y_test)
    output = capsys.readouterr().out

    assert len(results) == 3
    assert results[0]["parameters"] == {
        "fit_intercept": True,
        "positive": False,
    }
    assert results[0]["validation_mean_squared_error"] == 0.0
    assert "parameters:" in output
    assert "coefficients:" in output
    assert "validation_r_squared:" in output


def test_train_best_model_saves_the_lowest_mse_model(tmp_path):
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = np.array([3.0, 5.0, 7.0])
    results = [
        {
            "validation_mean_squared_error": 2.0,
            "parameters": {"fit_intercept": True, "positive": False},
        },
        {
            "validation_mean_squared_error": 1.0,
            "parameters": {"fit_intercept": False, "positive": False},
        },
    ]
    artifact_path = tmp_path / "model" / "best_model.joblib"

    train_best_model(X, y, results, artifact_path)

    artifact = joblib.load(artifact_path)
    assert artifact["parameters"] == {
        "fit_intercept": False,
        "positive": False,
    }
    assert artifact["feature_columns"] == ["feature"]
    assert artifact["model"].n_features_in_ == 1


def test_feature_schema_records_raw_and_encoded_columns(tmp_path):
    raw_features = pd.DataFrame({"amount": [1.0], "kind": ["A"]})
    features = pd.get_dummies(raw_features, drop_first=True)
    artifact_path = tmp_path / "feature_schema.json"

    schema = write_feature_schema(raw_features, features, artifact_path)

    assert schema["source_columns"] == ["amount", "kind"]
    assert schema["feature_columns"] == list(features.columns)
    assert json.loads(artifact_path.read_text()) == schema


def test_validation_artifacts_report_gate_and_predictions(tmp_path):
    results = [
        {
            "validation_mean_squared_error": 0.25,
            "validation_r_squared": 0.9,
            "parameters": {"fit_intercept": True, "positive": False},
        }
    ]
    validation_targets = np.array([1.0, 3.0])
    validation_predictions = np.array([1.5, 2.5])
    baseline_predictions = np.array([2.0, 2.0])

    report = write_validation_artifacts(
        results,
        validation_targets,
        validation_predictions,
        baseline_predictions,
        np.array([2.0, 3.0]),
        np.array([2.5, 2.5]),
        tmp_path,
    )

    assert report["validation"]["passed"] is True
    assert json.loads((tmp_path / "validation_report.json").read_text()) == report
    predictions = pd.read_csv(tmp_path / "validation_predictions.csv")
    assert predictions["predicted"].tolist() == [1.5, 2.5]


def test_validate_artifacts_accepts_passing_report(tmp_path):
    (tmp_path / "feature_schema.json").write_text(
        json.dumps({"feature_columns": ["feature"]})
    )
    (tmp_path / "best_model.joblib").write_bytes(b"candidate")
    (tmp_path / "validation_report.json").write_text(
        json.dumps(
            {
                "validation": {
                    "mean_squared_error": 0.5,
                    "baseline_mean_squared_error": 1.0,
                    "passed": True,
                }
            }
        )
    )

    report = validate_artifacts(tmp_path)

    assert report["validation"]["passed"] is True


def test_validate_artifacts_rejects_failing_report(tmp_path):
    (tmp_path / "feature_schema.json").write_text(
        json.dumps({"feature_columns": ["feature"]})
    )
    (tmp_path / "best_model.joblib").write_bytes(b"candidate")
    (tmp_path / "validation_report.json").write_text(
        json.dumps(
            {
                "validation": {
                    "mean_squared_error": 1.5,
                    "baseline_mean_squared_error": 1.0,
                    "passed": False,
                }
            }
        )
    )

    with pytest.raises(SystemExit, match="Model validation failed"):
        validate_artifacts(tmp_path)
