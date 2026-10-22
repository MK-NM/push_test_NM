import json
import math
from pathlib import Path


ARTIFACT_DIR = Path("artifacts")


def validate_artifacts(artifact_dir: Path = ARTIFACT_DIR) -> dict[str, object]:
    report_path = artifact_dir / "validation_report.json"
    schema_path = artifact_dir / "feature_schema.json"
    model_path = artifact_dir / "best_model.joblib"

    for path in (report_path, schema_path, model_path):
        if not path.is_file():
            raise SystemExit(f"Required artifact is missing: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validation = report.get("validation", {})
    validation_mse = validation.get("mean_squared_error")
    baseline_mse = validation.get("baseline_mean_squared_error")

    if not schema.get("feature_columns"):
        raise SystemExit("Feature schema has no feature columns.")
    if (
        validation.get("passed") is not True
        or not isinstance(validation_mse, (int, float))
        or not isinstance(baseline_mse, (int, float))
        or not math.isfinite(validation_mse)
        or not math.isfinite(baseline_mse)
        or validation_mse >= baseline_mse
    ):
        raise SystemExit("Model validation failed; the model will not be published.")

    print(
        "Model validation passed: "
        f"MSE {validation_mse:.4f} < baseline MSE {baseline_mse:.4f}"
    )
    return report


if __name__ == "__main__":
    validate_artifacts()