from __future__ import annotations

import json

import joblib
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from common import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    ROOT_DIR,
    ensure_project_dirs,
    load_dataset,
    regression_metrics,
    split_features,
    write_json,
)


EXPERIMENT_NAME = "skydrop-flight-time-min"
DOMAIN_TAG = "drone_delivery"


def candidate_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }


def main() -> None:
    ensure_project_dirs()
    df = load_dataset(DATA_DIR / "training_data.csv")
    x_train, x_test, y_train, y_test = split_features(df)

    mlflow.set_tracking_uri((ROOT_DIR / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    trained_models = {}

    for name, model in candidate_models().items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("domain", DOMAIN_TAG)
            params = model.get_params()
            mlflow.log_params(params)

            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            metrics = regression_metrics(y_test, predictions)
            mlflow.log_metrics(metrics)

            model_path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="models")

            results.append({"name": name, **metrics})
            trained_models[name] = model

    best = min(results, key=lambda item: item["mae"])
    champion_name = best["name"]
    champion_model = trained_models[champion_name]
    joblib.dump(champion_model, MODELS_DIR / "champion_model.joblib")

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "domain": DOMAIN_TAG,
        "best_model": champion_name,
        "best_metric_name": "mae",
        "best_metric_value": best["mae"],
        "random_state": 42,
        "test_size": 0.2,
    }
    (MODELS_DIR / "champion_metadata.json").write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8")

    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": results,
        "best_model": champion_name,
        "best_metric_name": "mae",
        "best_metric_value": best["mae"],
    }
    write_json(RESULTS_DIR / "step1_s1.json", output)
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    main()
