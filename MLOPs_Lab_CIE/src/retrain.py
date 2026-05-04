from __future__ import annotations

import json

import joblib
import pandas as pd

from common import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    read_json,
    regression_metrics,
    split_features,
    write_json,
)
from train import candidate_models


MIN_IMPROVEMENT_THRESHOLD = 0.3


def main() -> None:
    original = pd.read_csv(DATA_DIR / "training_data.csv")
    new_data = pd.read_csv(DATA_DIR / "new_data.csv")
    combined = pd.concat([original, new_data], ignore_index=True)

    metadata = read_json(MODELS_DIR / "champion_metadata.json")
    champion_name = metadata["best_model"]
    champion_model = joblib.load(MODELS_DIR / "champion_model.joblib")

    x_train, x_test, y_train, y_test = split_features(combined)
    retrained_model = candidate_models()[champion_name]
    retrained_model.fit(x_train, y_train)

    champion_metrics = regression_metrics(y_test, champion_model.predict(x_test))
    retrained_metrics = regression_metrics(y_test, retrained_model.predict(x_test))
    improvement = round(champion_metrics["rmse"] - retrained_metrics["rmse"], 6)

    action = "promoted" if improvement >= MIN_IMPROVEMENT_THRESHOLD else "kept_champion"
    if action == "promoted":
        joblib.dump(retrained_model, MODELS_DIR / "champion_model.joblib")
        metadata.update(
            {
                "best_model": champion_name,
                "promotion_source": "retrain.py",
                "comparison_metric": "rmse",
                "retrained_rmse": retrained_metrics["rmse"],
            }
        )
        (MODELS_DIR / "champion_metadata.json").write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8")

    result = {
        "original_data_rows": int(len(original)),
        "new_data_rows": int(len(new_data)),
        "combined_data_rows": int(len(combined)),
        "champion_rmse": champion_metrics["rmse"],
        "retrained_rmse": retrained_metrics["rmse"],
        "improvement": improvement,
        "min_improvement_threshold": MIN_IMPROVEMENT_THRESHOLD,
        "action": action,
        "comparison_metric": "rmse",
    }
    write_json(RESULTS_DIR / "step4_s8.json", result)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
