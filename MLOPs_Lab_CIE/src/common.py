from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"

FEATURE_COLUMNS = ["payload_kg", "distance_km", "wind_speed_kmph", "altitude_m"]
TARGET_COLUMN = "flight_time_min"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_project_dirs() -> None:
    for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features(df: pd.DataFrame):
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_project_dirs()
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_frame(payload: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[payload[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
