from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from common import MODELS_DIR, RESULTS_DIR, feature_frame, write_json


MODEL_PATH = MODELS_DIR / "champion_model.joblib"
SERVICE_NAME = "SkyDrop flight_time_min API"

app = FastAPI(title=SERVICE_NAME)


class FlightFeatures(BaseModel):
    payload_kg: float = Field(..., ge=0.1, le=5.0)
    distance_km: float = Field(..., ge=0.5, le=15.0)
    wind_speed_kmph: float = Field(..., ge=0.0, le=30.0)
    altitude_m: float = Field(..., ge=10.0, le=120.0)


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict(features: FlightFeatures) -> float:
    model = load_model()
    prediction = model.predict(feature_frame(features.model_dump()))[0]
    return round(float(prediction), 6)


@app.get("/health")
def health() -> dict[str, object]:
    return {"alive": True, "service": SERVICE_NAME}


@app.post("/infer")
def infer(features: FlightFeatures) -> dict[str, float]:
    return {"prediction": predict(features)}


def self_test(output_path: Path = RESULTS_DIR / "step3_s4.json") -> dict[str, object]:
    test_input = {
        "payload_kg": 2.4,
        "distance_km": 7.9,
        "wind_speed_kmph": 18.8,
        "altitude_m": 55.3,
    }
    response = {
        "health_endpoint": "/health",
        "predict_endpoint": "/infer",
        "port": 8000,
        "health_response": health(),
        "test_input": test_input,
        "prediction": predict(FlightFeatures(**test_input)),
    }
    write_json(output_path, response)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or self-test the SkyDrop FastAPI service.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=4))
    else:
        import uvicorn

        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
