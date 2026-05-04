from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from common import MODELS_DIR, RESULTS_DIR, feature_frame, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict SkyDrop drone flight time in minutes.")
    parser.add_argument("--payload_kg", type=float, required=True)
    parser.add_argument("--distance_km", type=float, required=True)
    parser.add_argument("--wind_speed_kmph", type=float, required=True)
    parser.add_argument("--altitude_m", type=float, required=True)
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "champion_model.joblib")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def predict(payload: dict[str, float], model_path: Path = MODELS_DIR / "champion_model.joblib") -> float:
    model = joblib.load(model_path)
    prediction = model.predict(feature_frame(payload))[0]
    return round(float(prediction), 6)


def main() -> None:
    args = build_parser().parse_args()
    payload = {
        "payload_kg": args.payload_kg,
        "distance_km": args.distance_km,
        "wind_speed_kmph": args.wind_speed_kmph,
        "altitude_m": args.altitude_m,
    }
    result = {
        "image_name": "skydrop-predictor",
        "image_tag": "v1",
        "base_image": "python:3.12-slim",
        "test_input": payload,
        "prediction": predict(payload, args.model_path),
    }

    if args.output:
        write_json(args.output, result)
    elif payload == {"payload_kg": 2.4, "distance_km": 7.9, "wind_speed_kmph": 18.8, "altitude_m": 55.3}:
        write_json(RESULTS_DIR / "step2_s3.json", result)

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
