# MLOPs Lab CIE - SkyDrop

GitHub: hr7raj
Email: 7hrraj@gmail.com

This project implements the `mlops-cie-192` lab tasks for the SkyDrop drone flight-time prediction scenario.

## Run order

```bash
python src/train.py
python src/predict_cli.py --payload_kg 2.4 --distance_km 7.9 --wind_speed_kmph 18.8 --altitude_m 55.3
python src/api.py --self-test
python src/retrain.py
```

## Docker

```bash
docker build -t skydrop-predictor:v1 .
docker run skydrop-predictor:v1 --payload_kg 2.4 --distance_km 7.9 --wind_speed_kmph 18.8 --altitude_m 55.3
```

## FastAPI

```bash
python src/api.py
```

Proof-of-execution files are written to `results/`.
