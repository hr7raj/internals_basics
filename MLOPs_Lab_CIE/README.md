# MLOps Lab CIE - SkyDrop Flight Time Prediction

GitHub: `hr7raj`

Email: `7hrraj@gmail.com`

Question paper code: `mlops-cie-192`

Repository name: `Internals_Basics`

Project folder: `MLOPs_Lab_CIE`

## What This Project Does

This project is a complete MLOps lab solution for the SkyDrop drone delivery scenario.

SkyDrop operates a drone delivery network for medical supplies. The operations team wants to predict how long a drone flight will take so that it can schedule battery swaps, plan dispatch, and avoid delays.

The machine learning task is a regression problem:

Input features:

- `payload_kg`: weight carried by the drone
- `distance_km`: flight distance
- `wind_speed_kmph`: wind speed during the trip
- `altitude_m`: drone altitude

Target:

- `flight_time_min`: predicted flight time in minutes

The project implements four lab tasks:

1. Train and compare ML models with MLflow tracking.
2. Package a command-line predictor with Docker.
3. Serve the model with FastAPI.
4. Retrain using new data and decide whether to promote the retrained model.

All proof-of-execution outputs are saved as machine-readable JSON files in `results/`, as required by the lab statement.

## Final Project Structure

```text
Internals_Basics/
    MLOPs_Lab_CIE/
        data/
            training_data.csv
            new_data.csv
        src/
            common.py
            train.py
            predict_cli.py
            api.py
            retrain.py
        models/
            LinearRegression.joblib
            RandomForest.joblib
            champion_model.joblib
            champion_metadata.json
        results/
            step1_s1.json
            step2_s3.json
            step3_s4.json
            step4_s8.json
        logs/
        requirements.txt
        Dockerfile
        .gitignore
        README.md
```

## Important Lab Rules Followed

The lab statement gave a few strict rules. This project follows them:

- Repository name is `Internals_Basics`.
- Main project folder is `MLOPs_Lab_CIE`.
- All code, data, models, and outputs are inside `MLOPs_Lab_CIE/`.
- Train/test splitting uses `random_state=42`.
- Train/test splitting uses `test_size=0.2`.
- MLflow is used for experiment tracking.
- Result proof files are JSON, not screenshots.
- The provided CSV data is stored exactly in `data/`.
- The four required proof files exist in `results/`.

## Dataset Files

There are two datasets.

### `data/training_data.csv`

This is the original dataset used for Task 1 model training and comparison.

It has 25 rows.

Columns:

- `payload_kg`
- `distance_km`
- `wind_speed_kmph`
- `altitude_m`
- `flight_time_min`

### `data/new_data.csv`

This is the newer dataset used in Task 4.

It has 20 rows.

The new data has shifted distributions. For example, it contains larger distances and higher wind speeds than the original training data. This simulates real production drift, where new incoming data may behave differently from the old data.

## Main Python Files

### `src/common.py`

This file contains shared constants and helper functions used by the rest of the project.

It defines:

- Project paths such as `DATA_DIR`, `MODELS_DIR`, and `RESULTS_DIR`
- Feature columns
- Target column
- `RANDOM_STATE = 42`
- `TEST_SIZE = 0.2`
- Dataset loading helper
- Train/test split helper
- Regression metric helper
- JSON read/write helper
- Helper to convert one input request into a Pandas DataFrame

This avoids repeating the same setup code in every script.

### `src/train.py`

This script handles Task 1.

It does the following:

1. Loads `data/training_data.csv`.
2. Splits the data using `test_size=0.2` and `random_state=42`.
3. Trains two models:
   - `LinearRegression`
   - `RandomForestRegressor`
4. Logs both models to MLflow.
5. Logs hyperparameters as MLflow params.
6. Logs `MAE` and `RMSE` as MLflow metrics.
7. Adds the MLflow tag `domain = drone_delivery`.
8. Uses experiment name `skydrop-flight-time-min`.
9. Selects the best model using lowest MAE.
10. Saves the best model as `models/champion_model.joblib`.
11. Saves model metadata as `models/champion_metadata.json`.
12. Writes Task 1 proof to `results/step1_s1.json`.

The best model selected in this run is:

```text
LinearRegression
```

### `src/predict_cli.py`

This script handles Task 2.

It is a command-line prediction tool. It loads the saved champion model and accepts feature values through command-line arguments.

Example:

```bash
python src/predict_cli.py --payload_kg 2.4 --distance_km 7.9 --wind_speed_kmph 18.8 --altitude_m 55.3
```

It returns JSON containing:

- Docker image name
- Docker image tag
- Base image
- Test input
- Prediction

It writes the Task 2 proof file:

```text
results/step2_s3.json
```

### `src/api.py`

This script handles Task 3.

It creates a FastAPI application that serves the champion model.

Endpoints:

```text
GET /health
POST /infer
```

`GET /health` returns:

```json
{
    "alive": true,
    "service": "SkyDrop flight_time_min API"
}
```

`POST /infer` accepts JSON input:

```json
{
    "payload_kg": 2.4,
    "distance_km": 7.9,
    "wind_speed_kmph": 18.8,
    "altitude_m": 55.3
}
```

It returns:

```json
{
    "prediction": 25.012651
}
```

The API uses Pydantic validation with correct feature ranges:

- `payload_kg`: 0.1 to 5.0
- `distance_km`: 0.5 to 15.0
- `wind_speed_kmph`: 0.0 to 30.0
- `altitude_m`: 10.0 to 120.0

Invalid inputs automatically return HTTP 422 because FastAPI and Pydantic handle validation errors.

The script also has a self-test mode:

```bash
python src/api.py --self-test
```

This writes:

```text
results/step3_s4.json
```

### `src/retrain.py`

This script handles Task 4.

It simulates a retraining pipeline.

It does the following:

1. Loads `data/training_data.csv`.
2. Loads `data/new_data.csv`.
3. Combines both datasets.
4. Reads the champion model type from `models/champion_metadata.json`.
5. Retrains the same model type that won Task 1.
6. Compares the existing champion model and the retrained model on the same test set.
7. Uses RMSE for comparison.
8. Promotes the retrained model only if RMSE improves by at least `0.3`.
9. Writes the decision to `results/step4_s8.json`.

In this run:

```text
Champion RMSE: 2.705525
Retrained RMSE: 3.016456
Improvement: -0.310931
Action: kept_champion
```

Because the retrained model did not improve RMSE by at least `0.3`, the original champion model was kept.

## Task 1: Experiment Tracking and Model Comparison

Task 1 required training `LinearRegression` and `RandomForest` models on the training data.

The script used:

```bash
python src/train.py
```

Models trained:

- `LinearRegression`
- `RandomForestRegressor`

Metrics calculated:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error

MLflow settings:

- Experiment name: `skydrop-flight-time-min`
- Tag: `domain = drone_delivery`
- Tracking store: local file store under `mlruns/`

Output file:

```text
results/step1_s1.json
```

Actual output:

```json
{
    "experiment_name": "skydrop-flight-time-min",
    "models": [
        {
            "name": "LinearRegression",
            "mae": 3.543241,
            "rmse": 3.960061
        },
        {
            "name": "RandomForest",
            "mae": 4.27204,
            "rmse": 4.603076
        }
    ],
    "best_model": "LinearRegression",
    "best_metric_name": "mae",
    "best_metric_value": 3.543241
}
```

Interpretation:

`LinearRegression` performed better because it had the lower MAE. Therefore, it became the champion model.

## Task 2: Docker Packaging

Task 2 required creating a CLI predictor and packaging it using Docker.

The CLI predictor is:

```text
src/predict_cli.py
```

The Dockerfile uses:

```text
python:3.12-slim
```

Docker image name:

```text
skydrop-predictor:v1
```

Build command:

```bash
docker build -t skydrop-predictor:v1 .
```

Run command:

```bash
docker run --rm skydrop-predictor:v1 --payload_kg 2.4 --distance_km 7.9 --wind_speed_kmph 18.8 --altitude_m 55.3
```

Output file:

```text
results/step2_s3.json
```

Actual output:

```json
{
    "image_name": "skydrop-predictor",
    "image_tag": "v1",
    "base_image": "python:3.12-slim",
    "test_input": {
        "payload_kg": 2.4,
        "distance_km": 7.9,
        "wind_speed_kmph": 18.8,
        "altitude_m": 55.3
    },
    "prediction": 25.012651
}
```

Interpretation:

The model predicts that the drone flight will take about `25.01` minutes for the given input.

## Task 3: FastAPI Serving

Task 3 required serving the best model through FastAPI.

Run the API:

```bash
python src/api.py
```

The service runs on:

```text
http://localhost:8000
```

Health endpoint:

```bash
curl http://localhost:8000/health
```

Inference endpoint:

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"payload_kg": 2.4, "distance_km": 7.9, "wind_speed_kmph": 18.8, "altitude_m": 55.3}'
```

Self-test command used for proof file:

```bash
python src/api.py --self-test
```

Output file:

```text
results/step3_s4.json
```

Actual output:

```json
{
    "health_endpoint": "/health",
    "predict_endpoint": "/infer",
    "port": 8000,
    "health_response": {
        "alive": true,
        "service": "SkyDrop flight_time_min API"
    },
    "test_input": {
        "payload_kg": 2.4,
        "distance_km": 7.9,
        "wind_speed_kmph": 18.8,
        "altitude_m": 55.3
    },
    "prediction": 25.012651
}
```

Interpretation:

The API is alive, exposes the required endpoints, validates inputs, and returns the same prediction as the CLI predictor.

## Task 4: Retraining Pipeline

Task 4 required combining old and new data, retraining the winning model type, and deciding whether the retrained model should replace the champion model.

Run command:

```bash
python src/retrain.py
```

Promotion rule:

```text
Promote only if champion_rmse - retrained_rmse >= 0.3
```

Output file:

```text
results/step4_s8.json
```

Actual output:

```json
{
    "original_data_rows": 25,
    "new_data_rows": 20,
    "combined_data_rows": 45,
    "champion_rmse": 2.705525,
    "retrained_rmse": 3.016456,
    "improvement": -0.310931,
    "min_improvement_threshold": 0.3,
    "action": "kept_champion",
    "comparison_metric": "rmse"
}
```

Interpretation:

The retrained model performed worse on the comparison test set. Its RMSE was higher than the champion model's RMSE. Since the improvement was negative and did not meet the `0.3` threshold, the system kept the existing champion model.

## How to Run Everything From Scratch

From the repository root:

```bash
cd MLOPs_Lab_CIE
```

Create or activate a Python environment.

If using the existing local virtual environment:

```bash
source venv/bin/activate
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Run all tasks in order:

```bash
python src/train.py
python src/predict_cli.py --payload_kg 2.4 --distance_km 7.9 --wind_speed_kmph 18.8 --altitude_m 55.3 --output results/step2_s3.json
python src/api.py --self-test
python src/retrain.py
```

After running these commands, the required proof files should exist:

```text
results/step1_s1.json
results/step2_s3.json
results/step3_s4.json
results/step4_s8.json
```

## How to Run With Docker

Build the image:

```bash
docker build -t skydrop-predictor:v1 .
```

Run the predictor:

```bash
docker run --rm skydrop-predictor:v1 --payload_kg 2.4 --distance_km 7.9 --wind_speed_kmph 18.8 --altitude_m 55.3
```

Expected prediction:

```text
25.012651
```

## How to View MLflow Runs

Task 1 logs runs locally under `mlruns/`.

To view them:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open:

```text
http://localhost:5000
```

You should see the experiment:

```text
skydrop-flight-time-min
```

Inside it, there should be runs for:

- `LinearRegression`
- `RandomForest`

Each run contains:

- Model parameters
- MAE metric
- RMSE metric
- `domain = drone_delivery` tag
- Saved model artifact

## Model Artifacts

The `models/` folder contains saved model files.

```text
models/LinearRegression.joblib
models/RandomForest.joblib
models/champion_model.joblib
models/champion_metadata.json
```

`champion_model.joblib` is the model used by:

- CLI predictor
- Docker container
- FastAPI service
- Retraining comparison

`champion_metadata.json` records which model won Task 1 and how it was selected.

## Why Linear Regression Won

The dataset is small and mostly follows a linear pattern between the input features and the target flight time.

`LinearRegression` can perform well on small structured datasets when the relationship is approximately linear.

`RandomForestRegressor` is more flexible, but with only 25 original rows it can overfit or generalize less reliably on the test split.

The actual Task 1 test metrics were:

```text
LinearRegression MAE: 3.543241
RandomForest MAE: 4.27204
```

Since lower MAE is better, `LinearRegression` became the champion model.

## Why the Retrained Model Was Not Promoted

The retraining pipeline combined the old and new data, giving 45 rows total.

However, the new data has shifted values, especially larger distances and higher wind speeds. The retrained model did not improve RMSE on the comparison test set.

Promotion required:

```text
RMSE improvement >= 0.3
```

Actual improvement:

```text
-0.310931
```

This means the retrained model was worse, so the correct MLOps decision was:

```text
kept_champion
```

This is important because MLOps is not just about retraining. It is about retraining, evaluating, and promoting only when the new model is actually better.

## Quick Viva Explanation

If asked what this project does, say:

```text
This project predicts drone flight time for SkyDrop using payload, distance, wind speed, and altitude. I trained Linear Regression and Random Forest models, tracked both experiments in MLflow, selected the best model by MAE, packaged the predictor with Docker, served the champion model through FastAPI, and implemented a retraining pipeline that promotes a new model only if RMSE improves by at least 0.3. The proof for each task is stored as JSON in the results folder.
```

If asked why JSON files matter, say:

```text
The lab requires machine-readable proof of execution instead of screenshots. Each task writes its output to a JSON file under results/.
```

If asked what the champion model is, say:

```text
LinearRegression won because it had the lowest MAE on the test split.
```

If asked what happened during retraining, say:

```text
The retrained model did not improve RMSE by the required 0.3 threshold, so the pipeline kept the existing champion model.
```

## Verification Commands Used

Python syntax check:

```bash
python -m compileall src
```

CLI predictor:

```bash
python src/predict_cli.py --payload_kg 2.4 --distance_km 7.9 --wind_speed_kmph 18.8 --altitude_m 55.3
```

API proof generation:

```bash
python src/api.py --self-test
```

Retraining:

```bash
python src/retrain.py
```

Docker predictor:

```bash
docker run --rm skydrop-predictor:v1 --payload_kg 2.4 --distance_km 7.9 --wind_speed_kmph 18.8 --altitude_m 55.3
```

## Final Status

The project is complete.

The required result files are present.

The champion model is saved.

The Docker image was built and verified.

The code has been committed and pushed to:

```text
https://github.com/hr7raj/Internals_Basics
```
