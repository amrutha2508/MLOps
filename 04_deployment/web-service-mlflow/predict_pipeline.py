import pickle
from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient

import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load variables from .env file
# In Jupyter, use Path.cwd() instead of __file__
env_path = Path.cwd() / '.env'

# If .env is in a parent directory, walk up to find it
if not env_path.exists():
    for parent in Path.cwd().parents:
        candidate = parent / '.env'
        if candidate.exists():
            env_path = candidate
            break
    else:
        raise FileNotFoundError(f"Could not find .env file. Searched from {Path.cwd()}")

print(f"Loading .env from: {env_path}")

# Load variables
load_dotenv(dotenv_path=env_path)

# 2. Map Tigris-specific variables to MLflow/S3 standards
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("AWS_ENDPOINT_URL_S3", "")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_REGION", "auto")

MLFLOW_TRACKING_URI = "http://0.0.0.0:5001"
RUN_ID = "e0c87176278948d5949f55f67b4a0dc7"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logged_model = f"runs:/{RUN_ID}/model"
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    features = {}
    features["PU_DO"] = '%s_%s' % (ride["PULocationID"], ride["DOLocationID"])
    features["trip_distance"] = ride["trip_distance"]
    return features

def predict(features):
    preds = model.predict(features)
    return float(preds[0])

app = Flask("duration-prediction")

@app.route("/predict",methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        "duration": pred,
        "model_version":RUN_ID
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)