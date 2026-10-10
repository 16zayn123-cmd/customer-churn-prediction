from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

app = Flask(__name__)


def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run `python src/train.py` before starting the API."
        )
    return joblib.load(MODEL_PATH)


model = None


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    global model

    if model is None:
        model = get_model()

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Send customer data as JSON."}), 400

    try:
        frame = pd.DataFrame([payload])
        probability = float(model.predict_proba(frame)[0, 1])
        prediction = int(probability >= 0.5)

        return jsonify(
            {
                "churn_probability": round(probability, 4),
                "predicted_churn": prediction,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
