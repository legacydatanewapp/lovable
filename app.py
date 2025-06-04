from flask import Flask, request, jsonify
import pandas as pd
import cloudpickle

app = Flask(__name__)

# Load the model
with open("tumor_model.pkl", "rb") as file:
    model = cloudpickle.load(file)

@app.route("/", methods=["GET"])
def home():
    return "Tumor Predictor API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            df = pd.read_csv(request.files["file"])
            preds = model.predict_proba(df)[:, 1]  # assuming binary classifier
            return jsonify({"predictions": preds.tolist()})
        else:
            data = request.json  # expects a dict of input features
            df = pd.DataFrame([data])
            pred = model.predict_proba(df)[0][1]
            return jsonify({"prediction": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
