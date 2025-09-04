from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved model and encoder
model = joblib.load("best_model.pkl")
ord_enc = joblib.load("encoder.pkl")

# Encoded column names
enc_col_names = [
    'Organisation_encoded', 'Detail_encoded',
    'Location_encoded', 'Rocket_Status_encoded',
    'Mission_Status_encoded'
]

def predict_price_from_raw(unseen_dict):
    """
    Predict price for a single rocket launch input.
    unseen_dict keys: Organisation, Detail, Location, Rocket_Status, Mission_Status
    """
    keys_needed = ['Organisation', 'Detail', 'Location', 'Rocket_Status', 'Mission_Status']
    # Ensure all keys exist
    for k in keys_needed:
        if k not in unseen_dict:
            raise ValueError(f"Missing key in input: {k}")

    # Convert input to DataFrame and cast all to string
    unseen_df = pd.DataFrame([{k: str(unseen_dict[k]) for k in keys_needed}])

    # Encode categorical features
    unseen_enc_np = ord_enc.transform(unseen_df[keys_needed])
    unseen_enc_df = pd.DataFrame(unseen_enc_np, columns=enc_col_names)

    # Predict price
    pred = model.predict(unseen_enc_df[enc_col_names])[0]

    # Confidence (if classifier supports predict_proba)
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(unseen_enc_df[enc_col_names])
            confidence = float(proba.max() * 100)
        except Exception:
            confidence = None

    return float(pred), confidence

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        prediction, confidence = predict_price_from_raw(data)
        response = {"predicted_price": round(prediction, 2)}
        if confidence is not None:
            response["confidence"] = round(confidence, 2)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
