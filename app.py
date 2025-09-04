from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model + encoder
model = joblib.load("best_model.pkl")
ord_enc = joblib.load("encoder.pkl")

enc_col_names = [
    'Organisation_encoded','Detail_encoded',
    'Location_encoded','Rocket_Status_encoded',
    'Mission_Status_encoded'
]

@app.route("/")
def index():
    return render_template("index.html")  # 👈 for a simple web page

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    input_enc = ord_enc.transform(input_df)
    input_enc_df = pd.DataFrame(input_enc, columns=enc_col_names)
    prediction = model.predict(input_enc_df)[0]
    return jsonify({"predicted_price": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(debug=True)
