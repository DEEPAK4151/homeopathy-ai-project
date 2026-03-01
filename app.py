from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# ===============================
# Load trained model & vectorizer
# ===============================
disease_model = joblib.load("disease_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ===============================
# Load remedy dataset
# ===============================
data = pd.read_csv("homeopathy_remedy_dataset_1000_rows.csv")

# -------------------------------
# Remedy lookup function
# -------------------------------
def get_remedies_for_disease(disease_name, top_n=3):
    matches = data[data["disease"] == disease_name]

    if matches.empty:
        return []

    remedies = (
        matches[["remedy", "potency"]]
        .drop_duplicates()
        .head(top_n)
        .to_dict(orient="records")
    )

    return remedies

# -------------------------------
# Prediction function
# -------------------------------
def predict_disease_and_remedy(user_symptoms):
    symptoms_text = " ".join(user_symptoms)
    X = vectorizer.transform([symptoms_text])

    predicted_disease = disease_model.predict(X)[0]

    remedies = get_remedies_for_disease(predicted_disease)

    return {
        "predicted_disease": predicted_disease,
        "recommended_remedies": remedies
    }

# ===============================
# Home page route
# ===============================
@app.route("/")
def home():
    return render_template("index.html")

# ===============================
# Prediction API
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.json
    symptoms = data_json.get("symptoms", [])

    result = predict_disease_and_remedy(symptoms)
    return jsonify(result)

# ===============================
# Run server
# ===============================
if __name__ == "__main__":
    app.run(debug=True)