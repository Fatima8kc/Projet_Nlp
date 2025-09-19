# app.py
from flask import Flask, request, jsonify,render_template
import joblib
import pandas as pd
import numpy as np

# --- Chargement du modèle et des transformateurs ---
model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Prédit la classe d'un seul texte.
    JSON attendu : {"text": "votre texte ici"}
    """
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "Le champ 'text' est obligatoire."}), 400

    text = [data["text"]]
    X = tfidf.transform(text)
    
    pred = model.predict(X)
    pred_proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    
    result = {
        "predicted_label": label_encoder.inverse_transform(pred)[0]
    }
    
    if pred_proba is not None:
        result["probabilities"] = dict(zip(label_encoder.classes_, pred_proba.round(4)))
    
    return jsonify(result)

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Prédit les classes pour une liste de textes.
    JSON attendu : {"texts": ["texte1", "texte2", ...]}
    """
    data = request.get_json()
    if "texts" not in data or not isinstance(data["texts"], list):
        return jsonify({"error": "Le champ 'texts' est obligatoire et doit être une liste."}), 400

    texts = data["texts"]
    X = tfidf.transform(texts)
    
    preds = model.predict(X)
    preds_labels = label_encoder.inverse_transform(preds)
    
    response = []
    for i, text in enumerate(texts):
        item = {"text": text, "predicted_label": preds_labels[i]}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(tfidf.transform([text]))[0]
            item["probabilities"] = dict(zip(label_encoder.classes_, proba.round(4)))
        response.append(item)
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
