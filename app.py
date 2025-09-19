# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import logging
import pandas as pd
import numpy as np

# --- Configuration des logs ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

# --- Chargement du modèle et des transformateurs ---
model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    logging.info("Page d'accueil demandée")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "text" not in data:
        logging.warning("Requête /predict invalide : champ 'text' manquant")
        return jsonify({"error": "Le champ 'text' est obligatoire."}), 400

    text = [data["text"]]
    logging.info(f"Texte reçu pour prédiction : {text[0][:50]}...")  # seulement les 50 premiers caractères

    try:
        X = tfidf.transform(text)
        pred = model.predict(X)
        pred_proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

        result = {"predicted_label": label_encoder.inverse_transform(pred)[0]}
        if pred_proba is not None:
            result["probabilities"] = dict(zip(label_encoder.classes_, pred_proba.round(4)))

        logging.info(f"Résultat de la prédiction : {result}")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}", exc_info=True)
        return jsonify({"error": "Erreur serveur lors de la prédiction."}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    if "texts" not in data or not isinstance(data["texts"], list):
        logging.warning("Requête /predict_batch invalide : champ 'texts' manquant ou non liste")
        return jsonify({"error": "Le champ 'texts' est obligatoire et doit être une liste."}), 400

    texts = data["texts"]
    logging.info(f"Batch reçu pour prédiction : {len(texts)} textes")
    
    response = []
    for text in texts:
        try:
            X = tfidf.transform([text])
            pred = model.predict(X)
            item = {"text": text, "predicted_label": label_encoder.inverse_transform(pred)[0]}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                item["probabilities"] = dict(zip(label_encoder.classes_, proba.round(4)))
            response.append(item)
        except Exception as e:
            logging.error(f"Erreur pour le texte : {text[:50]}... | {e}", exc_info=True)
            response.append({"text": text, "error": str(e)})

    logging.info("Batch terminé")
    return jsonify(response)

if __name__ == "__main__":
    logging.info("Démarrage de l'application Flask sur 0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000)
