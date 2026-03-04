import joblib
import torch
from transformers import pipeline

class AIOrchestrator:

    def __init__(self):

        self.lr = joblib.load("models/logistic.pkl")
        self.rf = joblib.load("models/random_forest.pkl")
        self.vectorizer = joblib.load("models/vectorizer.pkl")

        self.transformer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def run_all_models(self, text):

        X = self.vectorizer.transform([text])

        lr_score = self.lr.predict_proba(X)[0][1]
        rf_score = self.rf.predict_proba(X)[0][1]

        transformer_result = self.transformer(text)[0]

        transformer_score = transformer_result["score"]

        return {
            "logistic": lr_score,
            "random_forest": rf_score,
            "transformer": transformer_score
        }