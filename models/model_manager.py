import joblib

class ModelManager:

    def __init__(self):

        print("Loading trained models...")

        self.lr = joblib.load("models/logistic.pkl")
        self.rf = joblib.load("models/random_forest.pkl")
        self.vectorizer = joblib.load("models/vectorizer.pkl")

        print("Models loaded successfully")

    def predict(self, text):

        X = self.vectorizer.transform([text])

        lr_score = self.lr.predict_proba(X)[0][1]
        rf_score = self.rf.predict_proba(X)[0][1]

        if lr_score > rf_score:
            score = lr_score
            model_used = "Logistic Regression"
        else:
            score = rf_score
            model_used = "Random Forest"

        label = "Positive" if score > 0.5 else "Negative"

        return label, score, model_used