import shap
import joblib

def explain(text, vectorizer):

    model = joblib.load("models/random_forest.pkl")

    explainer = shap.TreeExplainer(model)

    X = vectorizer.transform([text])

    shap_values = explainer.shap_values(X)

    return shap_values