from fastapi import FastAPI
from serving.model_router import ModelRouter
from database.db import log_prediction, init_db

app = FastAPI()

router = ModelRouter()

init_db()

@app.get("/")
def home():

    return {"message": "AI Model Intelligence Engine Running"}


@app.post("/predict")
def predict(data: dict):

    text = data["text"]

    label, score, model = router.predict(text)

    log_prediction(text, label, score, model)

    return {
        "prediction": label,
        "confidence": score,
        "model_used": model
    }