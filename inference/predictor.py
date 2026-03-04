import torch
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neural_network import SentimentModel


class Predictor:

    def __init__(self):

        print("Loading trained model...")

        self.vectorizer = joblib.load("models/vectorizer.pkl")

        input_size = len(self.vectorizer.get_feature_names_out())
        hidden_size = 16

        self.model = SentimentModel(input_size, hidden_size)

        self.model.load_state_dict(
            torch.load("models/sentiment_model.pth")
        )

        self.model.eval()

        print("Model loaded successfully")

    def predict(self, text):

        vector = self.vectorizer.transform([text]).toarray()

        tensor = torch.FloatTensor(vector)

        with torch.no_grad():

            output = self.model(tensor)

        prediction = output.item()

        if prediction >= 0.5:
            return "Positive"
        else:
            return "Negative"
