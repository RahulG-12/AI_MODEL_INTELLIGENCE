import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from models.neural_network import SentimentModel
from utils.data_loader import load_data
import joblib

print("Training script started")

def train():

    print("Loading dataset...")

    X_train, X_test, y_train, y_test = load_data("data/dataset.csv")

    print("Vectorizing text...")

    vectorizer = CountVectorizer()

    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    input_size = X_train_vec.shape[1]
    hidden_size = 16

    print("Building neural network")

    model = SentimentModel(input_size, hidden_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.FloatTensor(X_train_vec)
    y_train_tensor = torch.FloatTensor(y_train).view(-1,1)

    epochs = 100

    print("Starting training loop")

    for epoch in range(epochs):

        outputs = model(X_train_tensor)

        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item()}")

    torch.save(model.state_dict(), "models/sentiment_model.pth")

    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Model saved successfully")


if __name__ == "__main__":
    train()