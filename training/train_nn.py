import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pipeline.data_loader import load_dataset
from pipeline.preprocess import vectorize_text

texts, labels = load_dataset("data/dataset.csv")

X, vectorizer = vectorize_text(texts)

X = torch.tensor(X.toarray(), dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

class NeuralNet(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


model = NeuralNet(X.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):

    outputs = model(X_train).squeeze()

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch", epoch, "Loss", loss.item())

torch.save(model.state_dict(), "models/nn_model.pth")