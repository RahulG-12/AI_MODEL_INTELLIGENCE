import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from pipeline.data_loader import load_dataset
from pipeline.preprocess import vectorize_text

print("Loading dataset")

texts, labels = load_dataset("data/dataset.csv")

print("Vectorizing text")

X, vectorizer = vectorize_text(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2
)

print("Training Logistic Regression")

lr = LogisticRegression()
lr.fit(X_train, y_train)

print("Training Random Forest")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

joblib.dump(lr, "models/logistic.pkl")
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Models saved")