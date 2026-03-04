import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):

    df = pd.read_csv(path)

    texts = df["text"].values
    labels = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test