import sqlite3

def init_db():

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        prediction TEXT,
        confidence REAL,
        model TEXT
    )
    """)

    conn.commit()
    conn.close()


def log_prediction(text, prediction, confidence, model):

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO predictions (text, prediction, confidence, model) VALUES (?, ?, ?, ?)",
        (text, prediction, confidence, model)
    )

    conn.commit()
    conn.close()