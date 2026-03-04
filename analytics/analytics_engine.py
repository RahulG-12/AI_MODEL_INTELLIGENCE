import sqlite3
import pandas as pd

def get_prediction_stats():

    conn = sqlite3.connect("predictions.db")

    df = pd.read_sql_query("SELECT * FROM predictions", conn)

    conn.close()

    return df