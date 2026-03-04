import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import requests
import plotly.express as px
from analytics.analytics_engine import get_prediction_stats

st.set_page_config(layout="wide")

st.title("AI Model Intelligence Engine")

text = st.text_input("Enter text")

if st.button("Analyze"):

    res = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": text}
    )

    result = res.json()

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction", result["prediction"])
    col2.metric("Confidence", round(result["confidence"], 2))
    col3.metric("Model Used", result["model_used"])

st.divider()

df = get_prediction_stats()

if len(df) > 0:

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            df,
            names="prediction",
            title="Prediction Distribution"
        )

        st.plotly_chart(fig)

    with col2:

        fig = px.bar(
            df,
            x="model",
            title="Model Usage"
        )

        st.plotly_chart(fig)

    st.dataframe(df)