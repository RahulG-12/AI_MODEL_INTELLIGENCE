# AI Model Intelligence Platform

A **production-style machine learning system** that trains, evaluates, compares, and deploys multiple AI models through a scalable architecture.

The platform demonstrates a **complete end-to-end AI pipeline**, starting from raw data ingestion to model training, evaluation, explainability, and real-time serving through APIs and dashboards.

This project replicates **real-world ML engineering systems used in production AI environments**.

---

# Project Overview

In real AI systems, building a single model is not enough. Production ML platforms require:

- Model experimentation
- Evaluation and comparison
- Explainability
- Monitoring
- Real-time inference

This platform implements a **complete ML system architecture** that supports these capabilities.

The pipeline includes:

1. Data ingestion
2. Preprocessing and feature engineering
3. Multi-model training
4. Model evaluation and comparison
5. Explainability using SHAP
6. Model routing
7. API-based inference
8. Monitoring via dashboard

---

# Key Features

• Multi-model training pipeline (Machine Learning + Deep Learning + Transformer models)

• Automated model evaluation and comparison

• Experiment tracking using **MLflow**

• Explainable AI using **SHAP**

• Intelligent **model routing system** that selects the best model

• **FastAPI inference server** for real-time predictions

• Interactive **Streamlit dashboard**

• Modular **production-style architecture**

---

# System Architecture

```
Raw Data
     │
     ▼
Data Loading
     │
     ▼
Preprocessing & Feature Engineering
     │
     ▼
Model Training
(ML / Neural Network / Transformer)
     │
     ▼
Evaluation & Metrics
     │
     ▼
Model Comparison
     │
     ▼
Explainability (SHAP)
     │
     ▼
Model Router
     │
     ▼
FastAPI Inference Server
     │
     ▼
Streamlit Dashboard
```

---

# Project Structure

```
AI_MODEL_INTELLIGENCE
│
├── data
│   └── dataset.csv
│
├── pipeline
│   ├── data_loader.py
│   └── preprocess.py
│
├── training
│   ├── train_models.py
│   ├── train_nn.py
│   └── train_transformer.py
│
├── evaluation
│   ├── metrics.py
│   └── model_comparison.py
│
├── tracking
│   └── mlflow_tracker.py
│
├── explainability
│   └── shap_analysis.py
│
├── serving
│   └── model_router.py
│
├── api
│   └── server.py
│
├── dashboard
│   └── app.py
│
└── requirements.txt
```

---

# Tech Stack

## Programming
Python

## Machine Learning
Scikit-learn  
PyTorch  
Transformers  

## Experiment Tracking
MLflow

## Explainable AI
SHAP

## Backend API
FastAPI

## Dashboard
Streamlit

## Data Processing
Pandas  
NumPy

---

# Machine Learning Pipeline

## 1. Data Processing

The pipeline loads the dataset and performs preprocessing steps:

• Text cleaning  
• Tokenization  
• Feature extraction  
• Vectorization  

This prepares the data for model training.

---

# Model Training

The platform trains **three different model types**.

### Classical Machine Learning Models

Examples include:

• Logistic Regression  
• Random Forest  
• Support Vector Machine  

These models provide baseline performance.

---

### Neural Network Model

A custom **PyTorch deep learning architecture** is implemented.

Example architecture:

```
Input Layer
     ↓
Hidden Layer (ReLU activation)
     ↓
Output Layer (Sigmoid)
```

---

### Transformer Model

A transformer-based architecture is used for advanced NLP tasks.

This allows the system to perform **complex language understanding tasks**.

---

# Model Evaluation

Each model is evaluated using multiple metrics:

• Accuracy  
• Precision  
• Recall  
• F1 Score  

The evaluation module automatically identifies the **best-performing model**.

---

# Explainable AI

The system uses **SHAP (SHapley Additive Explanations)** to understand model predictions.

SHAP helps analyze:

• Feature importance  
• Model behavior  
• Prediction reasoning  

This improves **model transparency and trust**.

---

# Model Serving

The best-performing model is deployed using **FastAPI**.

Example request:

```
POST /predict

{
"text": "This product is amazing"
}
```

Example response:

```
{
"prediction": "Positive"
}
```

---

# Dashboard

A **Streamlit dashboard** provides a visual interface to:

• Test predictions  
• View model performance  
• Compare models  
• Monitor experiments  

---

# How To Run The Project

## Install Dependencies

```
pip install -r requirements.txt
```

---

## Train Models

```
python training/train_models.py
```

---

## Start API Server

```
uvicorn api.server:app --reload
```

API will run at:

```
http://127.0.0.1:8000
```

---

## Run Dashboard

```
streamlit run dashboard/app.py
```

---

# Example Use Case

Suppose a company wants to **automatically analyze customer reviews**.

This platform can:

1. Train multiple AI models  
2. Compare their performance  
3. Explain predictions  
4. Deploy the best model through an API  

---

# Demo

🎥 Project Demo Video  

https://drive.google.com/file/d/1tQbttwZXyGUKMYXOaSz_CPS96x1dvC1v/view

---

# Future Improvements

Possible extensions:

• Automated hyperparameter tuning  
• Model drift detection  
• Distributed training  
• Kubernetes deployment  
• CI/CD pipelines for ML systems  

---

# Author

Rahul Giri  
AI / ML Engineer  
Mumbai, India

GitHub: https://github.com/RahulG-12

---

# License

This project is intended for **educational and research purposes**.
