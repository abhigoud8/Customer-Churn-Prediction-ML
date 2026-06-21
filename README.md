# Customer Churn Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)

## Overview
An end-to-end machine learning project that predicts whether a customer will churn based on historical behavioral data. Helps businesses proactively retain customers and reduce revenue loss.

## Features
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and selection
- Multiple ML models: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning with GridSearchCV
- Model evaluation: ROC-AUC, Confusion Matrix, Classification Report

## Tech Stack
- **Python** — pandas, numpy, matplotlib, seaborn
- **ML** — scikit-learn, XGBoost
- **Visualization** — Plotly, matplotlib

## Project Structure
```
Customer-Churn-Prediction-ML/
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks for EDA & modeling
├── src/
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Model training
│   └── evaluate.py        # Model evaluation
├── requirements.txt
└── README.md
```

## Results
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 80.2% | 0.84 |
| Random Forest | 85.6% | 0.91 |
| XGBoost | **87.3%** | **0.93** |

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
```