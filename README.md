# Credit-card-fruaud-detection-using-ml-dl-enhanced-with-smote-kfold-and-Explainable_Ai
"This project presents a Hybrid Credit Card Fraud Detection System combining Machine Learning and Deep Learning models to detect fraud. Data preprocessing, scaling, and SMOTE handle imbalance, while K-Fold improves robustness. ML and DL outputs are combined for accuracy. A Flask app enables user input, with XAI explaining key features."
This project presents a Hybrid Credit Card Fraud Detection System that combines Machine Learning (ML) and Deep Learning (DL) techniques to accurately detect fraudulent transactions. The system is designed to handle highly imbalanced financial transaction data and provide interpretable predictions using feature-based Explainable AI (XAI).

The final model is deployed using Flask, allowing users to manually input transaction features and receive fraud prediction results along with probability scores and feature impact analysis.

Dataset Description:

The dataset used in this project is the Credit Card Fraud Detection Dataset available on Kaggle.

🔗 Dataset Source: Kaggle – Credit Card Fraud Detection Dataset

Dataset Details:
The dataset contains transactions made by European cardholders.
It consists of 284,807 transactions.
Among them, only 492 transactions are fraudulent, making it highly imbalanced.
Features include:
Time
Amount
V1 to V28 (PCA-transformed features for confidentiality)
Class (Target Variable)
0 → Normal Transaction
1 → Fraud Transaction

The PCA transformation ensures customer privacy while retaining meaningful patterns for fraud detection.

Model Development (Google Colab)

Model training and evaluation were performed in Google Colab.

🔹 Data Preprocessing Steps:
Removed infinite and missing values
Applied StandardScaler for feature scaling
Handled class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
Split dataset using K-Fold Cross Validation for robust performance evaluation

Model Training:

Two types of models were developed:

Machine Learning Model
Trained using scikit-learn
Best-performing model selected after comparison
Saved as: best_ml_model.pkl
Deep Learning Model
Built using TensorFlow / Keras
Optimized for fraud classification
Saved as: best_dl_model.h5
🔹 Model Comparison:

Different models were evaluated and compared using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC

The best-performing ML and DL models were selected and used in the hybrid system.

🔎 Explainable AI (XAI)

To enhance transparency and interpretability, a Feature-Based Explainable AI module is integrated.

Feature importance is extracted from the ML model.
Contribution scores are calculated for each input feature.
The system displays the Top Influencing Features affecting fraud prediction.
This helps users understand why a transaction is classified as fraud or normal.
🌐 Deployment (Flask Web Application)

The system is deployed using Flask.

Features:
Manual transaction input (Time, V1–V28, Amount)
Fraud / Normal classification
ML Probability
DL Probability
Final Hybrid Probability
Risk Level (Low / Medium / High)
Feature Contribution Analysis

Users interact with the system via a clean web interface built using HTML and CSS.
