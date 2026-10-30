---
title: Heart Disease Predictor
emoji: ❤️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Heart Disease Predictor

Flask website that predicts heart disease risk from 11 patient measurements.
Five models (Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting) were trained in Google Colab
on the Kaggle *Heart Failure Prediction* dataset and compared; the best one is used here.

Needs `model.pkl` and `metrics.json` (created by the notebook) in the same folder as `app.py`.

For learning only. Not a medical device.
