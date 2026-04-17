# Big Data ML-Based Fake News Detection Using Distributed Learning
# Overview

This project focuses on detecting fake news using Big Data technologies, Machine Learning, and Distributed Learning techniques with the aim of reducing misinformation, improving trust in digital media, and enabling scalable real-time classification.

Modern online platforms generate massive volumes of textual data every second. Detecting fake news manually is difficult due to the volume, velocity, and variety of data. This system applies machine learning and deep learning models in a distributed environment to classify news articles as real or fake efficiently.

# TEAM MEMBERS
|Name                 |	Roll Number|
|Shaik Aasim          |160922737044|
|Mohammed Abdul Moyeed|160922737014|
|Shaik Addeeb Ahmed   |160922737015|
# Objectives
To detect fake news using data-driven models
To classify news articles as real or fake
To reduce the spread of misinformation
To improve trust in online news platforms
To enable scalable big data processing
# Technologies Used
Python
Flask (for web application development)
Apache Spark
Hadoop
Machine Learning algorithms
Deep Learning techniques
NLP (Natural Language Processing)
# Models Used

The system uses multiple machine learning and deep learning models:

Logistic Regression
Support Vector Machine (SVM)
Random Forest
XGBoost
Naive Bayes
LSTM (Long Short-Term Memory)
Ensemble Models
Data Preprocessing

The dataset was preprocessed using the following steps:

Removal of duplicate records
Handling missing values
Tokenization
Stop-word removal
Lemmatization / Stemming
Feature scaling
TF-IDF vectorization
Handling class imbalance
# Evaluation Metrics

The performance of the models was evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC Score
# Results

Among all models, the Random Forest and XGBoost classifiers achieved strong performance with high accuracy and balanced classification results. The LSTM model also showed good capability in understanding text patterns and sequential data.

The distributed learning environment significantly reduced training time and improved scalability for large datasets.

# Application

The system is deployed as a Flask-based web application, which allows users to:

Enter news headlines or article text
Perform real-time fake news prediction
Receive instant credibility results
Assist users in verifying online information

# Project Structure

FakeNewsDetection/
│
├── app.py
├── model_training.py
├── train_lstm.py
├── train_xgb.py
├── train_spark_model.py
├── requirements.txt
├── news_dataset.csv
│
├── models/
├── static/
├── templates/
├── outputs/
├── reports/

# Future Scope

Integration with live social media feeds
Multilingual fake news detection
Explainable AI for predictions
Deployment on cloud platforms
Mobile application development
Improved accuracy using transformer models like BERT

# Author

Shaik Aasim (160922737044)
Mohammed Abdul Moyeed (160922737014)
Shaik Addeeb Ahmed (160922737015)

# Conclusion

This project presents a practical solution for fake news detection using distributed learning
