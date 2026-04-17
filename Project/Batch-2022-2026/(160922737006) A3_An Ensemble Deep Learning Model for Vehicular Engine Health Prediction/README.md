# Vehicle Engine Health Prediction System

## Overview

This project focuses on predictive maintenance in the automotive domain with the aim of improving engine reliability, safety, and cost efficiency.

Modern vehicle engines generate complex and high-dimensional sensor data, which makes early detection of faults challenging. This system applies machine learning and deep learning techniques to analyze engine data and predict potential failures in advance.

---
## TEAM MEMBERS
| Name                        | Roll Number  |
|-----------------------------|--------------|
| Mohammed Osman Abdul Ghani  | 160922737006 |
| Md Ghouse                   | 160922737037 |
| Shaik Ahsan Uddin           | 160922737056 |

## Objectives

* To predict engine health using data-driven models
* To identify potential faults at an early stage
* To reduce maintenance costs
* To enhance vehicle safety

---

## Technologies Used

* Python
* Flask (for web application development)
* Machine Learning algorithms
* Deep Learning techniques

---

## Models Used

The system uses an ensemble approach involving multiple models:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* Deep Learning models

---

## Data Preprocessing

The dataset was preprocessed using the following steps:

* Removal of duplicate records
* Handling missing values using median imputation
* Feature scaling
* Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)

---

## Evaluation Metrics

The performance of the models was evaluated using:

* Accuracy
* Precision
* Root Mean Square Error (RMSE)
* Mean Absolute Error (MAE)
* Confusion Matrix
* Area Under the Curve (AUC)

---

## Results

Among all the models, the Random Forest Classifier achieved the best performance with a balanced accuracy of 71%. It demonstrated reliable prediction capability under different engine operating conditions.

---

## Application

The system is deployed as a Flask-based web application, which allows users to:

* Perform real-time engine health prediction
* Receive early warnings of potential engine faults

---

## Project Structure

```
VehicleEngineHealthPrediction/
│
├── app.py
├── model_training.py
├── train_ensemble.py
├── train_model_dl.py
├── train_model_xgb.py
├── requirements.txt
├── engine_data.xlsx
│
├── models/
├── static/
├── templates/
├── xgb_outputs/
├── best_stacked_outputs/
```

---

## Future Scope

* Integration with real-time IoT-based sensor systems
* Improving model performance using larger datasets
* Deployment on cloud platforms
* Development of a mobile-based interface

---

## Author

* Mohammed Osman Abdul Ghani (160922737006)
* Md Ghouse (160922737037)
* Shaik  Ahsan Uddin (160922737056)

---

## Conclusion

This project presents a practical approach to predictive maintenance using machine learning techniques. It helps in reducing maintenance costs and improving the overall safety and efficiency of vehicle engines.
