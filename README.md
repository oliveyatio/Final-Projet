# Flight Delay Prediction Project

#  Introduction
This project aims to predict flight delays based on a combination of flight data and weather conditions. The model leverages machine learning techniques such as SVM (Support Vector Machine), KNN (K-Nearest Neighbors), and Logistic Regression, combined using a stacking ensemble method for improved performance.

# Dataset
The dataset used for this project includes:

Flight Data: Collected from the OpenSky API, which provides detailed flight information (departure, arrival airports, timestamps, etc.).
Weather Data: Includes weather conditions such as temperature, humidity, and descriptions (e.g., "Rain, Overcast").

Weather Data: Includes weather conditions such as temperature, humidity, and descriptions (e.g., "Rain, Overcast").

Features Used:
Flight-related features:
icao24: Aircraft unique identifier
departure_airport, arrival_airport: Airport codes for departures and arrivals
departure_time, arrival_time: Timestamps for flight departure and arrival
flight_duration: Calculated flight duration based on timestamps

Weather-related features:
tempmax, tempmin, humidity: Maximum and minimum temperatures, humidity at departure and arrival
conditions: Descriptions of the weather (e.g., "Rain, Overcast")

Target:
delayed: Whether the flight was delayed or not (binary classification).

# Models
Three machine learning models were trained and evaluated for this task:

SVM (Support Vector Machine)
KNN (K-Nearest Neighbors)
Logistic Regression

# Ensemble learning Models
Additionally, a stacking ensemble was used to combine the strengths of these models and create a meta-model for improved performance.

# Results
The best-performing model was the STACKING model, achieving an accuracy of 99.94%. It considered a robust solution due to its ability to leverage multiple models.

           
Model	             Accuracy
SVM  	             98.93%
KNN	               99.01%
SVM OPTIMIZE       99.93%
Logistic Regression	99.93%
Stacking Ensemble	  99.94%


# How to Run the Project
Requirements
  . Python 3.x
  . Required libraries are listed in requirements.txt.

Steps:

1) Clone the repository:

git clone https://github.com/yourusername/Final-Projet.git
cd Final-Projet

2) Install dependencies:
pip install -r requirements.txt

3) Run the notebook or the Python script to train and test models:
python final_project.ipynb


# Model Saving
All models (SVM, KNN, Logistic Regression, stacking) and the scaler were saved using joblib. You can find the saved models in the models/ folder:

svm_model_final.pkl
optimized_svm_model.pkl
final_knn_model.pkl
logistic regression_model_final.pkl
stacking_model.pkl


# Future Improvements
Feature Engineering: More advanced features such as flight routes, airline, and specific weather parameters could improve the model's accuracy.
Model Optimization: Hyperparameter tuning can be further explored to improve the stacking model's performance.
Real-time Deployment: Deploy the model as an API to provide real-time predictions based on live flight data.


