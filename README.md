# Airline Passenger Satisfaction Prediction Web App

## Overview
This project is a Streamlit web app for predicting airline passenger satisfaction from passenger, travel, delay, and service-rating inputs. The app trains a Random Forest classifier from the local dataset and provides an interactive form for trying passenger profiles.

## Motivation
This repo demonstrates how a notebook-based machine learning workflow can be translated into a small interactive application. It is useful for communicating model behavior to nontechnical users and for practicing lightweight ML app deployment.

## Dataset
- **Source:** Kaggle Airline Passenger Satisfaction dataset.
- **File:** `data/airline_satisfaction.csv`
- **Target variable:** `satisfaction`.
- **Important features:** age, flight distance, travel type, customer type, class, delays, and service ratings.
- **Dataset size:** TODO: add dataset size after rerunning app/notebook.
- **Known limitations:** The app retrains on startup and is intended for demonstration, not production.

## Methods
- Load and clean the airline satisfaction dataset.
- Convert satisfaction labels to binary values.
- One-hot encode categorical variables.
- Train a Random Forest classifier.
- Use Streamlit widgets to collect passenger inputs and display predictions.

## Results
TODO: add metric after rerunning notebook.

## Key Insights
- A small Streamlit interface makes an ML model easier to inspect.
- Default values are used for fields not exposed in the form.
- This app is a demo layer for the broader airline satisfaction modeling work.

## Limitations
- The model retrains each time the app starts.
- Several model features are filled with default values in the UI.
- The app does not save model artifacts or track model versions.
- It should not be treated as an operational prediction system.

## Future Improvements
- Save and load a trained model artifact.
- Expose all important model features in the UI.
- Add model metrics and feature importance to the app.
- Add a deployment guide.

## How to Run
```bash
git clone https://github.com/BobbY-24/Airline-Passenger-Satisfaction-Prediction-Web-App.git
cd Airline-Passenger-Satisfaction-Prediction-Web-App
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
