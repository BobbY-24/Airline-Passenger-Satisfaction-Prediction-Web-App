# Airline Passenger Satisfaction Prediction Web App

## Overview
I built a Streamlit web app that predicts airline passenger satisfaction from passenger, travel, delay, and service-rating inputs. The app trains a Random Forest classifier from the local dataset and provides an interactive form for testing passenger profiles.

## Motivation
I use this repo to show how I can turn a notebook-based ML workflow into a small interactive application. It is one of the better public-facing projects because it demonstrates modeling, preprocessing, and user-facing deployment practice.

## Dataset
- **Source:** Kaggle Airline Passenger Satisfaction dataset.
- **File:** `data/airline_satisfaction.csv`
- **Target variable:** `satisfaction`.
- **Important features:** age, flight distance, travel type, customer type, class, delays, and service ratings.
- **Known limitations:** The app retrains on startup and is intended for demonstration, not production.

## Methods
- I load and clean the airline satisfaction dataset.
- I convert satisfaction labels to binary values.
- I one-hot encode categorical variables.
- I train a Random Forest classifier.
- I use Streamlit widgets to collect passenger inputs and display predictions.

## Results
I do not report a separate benchmark metric in this app repo. The modeling results are documented more clearly in my airline satisfaction notebook repositories, while this repo focuses on the interactive application layer.

## Key Insights
- A small Streamlit interface makes an ML model easier to inspect.
- Default values are used for fields not exposed in the form.
- This app is a demo layer for the broader airline satisfaction modeling work.

## Limitations
- The model retrains each time the app starts.
- Several model features are filled with default values in the UI.
- The app does not save model artifacts or track model versions.

## How to Run
```bash
git clone https://github.com/BobbY-24/Airline-Passenger-Satisfaction-Prediction-Web-App.git
cd Airline-Passenger-Satisfaction-Prediction-Web-App
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
