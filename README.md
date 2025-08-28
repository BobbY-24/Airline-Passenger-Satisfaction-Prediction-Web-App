
Web address: http://128.61.11.128:8501

Airline Passenger Satisfaction Prediction Web App
Project Overview
This project is an interactive web application that predicts airline passenger satisfaction based on various flight and service attributes. Users can input passenger details and receive an immediate prediction of whether the passenger is likely to be satisfied or dissatisfied. The app also provides interactive visualizations to explore patterns in the dataset.
The project uses Python, pandas, scikit-learn, and Streamlit for building the interactive web interface.

Dataset
Source: Kaggle – Airline Passenger Satisfaction


Description: The dataset contains information about passengers’ flight experiences, including demographics, flight distance, delays, and various service ratings.


Columns: Includes Age, Flight Distance, Customer Type, Type of Travel, Class, Inflight wifi service, Seat comfort, Food and drink, Inflight entertainment, Gate location, Baggage handling, Online boarding, Leg room service, Ease of Online booking, On-board service, Checkin service, Cleanliness, Inflight service, Departure/Arrival time convenient, Gender, and satisfaction.



Features
Interactive Visualization


Explore passenger satisfaction by numeric or categorical features.


Dynamic bar charts and histograms powered by Plotly.


Prediction Interface


Users can input passenger details via sliders and dropdowns.


Predicts passenger satisfaction using a Random Forest Classifier.


Shows result as Satisfied or Dissatisfied.


Machine Learning Pipeline


Preprocessing: One-hot encoding for categorical features and passthrough for numeric features.


Model: Random Forest Classifier trained on 80% of the dataset.


Evaluation: Displays classification report and confusion matrix.



Installation
Clone the repository:


git clone <your-repo-url>
cd <repo-folder>

Create a Python environment and install dependencies:


conda create -n airline_env python=3.10
conda activate airline_env
pip install -r requirements.txt

Install Streamlit if not included:


pip install streamlit


Usage
Run the Streamlit app:


streamlit run airline_satisfaction_app.py

A browser window will open showing:


Sliders for numeric passenger attributes.


Dropdowns for categorical attributes.


Predict button to see satisfaction prediction.


Interactively explore the dataset and visualize passenger satisfaction trends.



Dependencies
Python 3.10+


pandas


numpy


scikit-learn


matplotlib


seaborn


plotly


ipywidgets


streamlit



Example Prediction
Input a new passenger:
Age: 35


Flight Distance: 1200


Inflight Wifi Service: 4


Seat Comfort: 5


Customer Type: Loyal


Class: Economy


Prediction: 😊 Satisfied

