# airline_satisfaction_app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('/Users/bobyan/Desktop/Kaggle Datasets/airline_satisfaction.csv')

# Preprocessing
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True, errors="ignore")
df['satisfaction'] = df['satisfaction'].replace({
    "satisfied": 1,
    "neutral or dissatisfied": 0
})
df.dropna(inplace=True)

# Define features
numerical_features = df.select_dtypes(include=['number']).drop("satisfaction", axis=1).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

X = df[numerical_features + categorical_features]
y = df["satisfaction"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Streamlit App UI
st.title("Airline Passenger Satisfaction Predictor ✈️")

st.header("Enter Passenger Details:")

# Input widgets
age = st.slider("Age", 10, 80, 35)
flight_distance = st.slider("Flight Distance", 100, 5000, 1200)
dep_delay = st.slider("Departure Delay (minutes)", 0, 600, 15)
arr_delay = st.slider("Arrival Delay (minutes)", 0, 600, 10)

customer_type = st.selectbox("Customer Type", df["Customer Type"].unique())
travel_type = st.selectbox("Type of Travel", df["Type of Travel"].unique())
seat_class = st.selectbox("Class", df["Class"].unique())

wifi = st.slider("Inflight Wifi Service (0-5)", 0, 5, 4)
seat_comfort = st.slider("Seat Comfort (0-5)", 0, 5, 5)
food = st.slider("Food and Drink (0-5)", 0, 5, 3)
entertainment = st.slider("Inflight Entertainment (0-5)", 0, 5, 4)

# Prediction button
if st.button("Predict Satisfaction"):
    # Collect all input features for the model
    new_passenger = pd.DataFrame({
        "Age": [age],
        "Flight Distance": [flight_distance],
        "Departure Delay in Minutes": [dep_delay],
        "Arrival Delay in Minutes": [arr_delay],
        "Inflight wifi service": [wifi],
        "Seat comfort": [seat_comfort],
        "Food and drink": [food],
        "Inflight entertainment": [entertainment],
        "Gate location": [3],  # default value
        "Customer Type": [customer_type],
        "Type of Travel": [travel_type],
        "Class": [seat_class],
        "Departure/Arrival time convenient": [4],  # default
        "Online boarding": [5],  # default
        "Baggage handling": [4],  # default
        "Leg room service": [3],  # default
        "Ease of Online booking": [4],  # default
        "On-board service": [4],  # default
        "Checkin service": [5],  # default
        "Cleanliness": [5],  # default
        "Inflight service": [4],  # default
        "Gender": ["Male"]  # default
    })

    prediction = model.predict(new_passenger)[0]
    result = "😊 Satisfied" if prediction == 1 else "☹️ Dissatisfied"

    st.subheader(f"Prediction: {result}")

