# main.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

data = load_data()

# Sidebar for model selection and parameters
st.sidebar.title("Diabetes Prediction App")
model_type = st.sidebar.selectbox("Choose Model", ("Random Forest", "Support Vector Machine"))
split_size = st.sidebar.slider("Training Set Size (%)", 50, 90, 70)

# Show data preview
if st.sidebar.checkbox("Show Dataset"):
    st.write(data)

# Preprocess data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_size) / 100, random_state=42)

# Model training
if model_type == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 100, 50)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
else:
    c_value = st.sidebar.slider("SVM Regularization (C)", 0.1, 10.0, 1.0)
    model = SVC(C=c_value, kernel="linear", probability=True)

# Train and evaluate model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
st.write(f"Model Accuracy: {accuracy:.2f}")

# Predict user input
st.header("Predict Diabetes")
user_data = {}
for column in X.columns:
    user_data[column] = st.number_input(f"Input {column}", min_value=float(X[column].min()), max_value=float(X[column].max()))

input_data = pd.DataFrame([user_data])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    st.write("Diabetes Prediction:", "Positive" if prediction == 1 else "Negative")
    st.write("Prediction Probability:", f"{prediction_proba:.2f}")

# Save and load model (optional)
if st.sidebar.button("Save Model"):
    joblib.dump(model, f"{model_type}_diabetes_model.pkl")
    st.sidebar.write("Model saved successfully!")
