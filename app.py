import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit app
st.title("🌸 Iris Flower Classifier")
st.write("Masukkan fitur bunga iris untuk memprediksi spesiesnya:")

# Input user
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediksi
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Output
st.subheader("Hasil Prediksi")
st.write(f"Spesies: **{iris.target_names[prediction[0]]}**")
st.subheader("Probabilitas:")
for i, prob in enumerate(prediction_proba[0]):
    st.write(f"{iris.target_names[i]}: {prob:.2f}")

