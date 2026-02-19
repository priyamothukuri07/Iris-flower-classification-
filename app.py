import streamlit as st
import numpy as np
import joblib

st.title("Iris Flower Prediction")

# load model
model = joblib.load("iris_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# inputs
sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):
    data = np.array([[sl, sw, pl, pw]])
    pred = model.predict(data)
    result = encoder.inverse_transform(pred)
    st.success(result[0])
