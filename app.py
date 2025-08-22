import streamlit as st
import joblib
import numpy as np

# ------------------------------
# Load trained KMeans model
# ------------------------------
model = joblib.load("model.pkl")

st.title("Wholesale Customers Clustering App ")
st.write("Enter customer spending data to predict the cluster.")

# Input fields for the 6 features
fresh = st.number_input("Fresh", value=0.0)
milk = st.number_input("Milk", value=0.0)
grocery = st.number_input("Grocery", value=0.0)
frozen = st.number_input("Frozen", value=0.0)
detergents = st.number_input("Detergents_Paper", value=0.0)
delicassen = st.number_input("Delicassen", value=0.0)

if st.button("Predict Cluster"):
    # Make a prediction
    X = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])
    cluster = model.predict(X)
    st.success(f"Predicted Cluster: {cluster[0]}")
