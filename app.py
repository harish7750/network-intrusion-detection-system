import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("nids_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Network Intrusion Detection System",
    layout="wide"
)

st.title("Network Intrusion Detection System")

st.write("Upload network traffic CSV file for intrusion detection")

uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # Read CSV
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(data.head())

        # Scale
        scaled_data = scaler.transform(data)

        # Predict
        predictions = model.predict(scaled_data)

        # Add predictions
        data["Prediction"] = predictions

        # Convert labels
        data["Prediction"] = data["Prediction"].map({
            0: "Attack",
            1: "Normal"
        })

        st.subheader("Prediction Results")
        st.dataframe(data.head())

        # Count
        attack_count = (
            data["Prediction"] == "Attack"
        ).sum()

        normal_count = (
            data["Prediction"] == "Normal"
        ).sum()

        # Metrics
        col1, col2 = st.columns(2)

        col1.metric(
            "Normal Traffic",
            normal_count
        )

        col2.metric(
            "Attack Traffic",
            attack_count
        )

        # Pie chart
        st.subheader("Traffic Distribution")

        labels = ["Normal", "Attack"]
        values = [normal_count, attack_count]

        fig, ax = plt.subplots()

        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%"
        )

        st.pyplot(fig)

        # Alert
        if attack_count > 0:
            st.error("Warning! Intrusion Detected")
        else:
            st.success("No Intrusion Detected")

    except Exception as e:

        st.error(f"Error: {e}")
