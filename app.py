import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Network Intrusion Detection System",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("Network Intrusion Detection System")

st.write(
    "Machine Learning Based Intrusion Detection Dashboard"
)

# =========================
# LOAD MODEL
# =========================

model_path = "nids_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found")
    st.stop()

if not os.path.exists(scaler_path):
    st.error("Scaler file not found")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# =========================
# SIDEBAR
# =========================

st.sidebar.header("System Information")

st.sidebar.write("Algorithms Used:")
st.sidebar.write("• Random Forest")
st.sidebar.write("• Decision Tree")
st.sidebar.write("• KNN")
st.sidebar.write("• SVM")
st.sidebar.write("• XGBoost")

st.sidebar.write("---")

st.sidebar.success("System Status: ACTIVE")

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =========================
# PROCESS FILE
# =========================

if uploaded_file is not None:

    try:

        # Read CSV
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.dataframe(data.head())

        # Dataset information
        st.subheader("Dataset Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rows",
            data.shape[0]
        )

        col2.metric(
            "Columns",
            data.shape[1]
        )

        col3.metric(
            "Missing Values",
            data.isnull().sum().sum()
        )

        # =========================
        # SCALE DATA
        # =========================

        scaled_data = scaler.transform(data)

        # =========================
        # PREDICTION
        # =========================

        predictions = model.predict(scaled_data)

        # Add prediction column
        data["Prediction"] = predictions

        # Convert numeric values
        data["Prediction"] = data["Prediction"].map({
            0: "Attack",
            1: "Normal"
        })

        # =========================
        # RESULTS
        # =========================

        st.subheader("Prediction Results")

        st.dataframe(data.head(20))

        # =========================
        # COUNTS
        # =========================

        attack_count = (
            data["Prediction"] == "Attack"
        ).sum()

        normal_count = (
            data["Prediction"] == "Normal"
        ).sum()

        total_packets = len(data)

        attack_percentage = (
            attack_count / total_packets
        ) * 100

        normal_percentage = (
            normal_count / total_packets
        ) * 100

        # =========================
        # METRICS
        # =========================

        st.subheader("Traffic Analysis")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Records",
            total_packets
        )

        col2.metric(
            "Normal Traffic",
            normal_count
        )

        col3.metric(
            "Attack Traffic",
            attack_count
        )

        col4.metric(
            "Attack %",
            f"{attack_percentage:.2f}%"
        )

        # =========================
        # ALERT SYSTEM
        # =========================

        st.subheader("Security Alert")

        if attack_count > 0:

            st.error(
                "Warning! Malicious Traffic Detected"
            )

        else:

            st.success(
                "No Intrusion Detected"
            )

        # =========================
        # PIE CHART
        # =========================

        st.subheader("Traffic Distribution")

        labels = ["Normal", "Attack"]

        values = [
            normal_count,
            attack_count
        ]

        fig1, ax1 = plt.subplots()

        ax1.pie(
            values,
            labels=labels,
            autopct="%1.1f%%"
        )

        st.pyplot(fig1)

        # =========================
        # BAR CHART
        # =========================

        st.subheader("Attack Analysis Chart")

        fig2, ax2 = plt.subplots()

        ax2.bar(
            labels,
            values
        )

        ax2.set_xlabel("Traffic Type")
        ax2.set_ylabel("Count")

        st.pyplot(fig2)

        # =========================
        # LINE GRAPH
        # =========================

        st.subheader("Traffic Monitoring Graph")

        numeric_data = np.arange(total_packets)

        prediction_values = []

        for value in data["Prediction"]:

            if value == "Attack":
                prediction_values.append(1)
            else:
                prediction_values.append(0)

        fig3, ax3 = plt.subplots()

        ax3.plot(
            numeric_data,
            prediction_values
        )

        ax3.set_xlabel("Packet Number")
        ax3.set_ylabel("Traffic Status")

        st.pyplot(fig3)

        # =========================
        # DOWNLOAD RESULTS
        # =========================

        st.subheader("Download Results")

        csv = data.to_csv(
            index=False
        )

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Error: {e}")

# =========================
# FOOTER
# =========================

st.write("---")

st.write(
    "Developed using Machine Learning and Streamlit"
)
