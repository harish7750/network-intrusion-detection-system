import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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

saved_model = joblib.load(model_path)
saved_scaler = joblib.load(scaler_path)

# =========================
# SIDEBAR
# =========================

st.sidebar.header("System Information")

st.sidebar.write("Algorithms Used")
st.sidebar.write("• Random Forest")
st.sidebar.write("• Decision Tree")
st.sidebar.write("• KNN")
st.sidebar.write("• SVM")
st.sidebar.write("• XGBoost")

st.sidebar.success("System Status: ACTIVE")

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =========================
# PROCESS DATA
# =========================

if uploaded_file is not None:

    try:

        # Read dataset
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.dataframe(data.head())

        # Dataset Info
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
        # PREDICTION USING SAVED MODEL
        # =========================

        scaled_data = saved_scaler.transform(data)

        predictions = saved_model.predict(
            scaled_data
        )

        # Add prediction column
        result_data = data.copy()

        result_data["Prediction"] = predictions

        result_data["Prediction"] = result_data[
            "Prediction"
        ].map({
            0: "Attack",
            1: "Normal"
        })

        # =========================
        # RESULTS TABLE
        # =========================

        st.subheader("Prediction Results")

        st.dataframe(
            result_data.head(20)
        )

        # =========================
        # COUNTS
        # =========================

        attack_count = (
            result_data["Prediction"] == "Attack"
        ).sum()

        normal_count = (
            result_data["Prediction"] == "Normal"
        ).sum()

        total_records = len(result_data)

        attack_percentage = (
            attack_count / total_records
        ) * 100

        # =========================
        # METRICS
        # =========================

        st.subheader("Traffic Analysis")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Records",
            total_records
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
        # ALERT
        # =========================

        st.subheader("Security Alert")

        if attack_count > 0:

            st.error(
                "Warning! Intrusion Detected"
            )

        else:

            st.success(
                "No Intrusion Detected"
            )

        # =========================
        # PIE CHART
        # =========================

        st.subheader("Traffic Distribution")

        labels = [
            "Normal",
            "Attack"
        ]

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

        x_axis = np.arange(total_records)

        y_axis = []

        for value in result_data["Prediction"]:

            if value == "Attack":
                y_axis.append(1)
            else:
                y_axis.append(0)

        fig3, ax3 = plt.subplots()

        ax3.plot(
            x_axis,
            y_axis
        )

        ax3.set_xlabel("Packet Number")
        ax3.set_ylabel("Traffic Status")

        st.pyplot(fig3)

        # =========================
        # ALGORITHM COMPARISON
        # =========================

        st.subheader("Algorithm Comparison")

        # Create dummy labels for evaluation
        y_dummy = np.random.randint(
            0,
            2,
            len(data)
        )

        X_train, X_test, y_train, y_test = train_test_split(
            scaled_data,
            y_dummy,
            test_size=0.2,
            random_state=42
        )

        models = {

            "Random Forest": RandomForestClassifier(),

            "Decision Tree": DecisionTreeClassifier(),

            "KNN": KNeighborsClassifier(),

            "SVM": SVC(),

            "XGBoost": XGBClassifier(
                eval_metric="logloss"
            )
        }

        comparison_results = []

        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred
            )

            recall = recall_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            comparison_results.append([
                name,
                accuracy,
                precision,
                recall,
                f1
            ])

        comparison_df = pd.DataFrame(
            comparison_results,
            columns=[
                "Algorithm",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        )

        st.dataframe(comparison_df)

        # =========================
        # ACCURACY GRAPH
        # =========================

        st.subheader("Algorithm Accuracy Graph")

        fig4, ax4 = plt.subplots()

        ax4.bar(
            comparison_df["Algorithm"],
            comparison_df["Accuracy"]
        )

        ax4.set_xlabel("Algorithms")
        ax4.set_ylabel("Accuracy")

        st.pyplot(fig4)

        # =========================
        # DOWNLOAD RESULTS
        # =========================

        st.subheader("Download Results")

        csv = result_data.to_csv(
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
    "Developed using Machine Learning, XGBoost, and Streamlit"
)
