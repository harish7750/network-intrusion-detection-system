import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Hybrid Network Intrusion Detection System",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title(
    "Hybrid Network Intrusion Detection System"
)

st.write(
    "Machine Learning Based Intrusion Detection Dashboard"
)

# =========================
# DATASET SELECTION
# =========================

dataset_type = st.selectbox(

    "Select Dataset Type",

    [
        "NSL KDD",
        "TON IoT"
    ]
)

# =========================
# LOAD MODELS
# =========================

if dataset_type == "NSL KDD":

    model_path = "nsl_model.pkl"
    scaler_path = "nsl_scaler.pkl"
    features_path = "nsl_features.pkl"

else:

    model_path = "ton_model.pkl"
    scaler_path = "ton_scaler.pkl"
    features_path = "ton_features.pkl"

# =========================
# CHECK FILES
# =========================

if not os.path.exists(model_path):

    st.error("Model file missing")
    st.stop()

if not os.path.exists(scaler_path):

    st.error("Scaler file missing")
    st.stop()

if not os.path.exists(features_path):

    st.error("Feature file missing")
    st.stop()

# =========================
# LOAD FILES
# =========================

model = joblib.load(model_path)

scaler = joblib.load(scaler_path)

feature_names = joblib.load(features_path)

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

st.sidebar.success(
    "System Status: ACTIVE"
)

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(

    "Upload Dataset File",

    type=["csv", "txt"]
)

# =========================
# PROCESS FILE
# =========================

if uploaded_file is not None:

    try:

        # =========================
        # READ DATASET
        # =========================

        if dataset_type == "NSL KDD":

            columns = feature_names + [
                "label",
                "difficulty"
            ]

            data = pd.read_csv(
                uploaded_file,
                names=columns
            )

            # Remove label columns
            data.drop(
                ["label", "difficulty"],
                axis=1,
                inplace=True
            )

        else:

            data = pd.read_csv(
                uploaded_file
            )

            # Remove label column if exists
            if "label" in data.columns:

                data.drop(
                    "label",
                    axis=1,
                    inplace=True
                )

        # =========================
        # DISPLAY DATA
        # =========================

        st.subheader(
            "Uploaded Dataset"
        )

        st.dataframe(
            data.head()
        )

        # =========================
        # HANDLE OBJECT COLUMNS
        # =========================

        for col in data.columns:

            # Convert IP columns
            if "ip" in col.lower():

                data[col] = data[col].astype(str)

                data[col] = data[col].apply(

                    lambda x: sum(
                        [
                            int(i)
                            for i in x.split(".")
                        ]
                    ) if "." in x else 0
                )

            # Encode other object columns
            elif data[col].dtype == "object":

                data[col] = pd.factorize(
                    data[col]
                )[0]

        # =========================
        # HANDLE NULL VALUES
        # =========================

        data.fillna(
            0,
            inplace=True
        )

        # =========================
        # MATCH FEATURES
        # =========================

        data = data.reindex(
            columns=feature_names,
            fill_value=0
        )

        # =========================
        # SCALE DATA
        # =========================

        scaled_data = scaler.transform(
            data
        )

        # =========================
        # PREDICT
        # =========================

        predictions = model.predict(
            scaled_data
        )

        result_data = data.copy()

        # =========================
        # DECODE LABELS
        # =========================

        if dataset_type == "NSL KDD":

            label_encoder = joblib.load(
                "model/nsl_label_encoder.pkl"
            )

            result_data["Prediction"] = (
                label_encoder.inverse_transform(
                    predictions
                )
            )

        else:

            result_data["Prediction"] = predictions

        # =========================
        # DISPLAY RESULTS
        # =========================

        st.subheader(
            "Prediction Results"
        )

        st.dataframe(
            result_data.head(20)
        )

        # =========================
        # ATTACK COUNTS
        # =========================

        if dataset_type == "NSL KDD":

            attack_count = (
                result_data["Prediction"]
                != "Normal"
            ).sum()

            normal_count = (
                result_data["Prediction"]
                == "Normal"
            ).sum()

        else:

            attack_count = (
                result_data["Prediction"]
                == 1
            ).sum()

            normal_count = (
                result_data["Prediction"]
                == 0
            ).sum()

        total_records = len(
            result_data
        )

        attack_percentage = (
            attack_count / total_records
        ) * 100

        # =========================
        # METRICS
        # =========================

        st.subheader(
            "Traffic Analysis"
        )

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

        st.subheader(
            "Security Alert"
        )

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

        st.subheader(
            "Traffic Distribution"
        )

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
        # BAR GRAPH
        # =========================

        st.subheader(
            "Traffic Analysis Graph"
        )

        fig2, ax2 = plt.subplots()

        ax2.bar(
            labels,
            values
        )

        ax2.set_xlabel(
            "Traffic Type"
        )

        ax2.set_ylabel(
            "Count"
        )

        st.pyplot(fig2)

        # =========================
        # LINE GRAPH
        # =========================

        st.subheader(
            "Packet Monitoring"
        )

        x_axis = np.arange(
            total_records
        )

        y_axis = []

        if dataset_type == "NSL KDD":

            for value in result_data[
                "Prediction"
            ]:

                if value == "Normal":

                    y_axis.append(0)

                else:

                    y_axis.append(1)

        else:

            y_axis = list(
                predictions
            )

        fig3, ax3 = plt.subplots()

        ax3.plot(
            x_axis,
            y_axis
        )

        ax3.set_xlabel(
            "Packet Number"
        )

        ax3.set_ylabel(
            "Traffic Status"
        )

        st.pyplot(fig3)

        # =========================
        # DOWNLOAD RESULTS
        # =========================

        st.subheader(
            "Download Results"
        )

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

        st.error(
            f"Error: {e}"
        )

# =========================
# FOOTER
# =========================

st.write("---")

st.write(
    "Developed using Machine Learning, XGBoost, and Streamlit"
)
