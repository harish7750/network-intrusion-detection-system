import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Hybrid Network IDS",
    layout="wide"
)

st.title(
    "Hybrid Network Intrusion Detection System"
)

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

    model = joblib.load(
        "nsl_model.pkl"
    )

    scaler = joblib.load(
        "nsl_scaler.pkl"
    )

    feature_names = joblib.load(
        "nsl_features.pkl"
    )

else:

    model = joblib.load(
        "ton_model.pkl"
    )

    scaler = joblib.load(
        "ton_scaler.pkl"
    )

    feature_names = joblib.load(
        "ton_features.pkl"
    )

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "txt"]
)

# =========================
# PROCESS FILE
# =========================

if uploaded_file is not None:

    try:

        if dataset_type == "NSL KDD":

            data = pd.read_csv(
                uploaded_file,
                names=feature_names + ["label", "difficulty"]
            )

            data.drop(
                ["label", "difficulty"],
                axis=1,
                inplace=True
            )

        else:

            data = pd.read_csv(
                uploaded_file
            )

            if "label" in data.columns:

                data.drop(
                    "label",
                    axis=1,
                    inplace=True
                )

        # Encode object columns
        for col in data.columns:

            if data[col].dtype == "object":

                data[col] = pd.factorize(
                    data[col]
                )[0]

        # Match features
        data = data.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Scale
        scaled_data = scaler.transform(
            data
        )

        # Predict
        predictions = model.predict(
            scaled_data
        )

        result_data = data.copy()

        result_data["Prediction"] = predictions

        result_data["Prediction"] = result_data[
            "Prediction"
        ].map({
            0: "Attack",
            1: "Normal"
        })

        st.subheader("Prediction Results")

        st.dataframe(
            result_data.head(20)
        )

        attack_count = (
            result_data["Prediction"] == "Attack"
        ).sum()

        normal_count = (
            result_data["Prediction"] == "Normal"
        ).sum()

        col1, col2 = st.columns(2)

        col1.metric(
            "Normal Traffic",
            normal_count
        )

        col2.metric(
            "Attack Traffic",
            attack_count
        )

        if attack_count > 0:

            st.error(
                "Intrusion Detected"
            )

        else:

            st.success(
                "Network Safe"
            )

    except Exception as e:

        st.error(e)
