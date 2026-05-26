import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
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
# MODEL PATHS
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

    st.error(
        f"Missing model file: {model_path}"
    )

    st.stop()

if not os.path.exists(scaler_path):

    st.error(
        f"Missing scaler file: {scaler_path}"
    )

    st.stop()

if not os.path.exists(features_path):

    st.error(
        f"Missing feature file: {features_path}"
    )

    st.stop()

# =========================
# LOAD SAVED FILES
# =========================

saved_model = joblib.load(
    model_path
)

scaler = joblib.load(
    scaler_path
)

feature_names = joblib.load(
    features_path
)

# =========================
# SIDEBAR
# =========================

st.sidebar.header(
    "System Information"
)

st.sidebar.write(
    "Algorithms Included"
)

st.sidebar.write("• Random Forest")
st.sidebar.write("• Decision Tree")
st.sidebar.write("• KNN")
st.sidebar.write("• SVM")
st.sidebar.write("• XGBoost")

st.sidebar.success(
    "System Status: ACTIVE"
)

# =========================
# FILE UPLOADER
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

            actual_labels = data["label"]

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

                actual_labels = data["label"]

                data.drop(
                    "label",
                    axis=1,
                    inplace=True
                )

            else:

                actual_labels = None

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
        # SAVE ORIGINAL DATA
        # =========================

        original_data = data.copy()

        # =========================
        # HANDLE OBJECT COLUMNS
        # =========================

        for col in data.columns:

            data[col] = data[col].astype(str)

            data = data[
                data[col] != col
            ]

            # Handle IP columns

            if "ip" in col.lower():

                data[col] = data[col].apply(

                    lambda x: sum(

                        [
                            int(i)

                            for i in x.split(".")

                            if i.isdigit()
                        ]

                    ) if "." in x else 0
                )

            else:

                data[col] = pd.factorize(
                    data[col]
                )[0]

            data[col] = pd.to_numeric(

                data[col],

                errors="coerce"
            )

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
        # PREDICTION
        # =========================

        predictions = saved_model.predict(
            scaled_data
        )

        result_data = original_data.copy()

        # =========================
        # DECODE LABELS
        # =========================

        if dataset_type == "NSL KDD":

            label_encoder = joblib.load(
                "nsl_label_encoder.pkl"
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
        # TRAFFIC ANALYSIS
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
                predictions == 1
            ).sum()

            normal_count = (
                predictions == 0
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
        # ALGORITHM COMPARISON
        # =========================

        st.subheader(
            "Algorithm Comparison"
        )

        if actual_labels is not None:

            if dataset_type == "NSL KDD":

                label_encoder = joblib.load(
                    "nsl_label_encoder.pkl"
                )

                actual_labels = actual_labels.fillna(
                    "normal"
                )

                y_true = label_encoder.fit_transform(
                    actual_labels
                )

            else:

                y_true = pd.factorize(
                    actual_labels
                )[0]

            algorithms = {

                "Random Forest":
                RandomForestClassifier(),

                "Decision Tree":
                DecisionTreeClassifier(),

                "KNN":
                KNeighborsClassifier(),

                "SVM":
                SVC(),

                "XGBoost":
                XGBClassifier(
                    eval_metric="logloss"
                )
            }

            comparison_results = []

            for name, algo in algorithms.items():

                algo.fit(
                    scaled_data,
                    y_true
                )

                pred = algo.predict(
                    scaled_data
                )

                accuracy = accuracy_score(
                    y_true,
                    pred
                )

                precision = precision_score(
                    y_true,
                    pred,
                    average="weighted"
                )

                recall = recall_score(
                    y_true,
                    pred,
                    average="weighted"
                )

                f1 = f1_score(
                    y_true,
                    pred,
                    average="weighted"
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

            st.dataframe(
                comparison_df
            )

            # =========================
            # ACCURACY GRAPH
            # =========================

            st.subheader(
                "Accuracy Comparison"
            )

            fig4, ax4 = plt.subplots()

            ax4.bar(

                comparison_df["Algorithm"],

                comparison_df["Accuracy"]
            )

            ax4.set_xlabel(
                "Algorithms"
            )

            ax4.set_ylabel(
                "Accuracy"
            )

            plt.xticks(rotation=20)

            st.pyplot(fig4)

        else:

            st.warning(
                "No label column found for comparison"
            )

        # =========================
        # GENERATE REPORT
        # =========================

        st.subheader(
            "Download Security Report"
        )

        report_lines = []

        report_lines.append(
            "HYBRID NETWORK INTRUSION DETECTION SYSTEM REPORT"
        )

        report_lines.append(
            "=" * 60
        )

        report_lines.append("")

        report_lines.append(
            f"Dataset Type : {dataset_type}"
        )

        report_lines.append(
            f"Total Records Analyzed : {total_records}"
        )

        report_lines.append(
            f"Normal Traffic Count : {normal_count}"
        )

        report_lines.append(
            f"Attack Traffic Count : {attack_count}"
        )

        report_lines.append(
            f"Attack Percentage : {attack_percentage:.2f}%"
        )

        report_lines.append("")

        if attack_count > 0:

            report_lines.append(
                "SECURITY STATUS : INTRUSION DETECTED"
            )

        else:

            report_lines.append(
                "SECURITY STATUS : NETWORK SAFE"
            )

        report_lines.append("")

        report_lines.append(
            "=" * 60
        )

        report_lines.append(
            "ATTACK ANALYSIS"
        )

        report_lines.append(
            "=" * 60
        )

        report_lines.append("")

        if dataset_type == "NSL KDD":

            attack_details = result_data[
                result_data["Prediction"] != "Normal"
            ]

        else:

            attack_details = result_data[
                result_data["Prediction"] == 1
            ]

        if len(attack_details) > 0:

            report_lines.append(
                f"Total Suspicious Records : {len(attack_details)}"
            )

            report_lines.append("")

            report_lines.append(
                "Sample Attack Records:"
            )

            report_lines.append("")

            for i in range(
                min(5, len(attack_details))
            ):

                report_lines.append(
                    f"Attack Record {i+1}"
                )

                if dataset_type == "NSL KDD":

                    report_lines.append(
                        f"Attack Type : {attack_details.iloc[i]['Prediction']}"
                    )

                else:

                    report_lines.append(
                        "Attack Type : Malicious Traffic"
                    )

                report_lines.append(
                    "-" * 40
                )

        else:

            report_lines.append(
                "No suspicious activity detected."
            )

        report_lines.append("")

        report_lines.append(
            "=" * 60
        )

        report_lines.append(
            "ALGORITHM PERFORMANCE"
        )

        report_lines.append(
            "=" * 60
        )

        report_lines.append("")

        if actual_labels is not None:

            for index, row in comparison_df.iterrows():

                report_lines.append(
                    f"Algorithm : {row['Algorithm']}"
                )

                report_lines.append(
                    f"Accuracy : {row['Accuracy']:.4f}"
                )

                report_lines.append(
                    f"Precision : {row['Precision']:.4f}"
                )

                report_lines.append(
                    f"Recall : {row['Recall']:.4f}"
                )

                report_lines.append(
                    f"F1 Score : {row['F1 Score']:.4f}"
                )

                report_lines.append(
                    "-" * 40
                )

        report_lines.append("")

        report_lines.append(
            "=" * 60
        )

        report_lines.append(
            "RECOMMENDATIONS"
        )

        report_lines.append(
            "=" * 60
        )

        report_lines.append("")

        if attack_count > 0:

            report_lines.append(
                "• Investigate suspicious traffic immediately"
            )

            report_lines.append(
                "• Enable firewall protection"
            )

            report_lines.append(
                "• Monitor abnormal packet activity"
            )

            report_lines.append(
                "• Update IDS rules regularly"
            )

        else:

            report_lines.append(
                "• Network traffic appears safe"
            )

            report_lines.append(
                "• Continue regular monitoring"
            )

        report_lines.append("")

        report_lines.append(
            "=" * 60
        )

        report_lines.append(
            "Generated By Hybrid Network Intrusion Detection System"
        )

        report_text = "\n".join(
            report_lines
        )

        # =========================
        # REPORT PREVIEW
        # =========================

        st.text_area(

            "Security Report Preview",

            report_text,

            height=400
        )

        # =========================
        # DOWNLOAD BUTTON
        # =========================

        st.download_button(

            label="Download Security Report",

            data=report_text,

            file_name="security_report.txt",

            mime="text/plain"
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
