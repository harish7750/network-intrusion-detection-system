import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

# =========================
# LOAD NSL KDD DATASET
# =========================

columns = [
    "duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot",
    "num_failed_logins","logged_in","num_compromised","root_shell",
    "su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

data = pd.read_csv(
    "dataset/KDDTrain+.txt",
    names=columns
)

# =========================
# DROP UNUSED COLUMN
# =========================

data.drop(
    "difficulty",
    axis=1,
    inplace=True
)

# =========================
# MULTI ATTACK CLASSIFICATION
# =========================

attack_mapping = {

    "normal": "Normal",

    "neptune": "DoS",
    "back": "DoS",
    "land": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",

    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",

    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",

    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R"
}

data["label"] = data["label"].map(
    attack_mapping
)

# Remove rows with unknown labels
data.dropna(inplace=True)

# =========================
# ENCODE CATEGORICAL FEATURES
# =========================

encoder = LabelEncoder()

categorical_columns = [
    "protocol_type",
    "service",
    "flag"
]

for col in categorical_columns:

    data[col] = encoder.fit_transform(
        data[col]
    )

# =========================
# FEATURES AND LABELS
# =========================

X = data.drop(
    "label",
    axis=1
)

# Save feature names
joblib.dump(
    X.columns.tolist(),
    "model/nsl_features.pkl"
)

# Encode labels
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    data["label"]
)

# Save label encoder
joblib.dump(
    label_encoder,
    "model/nsl_label_encoder.pkl"
)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# Save scaler
joblib.dump(
    scaler,
    "model/nsl_scaler.pkl"
)

# =========================
# MODELS
# =========================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "SVM": SVC(
        kernel="rbf"
    ),

    "XGBoost": XGBClassifier(
        eval_metric="mlogloss"
    )
}

# =========================
# TRAIN MODELS
# =========================

results = []

best_accuracy = 0
best_model = None

for name, model in models.items():

    print("\n========================")
    print("Training:", name)

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predict
    y_pred = model.predict(
        X_test
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # Store results
    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    # Print metrics
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # Classification report
    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # Confusion matrix
    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # Save best model
    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model

# =========================
# SAVE BEST MODEL
# =========================

joblib.dump(
    best_model,
    "model/nsl_model.pkl"
)

print("\nBest model saved successfully")

# =========================
# RESULTS TABLE
# =========================

results_df = pd.DataFrame(
    results,
    columns=[
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n========================")
print("ALGORITHM COMPARISON")
print("========================")

print(results_df)

# =========================
# ACCURACY GRAPH
# =========================

plt.figure(figsize=(10,5))

plt.bar(
    results_df["Algorithm"],
    results_df["Accuracy"]
)

plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Algorithm Accuracy Comparison")

plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================

if hasattr(best_model, "feature_importances_"):

    importance = best_model.feature_importances_

    feature_importance_df = pd.DataFrame({

        "Feature": X.columns,
        "Importance": importance

    })

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop Important Features")

    print(
        feature_importance_df.head(10)
    )

    plt.figure(figsize=(12,6))

    plt.bar(
        feature_importance_df["Feature"][:10],
        feature_importance_df["Importance"][:10]
    )

    plt.xticks(rotation=45)

    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.title(
        "Top 10 Important Features"
    )

    plt.show()