import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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
# LOAD DATASET
# =========================

data = pd.read_csv(
    "dataset/train_test_network.csv"
)

print(data.head())

# =========================
# REMOVE NULL VALUES
# =========================

data.dropna(inplace=True)

# =========================
# DISPLAY COLUMNS
# =========================

print("\nColumns:\n")
print(data.columns)

# =========================
# LABEL COLUMN
# =========================

# Change this if needed
label_column = "label"

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================

encoder = LabelEncoder()

for col in data.columns:

    if data[col].dtype == "object":

        data[col] = encoder.fit_transform(
            data[col]
        )

# =========================
# SPLIT FEATURES AND LABEL
# =========================

X = data.drop(label_column, axis=1)

y = data[label_column]

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

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(
    scaler,
    "model/scaler.pkl"
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
        eval_metric="logloss"
    )
}

# =========================
# TRAIN MODELS
# =========================

results = []

best_accuracy = 0
best_model = None

for name, model in models.items():

    print("\n======================")
    print("Training:", name)

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict
    y_pred = model.predict(X_test)

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

    # Print
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # Save best model
    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model

# =========================
# SAVE BEST MODEL
# =========================

joblib.dump(
    best_model,
    "model/nids_model.pkl"
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

print("\n======================")
print("ALGORITHM COMPARISON")
print("======================")

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
joblib.dump(
    X.columns.tolist(),
    "model/ton_features.pkl"
)